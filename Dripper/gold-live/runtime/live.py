"""The live session process. One process, one session.

Wires the real components together and runs until told to stop:

    market feed -> engine -> MarketState  (shared plane)
    platform adapter -> comment pipeline  (per session)
    director -> generator -> safety -> audio router -> device

Run one of these per session under systemd. A crash takes down one session and
the supervisor restarts it; the other six never notice. That is the entire
argument for process-per-session over threads.

    python -m runtime.live --session SESSION_001
    python -m runtime.live --session SESSION_001 --market synthetic --adapter mock
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from intelligence.content import ContentPlanner, classify_phase, load_content
from intelligence.generation import build_generator
from intelligence.personas import load_personas
from intelligence.proposer import CoverageMemory, TopicProposer
from platform_.audio.router import AudioRouter
from platform_.market.engine import MarketEngine
from platform_.market.feeds import Feed, ReplayFeed, SyntheticFeed
from runtime.health import METRICS, HealthServer, heartbeat
from runtime.session import SessionRuntime
from shared.contracts import (
    AudioRequest,
    HealthCheck,
    HealthState,
    PlatformBinding,
    ServiceHealth,
    SessionState,
    SessionStatus,
)
from shared.store import TraceStore

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("live")


def load_session_config(session_id: str) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "sessions.yaml").read_text(encoding="utf-8"))
    for spec in cfg["sessions"]:
        if spec["session_id"] == session_id:
            return spec
    raise SystemExit(f"{session_id} not found in configs/sessions.yaml")


def build_feed(kind: str, path: str | None) -> Feed:
    if kind == "synthetic":
        return SyntheticFeed(interval_s=0.25)
    if kind == "replay":
        if not path:
            raise SystemExit("--market replay needs --market-path")
        return ReplayFeed(path, speed=0.0)
    if kind == "websocket":
        from platform_.market.feeds import WebSocketFeed

        url = os.environ.get("MARKET_WS_URL")
        if not url:
            raise SystemExit("--market websocket needs MARKET_WS_URL")
        return WebSocketFeed(url)
    raise SystemExit(f"unknown market feed: {kind}")


async def build_adapter(kind: str, session_id: str, salt: str):
    if kind == "mock":
        return None
    if kind == "screen":
        from platform_.adapters.ocr import build_ocr
        from platform_.adapters.screen import ScreenCaptureAdapter
        from shared.contracts import CaptureCalibration

        devices_path = ROOT / "configs" / "devices.json"
        if not devices_path.exists():
            raise SystemExit(
                "No configs/devices.json. Run scripts/calibrate_capture.py "
                f"--session {session_id} on this device first."
            )
        devices = json.loads(devices_path.read_text(encoding="utf-8"))
        entry = next(
            (d for d in devices.values() if d.get("bound_session") == session_id), None
        )
        if entry is None or "capture" not in entry:
            raise SystemExit(f"No capture calibration bound to {session_id}.")

        adapter = ScreenCaptureAdapter(
            session_id=session_id,
            calibration=CaptureCalibration(**entry["capture"]),
            ocr=build_ocr("paddle"),
            author_salt=salt,
        )
        await adapter.connect()
        return adapter
    raise SystemExit(f"unknown adapter: {kind}")


class LiveSession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.session_id = args.session
        self.stopping = asyncio.Event()
        self.runtime: SessionRuntime | None = None
        self.engine = MarketEngine()
        self.router: AudioRouter | None = None
        self.adapter = None
        self.store = TraceStore(args.db)
        self.health_server: HealthServer | None = None
        self._tasks: list[asyncio.Task] = []
        self.started_at = datetime.now(timezone.utc)

    # -- health -----------------------------------------------------------

    def health(self) -> list[ServiceHealth]:
        out: list[ServiceHealth] = []
        now = datetime.now(timezone.utc)

        conf = self.engine.confidence(now)
        market_ok = conf.value in ("live", "delayed")
        out.append(
            ServiceHealth(
                component="market_engine",
                state=HealthState.OK if market_ok else HealthState.DEGRADED,
                checks=[
                    HealthCheck(name="feed", ok=market_ok, detail=f"confidence={conf.value}"),
                    HealthCheck(name="warm", ok=self.engine.warm,
                                detail="detectors have ATR history"),
                ],
                degraded_reason=None if market_ok
                else f"market data {conf.value}; price quoting disabled",
            )
        )

        if self.router is not None:
            snap = self.router.snapshot()
            out.append(
                ServiceHealth(
                    component="audio_router", session_id=self.session_id,
                    state=HealthState.OK if snap["tts_failures"] < 3 else HealthState.DEGRADED,
                    checks=[
                        HealthCheck(name="queue", ok=snap["queue_depth"] < 6,
                                    detail=f"depth={snap['queue_depth']}"),
                        HealthCheck(name="tts", ok=snap["tts_failures"] < 3,
                                    detail=f"failures={snap['tts_failures']}"),
                    ],
                )
            )

        if self.adapter is not None:
            with contextlib.suppress(Exception):
                out.append(asyncio.get_event_loop().run_until_complete(self.adapter.health()))

        if self.runtime is not None:
            out.append(
                ServiceHealth(
                    component="session", session_id=self.session_id,
                    state=HealthState.OK,
                    checks=[
                        HealthCheck(
                            name="director_queue", ok=self.runtime.director.queue_depth < 40,
                            detail=f"depth={self.runtime.director.queue_depth}",
                        ),
                        HealthCheck(
                            name="content", ok=self.runtime.fallback_used == 0,
                            detail=f"scripted fallback used {self.runtime.fallback_used}x",
                        ),
                    ],
                )
            )
        return out

    # -- loops ------------------------------------------------------------

    async def _market_loop(self, feed: Feed) -> None:
        async for tick in feed.ticks():
            if self.stopping.is_set():
                return
            self.engine.on_tick(tick.bid, tick.ask, tick.at)

    async def _bar_loop(self) -> None:
        while not self.stopping.is_set():
            await asyncio.sleep(1.0)
            self.engine.on_bar_close_check(datetime.now(timezone.utc))

    async def _comment_loop(self) -> None:
        if self.adapter is None or self.runtime is None:
            return
        async for comment in self.adapter.comments():
            if self.stopping.is_set():
                return
            METRICS.inc("goldlive_comments_total", {"session": self.session_id})
            await self.runtime.on_comment(comment)

    async def _speak_loop(self) -> None:
        assert self.runtime is not None
        while not self.stopping.is_set():
            await asyncio.sleep(self.args.tick_s)
            now = datetime.now(timezone.utc)
            state = self.engine.snapshot(now)

            METRICS.gauge(
                "goldlive_market_staleness_ms", state.staleness_ms,
                {"session": self.session_id},
            )

            for event in self.engine.drain_events():
                self.runtime.on_market_event(event, now)

            view = state.timeframes.get("5m")
            phase = classify_phase(now, atr=view.atr if view else None)
            await self.runtime.offer_next_topic(phase, state, now)

            before_unsafe = len(self.runtime.dropped_unsafe)
            before_rep = len(self.runtime.dropped_repetitive)

            response = await self.runtime.tick(state, now)

            for _ in range(len(self.runtime.dropped_unsafe) - before_unsafe):
                METRICS.inc("goldlive_blocked_total",
                            {"session": self.session_id, "reason": "safety"})
                text, violations = self.runtime.dropped_unsafe[-1]
                self.store.record_blocked(
                    self.session_id, "safety", text, "; ".join(violations)
                )
            for _ in range(len(self.runtime.dropped_repetitive) - before_rep):
                METRICS.inc("goldlive_blocked_total",
                            {"session": self.session_id, "reason": "repetition"})
                text, sim = self.runtime.dropped_repetitive[-1]
                self.store.record_blocked(
                    self.session_id, "repetition", text, f"similarity={sim:.2f}"
                )

            if response is None:
                continue

            METRICS.inc("goldlive_utterances_total", {"session": self.session_id})
            if response.provenance.first_token_ms:
                METRICS.observe(
                    "goldlive_first_token_ms", response.provenance.first_token_ms,
                    {"session": self.session_id},
                )
            self.store.record_utterance(response)

            if self.router is not None:
                await self.router.submit(
                    AudioRequest(
                        utterance_id=response.utterance_id,
                        session_id=response.session_id,
                        trace_id=response.trace_id,
                        segments=response.segments,
                        voice_id=self.runtime.persona.voice_id,
                        priority=response.trigger.priority,
                    )
                )
                METRICS.gauge(
                    "goldlive_queue_depth", self.router.queue_depth,
                    {"session": self.session_id, "queue": "audio"},
                )

            log.info("[%s] %s", self.session_id, response.text)

    # -- lifecycle --------------------------------------------------------

    async def run(self) -> int:
        spec = load_session_config(self.session_id)
        personas = load_personas(ROOT / "configs" / "personas")
        persona = personas[spec["persona_id"]]

        await self.store.start()

        generator, llm = await build_generator(self.args.mode)
        proposer = (
            TopicProposer(llm, CoverageMemory()) if llm is not None else None
        )
        planner = ContentPlanner(load_content(ROOT / "configs" / "content.yaml"))

        if self.args.tts == "piper":
            from platform_.tts.piper import PiperTTS

            tts = PiperTTS(voices_dir=self.args.voices)
            if not tts.available():
                log.warning("piper binary not found; falling back to file TTS")
                from shared.mocks.tts import FileTTS

                tts = FileTTS()
        else:
            from shared.mocks.tts import FileTTS

            tts = FileTTS()

        self.runtime = SessionRuntime(
            state=SessionState(
                session_id=self.session_id,
                persona_id=spec["persona_id"],
                status=SessionStatus.LIVE,
                device_id=spec.get("device_id"),
                platform_binding=PlatformBinding(
                    platform=spec["platform"], channel_id=spec["channel_id"]
                ),
            ),
            persona=persona,
            generator=generator,
            tts=tts,
            out_dir=Path(self.args.out),
            planner=planner,
            proposer=proposer,
        )

        self.router = AudioRouter(self.session_id, tts, Path(self.args.out))
        await self.router.start()

        self.adapter = await build_adapter(
            self.args.adapter, self.session_id, os.environ.get("AUTHOR_SALT", "change-me")
        )

        self.health_server = HealthServer(self.health, port=self.args.health_port)
        self.health_server.start()
        METRICS.gauge("goldlive_session_up", 1, {"session": self.session_id})

        feed = build_feed(self.args.market, self.args.market_path)
        self._tasks = [
            asyncio.create_task(self._market_loop(feed), name="market"),
            asyncio.create_task(self._bar_loop(), name="bars"),
            asyncio.create_task(self._speak_loop(), name="speak"),
            asyncio.create_task(heartbeat(self.health, self.store), name="heartbeat"),
        ]
        if self.adapter is not None:
            self._tasks.append(asyncio.create_task(self._comment_loop(), name="comments"))

        log.info(
            "%s live: persona=%s market=%s adapter=%s generator=%s",
            self.session_id, spec["persona_id"], self.args.market,
            self.args.adapter, type(generator).__name__,
        )

        await self.stopping.wait()
        await self.shutdown(feed, llm)
        return 0

    async def shutdown(self, feed: Feed, llm) -> None:
        log.info("%s shutting down", self.session_id)
        METRICS.gauge("goldlive_session_up", 0, {"session": self.session_id})
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        with contextlib.suppress(Exception):
            await feed.close()
        if self.adapter is not None:
            with contextlib.suppress(Exception):
                await self.adapter.disconnect()
        if self.router is not None:
            await self.router.stop()
        if self.health_server is not None:
            self.health_server.stop()
        if llm is not None:
            with contextlib.suppress(Exception):
                await llm.close()
        await self.store.stop()
        log.info("%s stopped cleanly", self.session_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one live session")
    ap.add_argument("--session", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "local", "api", "offline"])
    ap.add_argument("--market", default="synthetic",
                    choices=["synthetic", "replay", "websocket"])
    ap.add_argument("--market-path")
    ap.add_argument("--adapter", default="mock", choices=["mock", "screen"])
    ap.add_argument("--tts", default="file", choices=["file", "piper"])
    ap.add_argument("--voices", default="voices")
    ap.add_argument("--out", default="out")
    ap.add_argument("--db", default="data/gold-live.db")
    ap.add_argument("--health-port", type=int, default=9101)
    ap.add_argument("--tick-s", type=float, default=2.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    session = LiveSession(args)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def request_stop(*_a) -> None:
        log.info("signal received, stopping")
        loop.call_soon_threadsafe(session.stopping.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError):
            signal.signal(sig, request_stop)

    try:
        sys.exit(loop.run_until_complete(session.run()))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
