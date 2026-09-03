"""Startup readiness: can this installation run a real session, right now?

Three concepts are kept apart on purpose, and collapsing them is how a system
ends up reporting green while producing nothing:

    provisioning     is what this machine needs installed?   shared/provisioning
    startup readiness can it run a production session now?   this module
    runtime liveness  is a running session still serving?    runtime/health.py
    process health    is the process alive and restarting?   runtime/supervisor

The rule every gate here follows: **prove it by doing it**. Not "websockets
imports" but "a tick arrived and the engine says LIVE". Not "the server
answered 200" but "the model wrote a sentence". Not "a voice file exists" but
"synthesis produced audio that is not silence". Every silent fallback in this
codebase -- the placeholder tone, the offline generator, the simulated audio
sink, the synthetic feed -- is designed to keep a session running, and every
one of them would satisfy a weaker check while broadcasting nothing of value.

Nothing here is ever cached. Installation state is cached; liveness is not.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from shared.paths import config_dir, config_path, data_path, data_root

log = logging.getLogger(__name__)

#: Feeds that cannot satisfy market_live no matter what they emit. They produce
#: perfectly good ticks, which is the problem: the engine would report LIVE and
#: the host would quote a price that does not exist.
SIMULATED_FEEDS = frozenset({"synthetic", "replay"})

TEST_SENTENCE = (
    "Gold is holding just under the level we marked out earlier this session."
)
#: Rough words-per-minute for a synthesised sentence; used only for bounds.
WORDS_PER_MINUTE = 165
SILENCE_FLOOR = 0.01


class Level(str, Enum):
    """Three levels, deliberately distinct.

    NOT_READY        cannot run a real session at all.
    SESSION_READY    the host genuinely speaks real words about real live
                     prices, through real synthesis, to a real audio device.
                     Nothing is simulated. It is not reaching an audience.
    BROADCAST_READY  additionally, audio is routed to a device broadcast
                     software can capture, and a comment source is connected.
    FULL_LIVE_READY  additionally, the route has been *physically verified*
                     end to end -- a human confirmed a viewer heard it.

    The last one can never be inferred from local checks. No amount of device
    enumeration proves a TikTok viewer heard anything, so it is only ever set
    by explicit human confirmation and is otherwise absent.
    """

    NOT_READY = "not_ready"
    SESSION_READY = "session_ready"
    BROADCAST_READY = "broadcast_ready"
    FULL_LIVE_READY = "full_live_ready"


@dataclass
class GateResult:
    name: str
    ok: bool
    evidence: str = ""
    reason: str = ""
    action: str = ""
    elapsed_ms: int = 0

    def render(self) -> str:
        if self.ok:
            return f"  [  ok  ] {self.name:<16} {self.evidence}"
        lines = [f"  [ FAIL ] {self.name:<16} {self.reason}"]
        if self.action:
            lines.append(f"           {'':<16} -> {self.action}")
        return "\n".join(lines)


#: Gates that must pass for the core AI session to be real.
SESSION_GATES = ("config_valid", "market_live", "model_real", "voice_real", "audio_out")
#: Additionally required to actually reach an audience.
BROADCAST_GATES = ("broadcast_route", "comments")


@dataclass
class Readiness:
    gates: list[GateResult] = field(default_factory=list)
    checked_at: str = ""

    def by_name(self, name: str) -> GateResult | None:
        return next((g for g in self.gates if g.name == name), None)

    def failed(self, names: tuple[str, ...]) -> list[GateResult]:
        return [g for g in self.gates if g.name in names and not g.ok]

    #: Set only by a human who confirmed a viewer actually heard the audio.
    #: See Level.FULL_LIVE_READY -- this is not something a local check can
    #: establish, so it is never inferred.
    broadcast_verified: bool = False

    @property
    def level(self) -> Level:
        if self.failed(SESSION_GATES):
            return Level.NOT_READY
        if self.failed(BROADCAST_GATES):
            return Level.SESSION_READY
        if self.broadcast_verified:
            return Level.FULL_LIVE_READY
        return Level.BROADCAST_READY

    def render(self) -> str:
        out = [g.render() for g in self.gates]
        out.append("")
        level = self.level
        if level is Level.NOT_READY:
            names = ", ".join(g.name for g in self.failed(SESSION_GATES))
            out.append(f"  NOT READY -- blocked by: {names}")
        elif level is Level.SESSION_READY:
            names = ", ".join(g.name for g in self.failed(BROADCAST_GATES))
            out.append("  SESSION READY -- the host can speak about real prices.")
            out.append(f"  Broadcasting is not set up ({names}).")
        elif level is Level.BROADCAST_READY:
            out.append("  BROADCAST READY -- audio is routed to a capture device.")
            out.append("  NOT yet FULL LIVE READY: nobody has confirmed a viewer")
            out.append("  actually heard it, and that cannot be checked from here.")
        else:
            out.append("  FULL LIVE READY -- the broadcast path was verified by a human.")
        return "\n".join(out)

    def to_json(self) -> dict:
        return {
            "level": self.level.value,
            "checked_at": self.checked_at,
            "gates": {
                g.name: {
                    "ok": g.ok, "evidence": g.evidence, "reason": g.reason,
                    "action": g.action, "elapsed_ms": g.elapsed_ms,
                }
                for g in self.gates
            },
        }


async def _timed(name: str, coro) -> GateResult:
    started = time.perf_counter()
    try:
        result: GateResult = await coro
    except Exception as exc:  # a gate that throws is a gate that failed
        log.debug("gate %s raised", name, exc_info=True)
        # Several exceptions worth reporting stringify to nothing --
        # asyncio.TimeoutError is the common one -- so a bare {exc} produces
        # "check raised: " and tells the user precisely nothing.
        detail = str(exc) or type(exc).__name__
        result = GateResult(name, False, reason=f"check raised: {detail}",
                            action="this is a bug; please report it")
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result


# -- config_valid ----------------------------------------------------------


async def gate_config_valid(session_id: str, voices_dir: Path) -> GateResult:
    """Config must resolve all the way down to a voice file on disk.

    Discovering that a persona names a voice nobody installed *after* the
    session starts means the first utterance is the error report.
    """
    import yaml

    name = "config_valid"
    try:
        cfg = yaml.safe_load(config_path("sessions.yaml").read_text(encoding="utf-8"))
    except Exception as exc:
        return GateResult(name, False, reason=f"sessions.yaml unreadable: {exc}",
                          action="run: GoldLive.exe setup --force")

    sessions = (cfg or {}).get("sessions") or []
    spec = next((s for s in sessions if s.get("session_id") == session_id), None)
    if spec is None:
        found = ", ".join(s.get("session_id", "?") for s in sessions) or "none"
        return GateResult(name, False, reason=f"{session_id} is not in sessions.yaml",
                          action=f"sessions defined: {found}")

    persona_id = spec.get("persona_id")
    if not persona_id:
        return GateResult(name, False, reason=f"{session_id} has no persona_id",
                          action="add persona_id to the session entry")

    persona_file = config_dir("personas") / f"{persona_id}.yaml"
    if not persona_file.is_file():
        return GateResult(name, False, reason=f"persona {persona_id!r} has no config",
                          action=f"expected {persona_file}")

    # Read the voice the SESSION will actually use, from the same loader
    # live.py uses. This gate previously read DEFAULT_PROFILE in
    # scripts/get_voices.py, which is a different mapping -- so it validated
    # one voice while the session loaded another, and passed while the session
    # was guaranteed to fail. A gate that checks something other than the code
    # under test is worse than no gate.
    try:
        from intelligence.personas import load_personas

        persona = load_personas(config_dir("personas"))[persona_id]
        voice_id = persona.voice_id
    except Exception as exc:
        return GateResult(name, False, reason=f"persona {persona_id!r} is unreadable: {exc}",
                          action="run: GoldLive.exe setup --force")

    if not voice_id or voice_id == "default":
        return GateResult(name, False, reason=f"persona {persona_id!r} names no voice",
                          action=f"set voice_id in {persona_file}")

    # A voice that is not in the catalogue cannot be provisioned, so no amount
    # of repair will fix it. That means stale configuration, not a missing
    # download -- seed_config only copies files that are ABSENT, so a config
    # written by an older version is kept forever and silently misread.
    if voice_id not in _catalogue_ids():
        return GateResult(
            name, False,
            reason=(f"persona {persona_id!r} names voice {voice_id!r}, which is not a "
                    "real voice -- this configuration is out of date"),
            action="run: GoldLive.exe setup --force   (then: GoldLive.exe provision)",
        )

    model = voices_dir / f"{voice_id}.onnx"
    if not model.is_file():
        return GateResult(
            name, False, reason=f"voice {voice_id} is not installed",
            action="run: GoldLive.exe provision",
        )

    return GateResult(name, True, evidence=f"{session_id} -> {persona_id} -> {voice_id}")


def _catalogue_ids() -> set[str]:
    """Voice ids that can actually be downloaded."""
    try:
        from scripts.get_voices import CATALOGUE

        return set(CATALOGUE)
    except Exception:
        return set()


def _voice_for_persona(persona_id: str) -> str | None:
    """The voice a persona actually uses, from its config -- not from the
    installer's default mapping, which can and did disagree."""
    try:
        from intelligence.personas import load_personas

        return load_personas(config_dir("personas"))[persona_id].voice_id
    except Exception:
        return None


# -- market_live -----------------------------------------------------------


async def gate_market_live(market: str, timeout_s: float = 30.0) -> GateResult:
    """A real feed, a real tick, and an engine that says LIVE.

    Rejecting simulated feeds by *name* rather than by outcome is deliberate.
    SyntheticFeed and ReplayFeed emit well-formed ticks, so the engine would
    happily report LIVE and may_quote_price() would return True -- and the host
    would quote a price that has never existed.
    """
    name = "market_live"
    if market in SIMULATED_FEEDS:
        return GateResult(
            name, False,
            reason=f"{market!r} is a simulated feed and cannot prove live pricing",
            action="use --market gold, or pass --allow-degraded for development",
        )

    from platform_.market.engine import MarketEngine
    from runtime.live import build_feed

    feed = build_feed(market, None)
    engine = MarketEngine()
    delayed, stale, unavailable = feed.staleness_thresholds()
    engine.delayed_after_s = delayed
    engine.stale_after_s = stale
    engine.unavailable_after_s = unavailable

    tick = None
    try:
        async def first_tick():
            async for t in feed.ticks():
                return t
            return None

        tick = await asyncio.wait_for(first_tick(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return GateResult(
            name, False,
            reason=f"no tick from the {market} feed within {timeout_s:.0f}s",
            action="check the network can reach data-stream.binance.vision:443",
        )
    except Exception as exc:
        return GateResult(name, False, reason=f"{market} feed failed: {exc}",
                          action="check network connectivity and try again")
    finally:
        with contextlib.suppress(Exception):
            await feed.close()

    if tick is None:
        return GateResult(name, False, reason=f"the {market} feed closed without a tick",
                          action="check network connectivity and try again")

    engine.on_tick(tick.bid, tick.ask, tick.at)
    state = engine.snapshot()
    if not state.may_quote_price():
        return GateResult(
            name, False,
            reason=f"tick received but confidence is {state.confidence.value}",
            action="the feed is lagging; retry in a moment",
        )

    return GateResult(name, True,
                      evidence=f"{state.price.mid:.2f} confidence={state.confidence.value}")


# -- model_real ------------------------------------------------------------


async def gate_model_real(min_words: int = 20, budget_s: float = 30.0) -> GateResult:
    """A named, installed model that actually writes a sentence.

    None of these is sufficient on its own, and each has been seen to pass
    while generation was impossible: an HTTP 200 from /v1/models, a non-empty
    response body, a model list that is `[]`, or the placeholder name
    "local-model" that LocalLLM falls back to when nothing is configured.
    """
    name = "model_real"
    from platform_.llm.discovery import discover
    from platform_.llm.base import ChatMessage

    llm = await discover()
    if llm is None:
        return GateResult(
            name, False,
            reason="no local model server is running, or it has no models installed",
            action="start Ollama (`ollama serve`) then `ollama pull llama3.2:3b`",
        )

    try:
        if not llm.model or llm.model == "local-model":
            return GateResult(
                name, False,
                reason="the model server did not name an installed model",
                action="run: ollama pull llama3.2:3b",
            )

        started = time.perf_counter()
        try:
            reply = await asyncio.wait_for(
                llm.complete([
                    ChatMessage(role="system", content="You are a market commentator."),
                    ChatMessage(
                        role="user",
                        content="In two sentences, describe what a support level is.",
                    ),
                ], max_tokens=120, temperature=0.6),
                timeout=budget_s,
            )
        except asyncio.TimeoutError:
            return GateResult(
                name, False,
                reason=f"{llm.model} did not answer within {budget_s:.0f}s",
                action="this machine may be too slow for that model; try a smaller one",
            )

        text = (getattr(reply, "text", None) or "").strip()
        words = len(text.split())
        if words < min_words:
            return GateResult(
                name, False,
                reason=f"{llm.model} returned {words} words, expected at least {min_words}",
                action="the model server is reachable but not generating usefully",
            )

        elapsed = time.perf_counter() - started
        return GateResult(name, True,
                          evidence=f"{llm.model} wrote {words} words in {elapsed:.1f}s")
    finally:
        with contextlib.suppress(Exception):
            await llm.close()


# -- voice_real ------------------------------------------------------------


async def gate_voice_real(voice_id: str, voices_dir: Path) -> GateResult:
    """Real synthesis of a fixed sentence, checked for plausibility.

    PiperTTS._model_path silently falls back to the default voice when the
    requested one is absent, so the returned path is checked against what was
    asked for -- otherwise a machine missing every voice but one passes this
    gate and then broadcasts in the wrong voice.
    """
    name = "voice_real"
    try:
        import piper  # noqa: F401
    except ImportError as exc:
        return GateResult(
            name, False, reason=f"piper is not available: {exc}",
            action="this build is incomplete; download Gold Live again",
        )

    model = voices_dir / f"{voice_id}.onnx"
    if not model.is_file():
        return GateResult(name, False, reason=f"voice {voice_id} is not installed",
                          action="run: GoldLive.exe provision")

    from platform_.tts.piper import PiperTTS

    tts = PiperTTS(voices_dir=voices_dir)
    resolved = tts._model_path(voice_id)
    if resolved.stem != voice_id:
        return GateResult(
            name, False,
            reason=f"requested {voice_id} but piper resolved {resolved.stem}",
            action="the voice file is missing or unreadable; run: GoldLive.exe provision",
        )

    out = data_path("selftest", f"voice-{voice_id}.wav")
    result = await tts.synthesize(TEST_SENTENCE, voice_id, out)
    if result.path is None or not Path(result.path).is_file():
        return GateResult(name, False, reason="synthesis produced no file",
                          action="check the voice model is not truncated")

    ok, detail = _inspect_wav(Path(result.path))
    if not ok:
        return GateResult(name, False, reason=detail,
                          action="the voice model may be corrupt; run: GoldLive.exe provision")

    return GateResult(name, True, evidence=f"{voice_id}: {detail}")


def _inspect_wav(path: Path) -> tuple[bool, str]:
    """Valid, non-empty, plausible length, and not silence."""
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            width = wav.getsampwidth()
            raw = wav.readframes(frames)
    except (wave.Error, OSError) as exc:
        return False, f"unreadable wav: {exc}"

    if frames == 0 or rate == 0:
        return False, "wav contains no audio"

    duration = frames / rate
    words = len(TEST_SENTENCE.split())
    expected = words / WORDS_PER_MINUTE * 60
    if not (expected * 0.4 <= duration <= expected * 2.5):
        return False, (f"{duration:.1f}s of audio for {words} words is implausible "
                       f"(expected around {expected:.1f}s)")

    peak = _peak_amplitude(raw, width)
    if peak < SILENCE_FLOOR:
        return False, f"audio is effectively silent (peak {peak:.4f})"

    return True, f"{duration:.1f}s of audio, peak {peak:.2f}"


def _peak_amplitude(raw: bytes, sample_width: int) -> float:
    if not raw or sample_width not in (1, 2, 4):
        return 0.0
    try:
        import numpy as np

        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sample_width]
        data = np.frombuffer(raw, dtype=dtype)
        if data.size == 0:
            return 0.0
        if sample_width == 1:
            return float(np.max(np.abs(data.astype(np.int16) - 128)) / 128.0)
        full = float(np.iinfo(dtype).max)
        return float(np.max(np.abs(data.astype(np.float64))) / full)
    except ImportError:
        import audioop  # pragma: no cover - stdlib fallback

        return audioop.max(raw, sample_width) / float(2 ** (8 * sample_width - 1))
    except Exception:
        return 0.0


# -- audio_out -------------------------------------------------------------


async def gate_audio_out(device_index: int | None = None) -> GateResult:
    """Open a real output stream and play a short quiet tone.

    Deliberately does NOT go through AudioSink.play(), which catches ImportError
    and returns as if it had played so the rest of the pipeline can be exercised
    on a machine with no sound card. That behaviour is correct for its purpose
    and useless as proof: it succeeds on a machine that cannot make a sound.
    """
    name = "audio_out"
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        return GateResult(
            name, False, reason=f"no audio backend: {exc}",
            action="this build is incomplete; download Gold Live again",
        )

    try:
        from platform_.audio.devices import list_output_devices

        devices = list_output_devices()
    except Exception as exc:
        return GateResult(name, False, reason=f"could not enumerate audio devices: {exc}",
                          action="check Windows sound settings")

    if not devices:
        return GateResult(name, False, reason="no audio output devices on this machine",
                          action="connect speakers or install an audio driver")

    rate = 44100
    duration = 0.2
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    # -40 dBFS: audible to a meter, inaudible across a room. Nobody should be
    # startled by a readiness check.
    tone = (0.01 * np.sin(2 * math.pi * 440 * t)).astype("float32")

    try:
        await asyncio.to_thread(_play_blocking, sd, tone, rate, device_index)
    except Exception as exc:
        target = f"device {device_index}" if device_index is not None else "the default device"
        return GateResult(name, False, reason=f"could not play to {target}: {exc}",
                          action="check the output device is present and not exclusive-locked")

    chosen = devices[device_index].name if (
        device_index is not None and 0 <= device_index < len(devices)
    ) else "default device"
    return GateResult(name, True, evidence=f"played a test tone to {chosen}")


def _play_blocking(sd, tone, rate: int, device_index: int | None) -> None:
    sd.play(tone, rate, device=device_index, blocking=True)
    sd.stop()


# -- broadcast_route -------------------------------------------------------


async def gate_broadcast_route() -> GateResult:
    name = "broadcast_route"
    try:
        from platform_.audio.devices import find_virtual_cable, list_output_devices

        cable = find_virtual_cable()
        count = len(list_output_devices())
    except Exception as exc:
        return GateResult(name, False, reason=f"could not enumerate audio devices: {exc}",
                          action="check Windows sound settings")

    if cable is None:
        return GateResult(
            name, False,
            reason=f"no virtual audio cable among {count} output device(s)",
            action=("install VB-CABLE from vb-audio.com, then point LIVE Studio's "
                    "MICROPHONE at it -- not desktop audio"),
        )
    return GateResult(name, True, evidence=f"routing through {cable.name}")


# -- comments --------------------------------------------------------------


async def gate_comments(adapter: str) -> GateResult:
    """Whether a comment source can be reached. Broadcast-only, never a
    session gate: the host can talk about gold perfectly well with no audience
    input, and blocking a session on OCR would be wrong."""
    name = "comments"
    if adapter == "mock":
        return GateResult(name, False, reason="the mock adapter is a development stub",
                          action="use --adapter file, or screen once OCR is set up")
    if adapter == "screen":
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return GateResult(
                name, False, reason="paddleocr is not installed",
                action="install the ocr extra, or use --adapter file for now",
            )
        calib = data_path("configs", "devices.json", create_parent=False)
        if not calib.exists():
            return GateResult(name, False, reason="screen capture is not calibrated",
                              action="run: GoldLive.exe calibrate --session SESSION_001")
        return GateResult(name, True, evidence="paddleocr present and calibrated")
    if adapter == "file":
        path = data_path("comments.txt")
        if not path.exists():
            path.write_text("", encoding="utf-8")
        return GateResult(name, True, evidence=f"watching {path}")
    if adapter == "youtube":
        import os

        if not os.environ.get("YOUTUBE_VIDEO_ID"):
            return GateResult(name, False, reason="YOUTUBE_VIDEO_ID is not set",
                              action="set it in the .env file in the data directory")
        return GateResult(name, True, evidence="youtube adapter configured")
    return GateResult(name, False, reason=f"unknown adapter {adapter!r}",
                      action="use one of: file, screen, youtube")


# -- the whole check -------------------------------------------------------


async def check(
    session_id: str = "SESSION_001",
    market: str = "gold",
    adapter: str = "file",
    voice_id: str | None = None,
    voices_dir: Path | None = None,
    include_broadcast: bool = True,
    device_index: int | None = None,
) -> Readiness:
    """Run every gate. Order is cheapest-first so failures surface quickly."""
    voices_dir = voices_dir or data_root() / "voices"
    if voice_id is None:
        voice_id = _voice_for_session(session_id) or "en_US-john-medium"

    gates = [
        ("config_valid", gate_config_valid(session_id, voices_dir)),
        ("audio_out", gate_audio_out(device_index)),
        ("voice_real", gate_voice_real(voice_id, voices_dir)),
        ("market_live", gate_market_live(market)),
        ("model_real", gate_model_real()),
    ]
    if include_broadcast:
        gates += [
            ("broadcast_route", gate_broadcast_route()),
            ("comments", gate_comments(adapter)),
        ]

    results = [await _timed(name, coro) for name, coro in gates]
    return Readiness(
        gates=results,
        checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _voice_for_session(session_id: str) -> str | None:
    import yaml

    try:
        cfg = yaml.safe_load(config_path("sessions.yaml").read_text(encoding="utf-8"))
        spec = next(
            s for s in (cfg or {}).get("sessions", []) if s.get("session_id") == session_id
        )
        return _voice_for_persona(spec["persona_id"])
    except Exception:
        return None
