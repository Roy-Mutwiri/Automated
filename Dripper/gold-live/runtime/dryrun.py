"""M1: the dry-run session.

Mock market + scripted comments -> Director -> generator -> safety -> TTS -> wav.
No platform, no device, no network required.

This exists to answer the one question no architecture can: does the output
sound like something a person would listen to for an hour? Run it, read the
transcript, listen to the audio. If it is boring, everything downstream is
wasted effort -- and you have found that out in week one.

    python -m runtime.dryrun                    # local model if up, else offline
    python -m runtime.dryrun --mode local       # require the local model
    python -m runtime.dryrun --mode offline     # templates, structure only
    python -m runtime.dryrun --sessions 3       # isolation check across 3 sessions
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import timedelta
from pathlib import Path

import yaml

from intelligence.generation import build_generator
from intelligence.personas import load_personas
from runtime.session import SessionRuntime
from shared.contracts import (
    PlatformBinding,
    SessionState,
    SessionStatus,
    utcnow,
)
from shared.mocks import FileTTS, MockCommentSource, MockMarketEngine

from shared.paths import config_dir, config_path, data_path

ROOT = Path(__file__).resolve().parent.parent
FILLER_TOPICS = [
    "position sizing",
    "what invalidates a setup",
    "why liquidity pools form at obvious highs",
    "the difference between a break and a retest",
]


def load_sessions(n: int) -> list[dict]:
    cfg = yaml.safe_load(config_path("sessions.yaml").read_text(encoding="utf-8"))
    return cfg["sessions"][:n]


async def run(beats: int, n_sessions: int, mode: str, out_dir: Path) -> int:
    personas = load_personas(config_dir("personas"))
    generator, _llm = await build_generator(mode)
    tts = FileTTS()
    market = MockMarketEngine()

    runtimes: list[SessionRuntime] = []
    sources: list[MockCommentSource] = []
    for spec in load_sessions(n_sessions):
        state = SessionState(
            session_id=spec["session_id"],
            persona_id=spec["persona_id"],
            status=SessionStatus.LIVE,
            device_id=spec.get("device_id"),
            platform_binding=PlatformBinding(
                platform=spec["platform"], channel_id=spec["channel_id"]
            ),
        )
        runtimes.append(
            SessionRuntime(
                state=state,
                persona=personas[spec["persona_id"]],
                generator=generator,
                tts=tts,
                out_dir=out_dir,
            )
        )
        sources.append(MockCommentSource(spec["session_id"]))

    t0 = utcnow()
    print(f"\n{'=' * 78}\n  GOLD LIVE - M1 DRY RUN")
    print(f"  generator: {type(generator).__name__} (mode={mode})   "
          f"sessions: {n_sessions}   beats: {beats}")
    print(f"{'=' * 78}\n")

    for beat in range(beats):
        state = market.tick()
        events = market.drain_events()
        # Simulated clock so cooldowns and silence boosts behave realistically.
        now = t0 + timedelta(seconds=beat * 20)

        header = (
            f"[beat {beat:02d}] {state.price.mid:>8.2f}  "
            f"{state.confidence.value:<11} "
            f"stale={state.staleness_ms:>6}ms  "
            f"{state.timeframes['5m'].structure.value}"
        )
        print(header)
        for ev in events:
            print(f"           ! {ev.kind.value} sev={ev.severity} :: {ev.narrative_hint}")

        for rt, src in zip(runtimes, sources, strict=True):
            # `now` is threaded through everything so intent ages are measured
            # against the same clock the Director scores with. Passing the
            # simulated clock here and letting intents default to wall-clock
            # made every intent look instantly expired.
            for c in src.at_beat(beat):
                await rt.on_comment(c, now)
            for ev in events:
                rt.on_market_event(ev, now)
            if beat % 4 == 0:
                rt.offer_filler(FILLER_TOPICS[beat % len(FILLER_TOPICS)], now)

            resp = await rt.tick(state, now)
            if resp is not None:
                tag = f"{rt.state.session_id}/{rt.persona.persona_id}"
                print(f"           > [{tag}] {resp.text}")

    # -- report ------------------------------------------------------------

    print(f"\n{'=' * 78}\n  RESULTS\n{'=' * 78}")
    total_unsafe = 0
    for rt in runtimes:
        print(f"\n  {rt.state.session_id} ({rt.persona.display_name})")
        print(f"    utterances spoken : {rt.spoken_count}")
        print(f"    blocked by safety : {len(rt.dropped_unsafe)}")
        print(f"    dropped as repeat : {len(rt.dropped_repetitive)}")
        print(f"    director queue    : {rt.director.queue_depth} still pending")
        total_unsafe += len(rt.dropped_unsafe)
        for text, sim in rt.dropped_repetitive:
            print(f"      REPEAT (sim={sim:.2f}): {text[:64]}...")
        for text, violations in rt.dropped_unsafe:
            print(f"      BLOCKED: {violations} :: {text[:70]}...")

    # Isolation check: no session may contain another's comment ids.
    if n_sessions > 1:
        print("\n  ISOLATION CHECK")
        ok = True
        for i, rt in enumerate(runtimes):
            for j, other in enumerate(runtimes):
                if i == j:
                    continue
                mine = {r.trace_id for r in rt.transcript}
                theirs = {r.trace_id for r in other.transcript}
                if mine & theirs:
                    ok = False
                    print(f"    FAIL: {rt.state.session_id} shares traces with "
                          f"{other.state.session_id}")
        print(f"    {'PASS - no cross-session leakage' if ok else 'FAILED'}")

    print(f"\n  audio + transcripts written to: {out_dir}\n")
    return total_unsafe


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold Live M1 dry run")
    ap.add_argument("--beats", type=int, default=18, help="market ticks to simulate")
    ap.add_argument("--sessions", type=int, default=1, help="concurrent sessions")
    ap.add_argument("--mode", default="auto", choices=["auto","local","api","offline"],
                    help="generator backend (default: local if running, else offline)")
    ap.add_argument("--out", default=str(data_path("out", create_parent=False)),
                    help="output directory")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    asyncio.run(run(args.beats, args.sessions, args.mode, Path(args.out)))


if __name__ == "__main__":
    main()
