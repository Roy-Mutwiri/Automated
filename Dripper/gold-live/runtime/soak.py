"""Soak test: does the system still sound alive after N hours?

This is the test that matters for a 24/7 product. Ten minutes of good output
proves nothing -- the real failure modes only appear over days:

  - content exhaustion     the planner runs out of things to say
  - repetition drift       hour 6 sounds like hour 2
  - dead air               long stretches with nothing said at all
  - weekend collapse       48 hours with no price action and no plan for it
  - unbounded memory       similarity index grows until the process dies

Runs entirely offline on a simulated clock, so 72 hours takes seconds.

    python -m runtime.soak --hours 24
    python -m runtime.soak --hours 72 --start-friday   # includes the weekend
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from intelligence.content import (
    ContentPlanner,
    MarketPhase,
    classify_phase,
    load_content,
    market_is_closed,
)
from intelligence.generation import OfflineGenerator
from intelligence.personas import load_personas
from runtime.session import SessionRuntime
from shared.contracts import (
    MarketConfidence,
    MarketState,
    PlatformBinding,
    Price,
    SessionState,
    SessionStatus,
    TradingSession,
)
from shared.mocks.market import MockMarketEngine
from shared.mocks.tts import MockTTS

ROOT = Path(__file__).resolve().parent.parent
TICK_SECONDS = 20


def synth_state(now: datetime, engine: MockMarketEngine, closed: bool) -> MarketState:
    """Market state for a given simulated moment."""
    if closed:
        # No feed at all over the weekend. This is not a degraded state to
        # recover from -- it is the normal condition for ~48h a week, and the
        # system must be fine talking through it.
        return MarketState(
            as_of=now - timedelta(hours=1),
            computed_at=now,
            confidence=MarketConfidence.UNAVAILABLE,
            price=Price(bid=0.0, ask=0.0),
            session=TradingSession.OFF_HOURS,
        )
    state = engine.tick()
    return state.model_copy(update={"computed_at": now, "as_of": now - timedelta(seconds=0.4)})


async def soak(hours: int, start: datetime, persona_id: str) -> dict:
    personas = load_personas(ROOT / "configs" / "personas")
    items = load_content(ROOT / "configs" / "content.yaml")
    planner = ContentPlanner(items, seed=3)

    rt = SessionRuntime(
        state=SessionState(
            session_id="SESSION_001",
            persona_id=persona_id,
            status=SessionStatus.LIVE,
            platform_binding=PlatformBinding(platform="mock", channel_id="soak"),
        ),
        persona=personas[persona_id],
        generator=OfflineGenerator(),
        tts=MockTTS(),
        out_dir=Path("out-soak"),
        planner=planner,
    )

    engine = MockMarketEngine(cyclic=True)
    ticks = int(hours * 3600 / TICK_SECONDS)

    spoken_at: list[datetime] = []
    phase_counts: Counter[str] = Counter()
    hourly_utterances: Counter[int] = Counter()
    hourly_repeats: Counter[int] = Counter()
    beats_used: list[str] = []
    exhausted_ticks = 0
    t_start = time.perf_counter()

    for i in range(ticks):
        now = start + timedelta(seconds=i * TICK_SECONDS)
        closed = market_is_closed(now)
        state = synth_state(now, engine, closed)
        atr = state.timeframes.get("5m").atr if state.timeframes else None
        phase = classify_phase(now, atr=atr)
        phase_counts[phase.value] += 1
        hour = int((now - start).total_seconds() // 3600)

        if not closed:
            for ev in engine.drain_events():
                rt.on_market_event(ev, now)

        # Keep the content queue topped up whenever the director is short of
        # material. This is the mechanism that fills closed markets.
        if rt.director.queue_depth < 2:
            rt.offer_planned_content(phase, now)
        # Only genuine exhaustion counts -- next_beat() also returns None when
        # merely rate-limited, which is normal and not a failure.
        if planner.is_exhausted(phase, now):
            exhausted_ticks += 1

        before_repeats = len(rt.dropped_repetitive)
        resp = await rt.tick(state, now)
        if resp is not None:
            spoken_at.append(now)
            hourly_utterances[hour] += 1
            if resp.trigger.type.value == "education":
                beats_used.append(resp.text[:40])
        hourly_repeats[hour] += len(rt.dropped_repetitive) - before_repeats

    wall = time.perf_counter() - t_start

    gaps = [
        (b - a).total_seconds() / 60
        for a, b in zip(spoken_at, spoken_at[1:], strict=False)
    ]
    first_half = sum(hourly_repeats[h] for h in range(hours // 2))
    second_half = sum(hourly_repeats[h] for h in range(hours // 2, hours))

    return {
        "offline": isinstance(rt.generator, OfflineGenerator),
        "hours": hours,
        "ticks": ticks,
        "utterances": len(spoken_at),
        "per_hour": round(len(spoken_at) / hours, 1),
        "repeats_blocked": len(rt.dropped_repetitive),
        "unsafe_blocked": len(rt.dropped_unsafe),
        "repeats_first_half": first_half,
        "repeats_second_half": second_half,
        "longest_silence_min": round(max(gaps), 1) if gaps else 0.0,
        "median_gap_min": round(statistics.median(gaps), 2) if gaps else 0.0,
        "content_exhausted_ticks": exhausted_ticks,
        "content_exhausted_at": rt.content_exhausted_at.isoformat()
        if rt.content_exhausted_at
        else None,
        "distinct_education_beats": len(set(beats_used)),
        "phase_distribution": dict(phase_counts),
        "coverage_open": planner.coverage_report(MarketPhase.QUIET),
        "coverage_closed": planner.coverage_report(MarketPhase.CLOSED),
        "memory_utterances": rt.memory.utterance_count,
        "wall_seconds": round(wall, 1),
        "hourly": {h: hourly_utterances[h] for h in range(hours)},
    }


def report(r: dict) -> int:
    print(f"\n{'=' * 74}\n  SOAK: {r['hours']}h simulated in {r['wall_seconds']}s wall clock")
    print(f"{'=' * 74}\n")
    print(f"  utterances            {r['utterances']}  ({r['per_hour']}/hour)")
    print(f"  median gap            {r['median_gap_min']} min")
    print(f"  longest silence       {r['longest_silence_min']} min")
    print(f"  repeats blocked       {r['repeats_blocked']}")
    print(f"  unsafe blocked        {r['unsafe_blocked']}")
    print(f"  distinct edu beats    {r['distinct_education_beats']}")
    print(f"  content exhausted     {r['content_exhausted_ticks']} ticks"
          f"{'  at ' + r['content_exhausted_at'] if r['content_exhausted_at'] else ''}")
    print(f"  phases                {r['phase_distribution']}")
    print(f"  inventory (open)      {r['coverage_open']}")
    print(f"  inventory (closed)    {r['coverage_closed']}")

    print("\n  utterances per hour")
    peak = max(r["hourly"].values()) or 1
    for h, n in r["hourly"].items():
        bar = "#" * int(24 * n / peak)
        print(f"    h{h:02d} {n:>4} |{bar}")

    print("\n  DRIFT CHECK")
    a, b = r["repeats_first_half"], r["repeats_second_half"]
    print(f"    repetition pressure first half {a}, second half {b}")

    # Structural criteria: these hold regardless of who writes the words.
    fails = []
    if r["longest_silence_min"] > 20:
        fails.append(f"dead air: {r['longest_silence_min']} min without speaking")
    if r["content_exhausted_ticks"] > r["ticks"] * 0.05:
        fails.append(f"content exhausted on {r['content_exhausted_ticks']} ticks")
    drift = r["repeats_second_half"] - r["repeats_first_half"]
    if r["repeats_first_half"] and drift > r["repeats_first_half"] * 0.5:
        fails.append(f"repetition drift: pressure grew {drift} between halves")

    print()
    if fails:
        print("  RESULT: FAIL")
        for f in fails:
            print(f"    - {f}")
    else:
        print("  RESULT: PASS - no exhaustion, no dead air, no repetition drift")

    if r["offline"]:
        print(
            "\n  NOTE: offline generator. Templates cannot produce hundreds of\n"
            "  genuinely distinct utterances, so most planned content is blocked\n"
            "  as repetitive and throughput is understated. This run validates\n"
            f"  STRUCTURE only ({r['per_hour']}/hour here). Run with --live to\n"
            "  judge content volume and quality."
        )
    elif r["per_hour"] < 20:
        fails.append(f"too quiet: {r['per_hour']}/hour")
        print(f"  FAIL: too quiet at {r['per_hour']}/hour")
    print()
    return len(fails)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold Live soak test")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--persona", default="educator")
    ap.add_argument(
        "--start-friday",
        action="store_true",
        help="start Friday 18:00 UTC so the run covers the weekend close",
    )
    args = ap.parse_args()

    if args.start_friday:
        start = datetime(2026, 9, 4, 18, 0, tzinfo=timezone.utc)  # a Friday
    else:
        start = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)  # a Tuesday

    result = asyncio.run(soak(args.hours, start, args.persona))
    raise SystemExit(1 if report(result) else 0)


if __name__ == "__main__":
    main()
