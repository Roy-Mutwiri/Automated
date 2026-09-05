"""High volume, no wall clock: does anything accumulate?

A real soak takes a real hour and can only be run occasionally. This runs the
same shapes of work thousands of times in seconds, so a leak or an unbounded
structure shows up on every commit rather than once a week.

Deliberately no sleeps except where timing is the thing under test. What is
being measured is accumulation -- tasks, queue depth, retained objects,
descriptors -- not latency.
"""

from __future__ import annotations

import asyncio
import gc

import pytest
from fakes import FakeSink, FakeTTS, audio_request

from uuid import UUID

from platform_.audio.router import AudioRouter, PlaybackState

#: Enough volume to expose an accidental retained reference, small enough that
#: the suite still runs in seconds.
UTTERANCES = 2_000
CYCLES = 300


async def settle(ticks: int = 40) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


# -- router under sustained load -------------------------------------------


async def test_thousands_of_utterances_leave_no_residue(tmp_path):
    sink = FakeSink()
    router = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=sink, max_queue=8)
    await router.start()

    for i in range(UTTERANCES):
        await router.submit(audio_request(segments=[f"line {i}"]))
        if i % 50 == 0:
            await settle(4)
    await settle(400)
    await router.stop()

    assert router.queue_depth == 0, "the queue must drain"
    assert router.current_utterance is None
    assert router._preempted is None
    assert router.state is PlaybackState.STOPPED


async def test_sustained_interruption_does_not_accumulate_state(tmp_path):
    """Barge-in on nearly every utterance -- the pathological case."""
    from shared.contracts import Priority

    router = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=FakeSink())
    await router.start()

    for i in range(500):
        priority = Priority.CRITICAL if i % 2 else Priority.LOW
        router._current = audio_request(segments=["in flight"])
        await router.submit(audio_request(segments=[f"u{i}"], priority=priority))
        if i % 25 == 0:
            await settle(4)
    await settle(400)
    await router.stop()

    assert router._preempted is None or isinstance(router._preempted, UUID)
    assert router.queue_depth == 0


async def test_a_mixed_failure_workload_drains_cleanly(tmp_path):
    """The realistic mix: most succeed, some fail, some are empty."""
    for tts in (FakeTTS(), FakeTTS(fail_on=3), FakeTTS(empty=True)):
        router = AudioRouter("SESSION_001", tts, tmp_path, sink=FakeSink(), max_queue=8)
        await router.start()
        for i in range(200):
            await router.submit(audio_request(segments=[f"x{i}"]))
            if i % 20 == 0:
                await settle(4)
        await settle(400)
        await router.stop()

        assert router.queue_depth == 0
        assert router.current_utterance is None


# -- repeated lifecycle cycles ---------------------------------------------


async def test_hundreds_of_router_start_stop_cycles_leak_nothing(tmp_path):
    """Lifecycle churn is where task leaks hide."""
    gc.collect()
    before = len(asyncio.all_tasks())

    for _ in range(CYCLES):
        router = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=FakeSink())
        await router.start()
        await router.submit(audio_request(segments=["x"]))
        await router.stop()

    gc.collect()
    after = len(asyncio.all_tasks())
    assert after <= before + 2, f"asyncio tasks accumulated: {before} -> {after}"


def test_hundreds_of_lifecycle_transitions_leave_one_state_file(tmp_path, monkeypatch):
    """START/STOP recorded many times must not churn out files or grow."""
    from runtime import lifecycle

    path = tmp_path / "lifecycle.json"
    monkeypatch.setattr(lifecycle, "state_path", lambda: path)

    for _ in range(CYCLES):
        lifecycle.request_start()
        lifecycle.request_stop()
        lifecycle.mark_stopped()

    files = list(tmp_path.iterdir())
    assert [f.name for f in files] == ["lifecycle.json"], f"stray files: {files}"
    assert path.stat().st_size < 4096, "the state file must not grow with use"
    assert lifecycle.load().state is lifecycle.RunState.STOPPED


def test_reading_state_repeatedly_does_not_rewrite_it(tmp_path, monkeypatch):
    """The panel polls every 1.5s. If every read rewrote the file, a day of
    idling would be ~57,000 writes to the user's disk."""
    from runtime import lifecycle

    path = tmp_path / "lifecycle.json"
    monkeypatch.setattr(lifecycle, "state_path", lambda: path)

    lifecycle.mark_stopped()
    stamp = path.stat().st_mtime_ns
    content = path.read_text(encoding="utf-8")

    for _ in range(500):
        lifecycle.load()

    assert path.read_text(encoding="utf-8") == content
    assert path.stat().st_mtime_ns == stamp, "reads must not touch the file"


# -- session-level accumulation --------------------------------------------


def test_a_long_session_of_mostly_rejected_output_stays_bounded():
    """The production mix: about three rejections per utterance spoken."""
    from collections import deque

    from intelligence.memory import SessionMemory
    from runtime.session import RETAINED_DROPS, RETAINED_UTTERANCES

    memory = SessionMemory("SOAK")
    transcript: deque[str] = deque(maxlen=RETAINED_UTTERANCES)
    dropped: deque[tuple[str, float]] = deque(maxlen=RETAINED_DROPS)
    spoken = rejected = 0

    for i in range(UTTERANCES * 5):
        text = f"observation number {i} about the level we marked earlier"
        if i % 4:
            dropped.append((text, 0.55))
            rejected += 1
        else:
            transcript.append(text)
            memory.record_utterance(text, topic=f"topic-{i % 30}")
            spoken += 1

    assert len(transcript) <= RETAINED_UTTERANCES
    assert len(dropped) <= RETAINED_DROPS
    assert len(memory.short_term) <= (memory.short_term.maxlen or 0)
    assert len(memory._index._vecs) <= (memory._index._vecs.maxlen or 0)
    assert len(memory.topics_last_seen) <= 30
    assert spoken + rejected == UTTERANCES * 5
    assert memory.utterance_count == spoken


def test_thousands_of_viewer_questions_stay_bounded():
    from intelligence.memory import QUESTION_TABLE_LIMIT, SessionMemory

    memory = SessionMemory("SOAK")
    for i in range(QUESTION_TABLE_LIMIT * 4):
        memory.record_question(f"question {i}")

    assert len(memory.audience_questions) <= QUESTION_TABLE_LIMIT


def test_metrics_do_not_grow_with_event_count():
    from runtime.health import Metrics

    m = Metrics()
    for i in range(50_000):
        m.inc("goldlive_utterances_total", {"session": "SESSION_001"})
        m.observe("goldlive_first_token_ms", float(i % 900))
        m.gauge("goldlive_queue_depth", float(i % 8), {"session": "SESSION_001"})

    assert len(m._counters) == 1
    assert len(m._gauges) == 1
    assert all(len(b) <= 2_000 for b in m._hist.values())
    assert len(m.render()) < 4_000, "the exposition must not grow with traffic"


# -- the event loop discipline the memory probe exposed --------------------


def test_synthesis_is_never_driven_by_a_fresh_event_loop_per_call():
    """Regression on the probe's own finding.

    Calling asyncio.run() per synthesis builds and tears down a loop every
    time, which made the audio path look like it leaked ~6 MB a call. Nothing
    in the runtime may do that: the session owns one loop for its lifetime.
    """
    import pathlib

    offenders = []
    for path in pathlib.Path(".").rglob("*.py"):
        s = str(path)
        if any(x in s for x in (".venv", "build\\", "dist\\", "tests\\", "scripts\\")):
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if "asyncio.run(" not in line or line.strip().startswith("#"):
                continue
            # Entry points legitimately start the one loop the process uses.
            # Entry points and the synchronous shutdown drain legitimately
            # own the loop for their process; nothing in a hot path may.
            if any(k in s for k in ("cli.py", "bootstrap.py", "setup_wizard.py",
                                    "readiness.py", "dryrun.py", "soak.py",
                                    "selftest.py", "store.py")):
                continue
            offenders.append(f"{s}:{lineno}")

    assert not offenders, (
        "asyncio.run() outside an entry point creates a loop per call: "
        + ", ".join(offenders)
    )


@pytest.mark.parametrize("calls", [200])
async def test_one_loop_serves_many_synthesis_calls(tmp_path, calls):
    """The intended shape: a single loop, reused."""
    tts = FakeTTS()
    loop = asyncio.get_running_loop()

    for i in range(calls):
        await tts.synthesize(f"utterance {i}", "v", tmp_path / f"s{i % 4}.wav")
        assert asyncio.get_running_loop() is loop, "the loop must not be replaced"

    assert len(tts.calls) == calls
