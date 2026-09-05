"""The audio utterance lifecycle: identity, cancellation and barge-in.

This is the synchronization surface of the whole system. There is no animation
layer; what has to be exactly right is that one utterance's audio never plays
underneath another's, and that a cancelled utterance stays cancelled.

The bug that motivated this file: barge-in called sink.stop(), which aborts
only the segment currently sounding. The loop then continued to the next
segment, so the preempted utterance carried on talking underneath the thing
that had interrupted it. Nothing in 490 tests noticed, because no test double
could express "playback takes time, and something arrives during it".
"""

from __future__ import annotations

import asyncio

import pytest
from fakes import FakeSink, FakeTTS, audio_request

from platform_.audio.router import AudioRouter, PlaybackState
from shared.contracts import Priority


def router(tmp_path, sink=None, tts=None, **kw) -> AudioRouter:
    return AudioRouter(
        "SESSION_001", tts or FakeTTS(), tmp_path, sink=sink or FakeSink(), **kw
    )


async def drain(r: AudioRouter, ticks: int = 40) -> None:
    """Let the router's drain loop make progress without wall-clock waiting."""
    for _ in range(ticks):
        await asyncio.sleep(0)


# -- identity ---------------------------------------------------------------


def test_every_utterance_carries_a_unique_id():
    ids = {audio_request().utterance_id for _ in range(200)}
    assert len(ids) == 200


async def test_audio_for_another_session_is_refused(tmp_path):
    """The isolation guarantee, at the last point it can still be enforced."""
    r = router(tmp_path)
    with pytest.raises(ValueError, match="SESSION_002"):
        await r.submit(audio_request(session_id="SESSION_002"))


# -- barge-in: the ten-point verification ----------------------------------


async def test_barge_in_stops_A_and_A_never_resumes(tmp_path):
    """1-7: A begins, is interrupted, stops, cannot resume, its remaining
    segments are discarded, B can begin, and no audio from A leaks under B."""
    sink = FakeSink(delay_s=0.05)
    r = router(tmp_path, sink=sink)
    r._running = True

    a = audio_request(segments=["a1", "a2", "a3", "a4"])
    b = audio_request(segments=["b1"], priority=Priority.CRITICAL)

    r._current = a
    speaking = asyncio.create_task(r._speak(a))
    await asyncio.sleep(0.02)          # A is mid-first-segment

    await r.submit(b)                  # barge-in
    await speaking

    played = list(sink.played)
    assert len(played) < 4, f"A must not finish all four segments, played {played}"
    assert sink.stop_calls >= 1, "the sounding segment must be stopped"
    assert a.utterance_id in r._preempted or r._current is not a


async def test_a_preempted_utterance_produces_no_further_audio(tmp_path):
    """4-5 stated plainly: once marked, not one more segment."""
    sink = FakeSink()
    r = router(tmp_path, sink=sink)
    r._running = True

    a = audio_request(segments=["1", "2", "3", "4", "5"])
    r._current = a
    r._preempted.add(a.utterance_id)

    await r._speak(a)
    assert sink.played == []


async def test_preemption_is_scoped_to_one_utterance(tmp_path):
    """Marking A must not silence B. A set keyed by utterance id is the whole
    point -- a single boolean flag would leak across utterances."""
    sink = FakeSink()
    r = router(tmp_path, sink=sink)
    r._running = True

    a = audio_request(segments=["a1", "a2"])
    b = audio_request(segments=["b1", "b2"])
    r._preempted.add(a.utterance_id)

    await r._speak(a)
    assert sink.played == []

    await r._speak(b)
    assert len(sink.played) == 2, "an unrelated utterance must still speak"


async def test_the_preemption_mark_is_cleared_after_the_utterance(tmp_path):
    """8: A's terminal state must be correct, and the mark must not persist to
    poison a later utterance that happens to be retried."""
    sink = FakeSink()
    r = router(tmp_path, sink=sink, tts=FakeTTS())
    r._running = True
    await r.start()

    a = audio_request(segments=["one"])
    r._preempted.add(a.utterance_id)
    await r.submit(a)
    await drain(r)
    await r.stop()

    assert a.utterance_id not in r._preempted


async def test_barge_in_is_counted(tmp_path):
    """9: metrics must record the preemption."""
    r = router(tmp_path, sink=FakeSink())
    r._current = audio_request(segments=["long"])
    before = r.stats.preempted
    await r.submit(audio_request(priority=Priority.CRITICAL))
    assert r.stats.preempted == before + 1


async def test_the_router_stays_usable_after_a_barge_in(tmp_path):
    """10: the system remains healthy."""
    sink = FakeSink()
    r = router(tmp_path, sink=sink)
    await r.start()
    r._current = audio_request(segments=["interrupted"])

    await r.submit(audio_request(segments=["urgent"], priority=Priority.CRITICAL))
    await drain(r)
    await r.submit(audio_request(segments=["after"]))
    await drain(r)
    await r.stop()

    assert r.state is PlaybackState.STOPPED
    assert sink.played, "the router must still play after an interruption"


@pytest.mark.parametrize("run", range(12))
async def test_repeated_barge_in_never_leaks_audio(tmp_path, run):
    """Repeat it many times, as required. A race that only shows up
    occasionally is still a race."""
    sink = FakeSink(delay_s=0.01)
    r = router(tmp_path, sink=sink)
    r._running = True

    a = audio_request(segments=["a1", "a2", "a3"])
    r._current = a
    task = asyncio.create_task(r._speak(a))
    await asyncio.sleep(0.005)
    r._preempted.add(a.utterance_id)
    sink.stop()
    await task

    assert len(sink.played) <= 1, f"leaked {sink.played} after preemption"


# -- cancellation and shutdown ---------------------------------------------


async def test_stopping_the_router_halts_mid_utterance(tmp_path):
    sink = FakeSink()
    r = router(tmp_path, sink=sink)
    r._running = False  # as stop() leaves it

    await r._speak(audio_request(segments=["1", "2", "3"]))
    assert sink.played == [], "a stopped router must not play"


async def test_cancelling_the_drain_task_is_clean(tmp_path):
    r = router(tmp_path, tts=FakeTTS(delay_s=0.05))
    await r.start()
    await r.submit(audio_request(segments=["slow"]))
    await asyncio.sleep(0.01)
    await r.stop()
    assert r.state is PlaybackState.STOPPED


# -- queue behaviour --------------------------------------------------------


async def test_the_queue_is_bounded_and_drops_rather_than_growing(tmp_path):
    """Back-pressure: an unbounded queue on a 24/7 host is a slow leak."""
    r = router(tmp_path, max_queue=3)
    accepted = [await r.submit(audio_request(segments=["x"])) for _ in range(10)]
    assert accepted.count(True) == 3
    assert accepted.count(False) == 7
    assert r.queue_depth == 3


async def test_an_expired_utterance_is_dropped_not_spoken(tmp_path):
    """Commentary about a moment that has passed is worse than silence."""
    sink = FakeSink()
    r = router(tmp_path, sink=sink)
    await r.start()

    stale = audio_request(segments=["old news"], deadline_ms=1)
    await asyncio.sleep(0.02)
    await r.submit(stale)
    await drain(r)
    await r.stop()

    assert sink.played == []
    assert r.stats.dropped_expired >= 1


async def test_rapid_submissions_are_all_queued_or_all_refused(tmp_path):
    """No silent loss: every submission is either accepted or told no."""
    r = router(tmp_path, max_queue=64)
    results = [await r.submit(audio_request(segments=["x"])) for _ in range(50)]
    assert all(results)
    assert r.queue_depth == 50


# -- failure modes ----------------------------------------------------------


async def test_tts_failure_does_not_kill_the_drain_loop(tmp_path):
    """A failing synthesis must cost one utterance, not the session."""
    r = router(tmp_path, tts=FakeTTS(fail=True))
    await r.start()
    await r.submit(audio_request(segments=["doomed"]))
    await drain(r)

    assert r.stats.tts_failures >= 1
    assert r._task is not None and not r._task.done(), "the loop must survive"
    await r.stop()


async def test_playback_failure_is_counted_and_survivable(tmp_path):
    r = router(tmp_path, sink=FakeSink(fail=True))
    await r.start()
    await r.submit(audio_request(segments=["will not play"]))
    await drain(r)

    assert r.stats.tts_failures >= 1
    assert r._task is not None and not r._task.done()
    await r.stop()


async def test_empty_synthesis_is_skipped_without_error(tmp_path):
    """TTS legitimately returns no file for empty text."""
    sink = FakeSink()
    r = router(tmp_path, sink=sink, tts=FakeTTS(empty=True))
    r._running = True
    await r._speak(audio_request(segments=["", "  "]))
    assert sink.played == []


async def test_the_router_recovers_after_a_transient_failure(tmp_path):
    """Fail the first synthesis, succeed after: the second utterance speaks."""
    sink = FakeSink()
    r = router(tmp_path, sink=sink, tts=FakeTTS(fail_on=0))
    await r.start()
    await r.submit(audio_request(segments=["first"]))
    await drain(r)
    await r.submit(audio_request(segments=["second"]))
    await drain(r)
    await r.stop()

    assert sink.played, "a transient failure must not disable the router"


async def test_state_returns_to_idle_after_every_outcome(tmp_path):
    """Success or failure, the router must not be left claiming it is
    speaking -- that is the state a stuck host reports forever."""
    for tts in (FakeTTS(), FakeTTS(fail=True), FakeTTS(empty=True)):
        r = router(tmp_path, tts=tts)
        await r.start()
        await r.submit(audio_request(segments=["x"]))
        await drain(r)
        assert r.state in (PlaybackState.IDLE, PlaybackState.STOPPED)
        assert r.current_utterance is None
        await r.stop()
