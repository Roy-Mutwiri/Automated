"""The audio path under a delivery plan: prefetch, planned silence, and the
guarantees that must survive both.

The naturalness work added two things that can go wrong in ways ordinary tests
would not catch. Prefetching starts a synthesis task that may never be played,
so it can leak. Planned pauses put deliberate silence into the speaking loop,
so they can hold an interruption. Both are checked here against wall-clock
behaviour rather than against structure.
"""

from __future__ import annotations

import asyncio
import time
import wave
from pathlib import Path

import pytest
from fakes import audio_request

from platform_.audio.router import (
    PAUSE_SLICE_S,
    REACTION_GAP_MS,
    AudioRouter,
    plan_units,
)
from platform_.tts.base import TTSResult
from shared.delivery import NULL_CAPABILITIES, PIPER_CAPABILITIES, DeliveryPlan


class ClockTTS:
    """Records when each synthesis started and finished, and how long the
    render took, so overlap with playback can be proven rather than assumed."""

    capabilities = PIPER_CAPABILITIES

    def __init__(self, synth_s: float = 0.05) -> None:
        self.synth_s = synth_s
        self.events: list[tuple[str, str, float]] = []

    async def synthesize(self, text, voice_id, out_path, *, prosody=None):
        self.events.append(("synth_start", text, time.perf_counter()))
        await asyncio.sleep(self.synth_s)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_silent_wav(out_path, seconds=0.05)
        self.events.append(("synth_end", text, time.perf_counter()))
        return TTSResult(path=out_path, duration_ms=50, first_audio_ms=5)


class ClockSink:
    def __init__(self, play_s: float = 0.15) -> None:
        self.play_s = play_s
        self.events: list[tuple[str, str, float]] = []
        self.played: list[str] = []
        self.stop_calls = 0
        self._stop = asyncio.Event()

    async def play(self, path: Path) -> None:
        self._stop.clear()
        self.events.append(("play_start", path.name, time.perf_counter()))
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=self.play_s)
        except asyncio.TimeoutError:
            pass
        self.played.append(path.name)
        self.events.append(("play_end", path.name, time.perf_counter()))

    def stop(self) -> None:
        self.stop_calls += 1
        self._stop.set()


def _write_silent_wav(path: Path, seconds: float, rate: int = 22050) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def at(events, kind: str, needle: str) -> float:
    for k, label, t in events:
        if k == kind and needle in label:
            return t
    raise AssertionError(f"no {kind} for {needle!r} in {[e[:2] for e in events]}")


# -- prefetch ---------------------------------------------------------------


async def test_the_next_chunk_is_rendered_while_the_current_one_plays(tmp_path):
    """The fix for ~562ms of dead air in the middle of one utterance.

    Rendering strictly in sequence put the next chunk's synthesis time inside
    the gap. Piper's real-time factor is ~0.05, so there is ample room to
    render during playback -- this asserts we actually do.
    """
    tts, sink = ClockTTS(synth_s=0.05), ClockSink(play_s=0.15)
    r = AudioRouter("SESSION_001", tts, tmp_path, sink=sink)
    r._running = True
    await r._speak(audio_request(segments=["One.", "Two.", "Three."]))

    first_play_end = at(sink.events, "play_end", "_00")
    tail_synth_start = at(tts.events, "synth_start", "Two. Three.")
    assert tail_synth_start < first_play_end, (
        "the tail must be rendered during playback of the lead, not after it"
    )


async def test_a_single_chunk_utterance_starts_no_prefetch(tmp_path):
    tts, sink = ClockTTS(), ClockSink(play_s=0.02)
    r = AudioRouter("SESSION_001", tts, tmp_path, sink=sink)
    r._running = True
    await r._speak(audio_request(segments=["Only one."]))
    assert sum(1 for k, _, _ in tts.events if k == "synth_start") == 1


async def test_a_cancelled_prefetch_leaves_nothing_pinned(tmp_path):
    """A prefetch that is never played must not hold a file against the
    retention sweep for the life of the process."""
    tts, sink = ClockTTS(synth_s=0.2), ClockSink(play_s=0.02)
    r = AudioRouter("SESSION_001", tts, tmp_path, sink=sink)
    r._running = True

    request = audio_request(segments=["One.", "Two.", "Three."])
    speaking = asyncio.create_task(r._speak(request))
    await asyncio.sleep(0.05)
    r._preempted = request.utterance_id  # barge-in mid-utterance
    sink.stop()
    await speaking

    assert r._in_use == set(), f"files left pinned: {r._in_use}"


# -- planned silence must not hold an interruption --------------------------


async def test_a_planned_pause_does_not_delay_barge_in(tmp_path):
    """The constraint that outranks every expressive feature here.

    A 600ms pause slept in one go would hold an interruption for 600ms. It is
    slept in slices instead, so the worst case is one slice.
    """
    tts, sink = ClockTTS(synth_s=0.01), ClockSink(play_s=0.02)
    r = AudioRouter("SESSION_001", tts, tmp_path, sink=sink)
    r._running = True

    request = audio_request(segments=["One."])
    plan = DeliveryPlan(pause_before_ms=600).renderable_with(PIPER_CAPABILITIES)
    speaking = asyncio.create_task(r._speak(request, plan))

    await asyncio.sleep(0.05)  # we are inside the 600ms pause
    started = time.perf_counter()
    r._preempted = request.utterance_id
    sink.stop()
    await speaking
    elapsed = time.perf_counter() - started

    assert elapsed < 0.15, f"barge-in held for {elapsed * 1000:.0f}ms by a pause"
    assert not sink.played, "nothing should have been spoken after the barge-in"


async def test_a_long_tail_does_not_block_interruption(tmp_path):
    """Grouping made the tail one long span. Interruption is sink-level, so
    the span length must not matter."""
    tts = ClockTTS(synth_s=0.01)
    sink = ClockSink(play_s=10.0)  # a very long chunk
    r = AudioRouter("SESSION_001", tts, tmp_path, sink=sink)
    r._running = True

    request = audio_request(segments=["One.", "Two.", "Three.", "Four."])
    speaking = asyncio.create_task(r._speak(request))
    await asyncio.sleep(0.05)

    started = time.perf_counter()
    r._preempted = request.utterance_id
    sink.stop()
    await speaking
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"interruption took {elapsed * 1000:.0f}ms"


async def test_pause_slices_are_short_enough_to_be_invisible():
    """If this constant grows, the barge-in guarantee above weakens with it."""
    assert PAUSE_SLICE_S <= 0.02


# -- planned silence is actually inserted -----------------------------------


async def test_the_planned_gap_is_the_silence_between_chunks(tmp_path):
    tts, sink = ClockTTS(synth_s=0.01), ClockSink(play_s=0.02)
    r = AudioRouter("SESSION_001", tts, tmp_path, sink=sink)
    r._running = True

    plan = DeliveryPlan(gap_ms=200).renderable_with(PIPER_CAPABILITIES)
    await r._speak(audio_request(segments=["One.", "Two."]), plan)

    gap = at(sink.events, "play_start", "_01") - at(sink.events, "play_end", "_00")
    assert 0.15 < gap < 0.35, f"planned 200ms gap measured {gap * 1000:.0f}ms"


async def test_no_plan_means_ordinary_spacing(tmp_path):
    """Normal speech remains the default when nothing happened."""
    tts, sink = ClockTTS(synth_s=0.01), ClockSink(play_s=0.02)
    r = AudioRouter("SESSION_001", tts, tmp_path, sink=sink)
    r._running = True
    await r._speak(audio_request(segments=["One.", "Two."]))

    gap = at(sink.events, "play_start", "_01") - at(sink.events, "play_end", "_00")
    assert gap < 0.25, f"default gap measured {gap * 1000:.0f}ms"


# -- reaction units ---------------------------------------------------------


def test_a_reaction_becomes_its_own_unit_with_a_beat_after_it():
    plan = DeliveryPlan(reaction="Wait.", gap_ms=60).renderable_with(PIPER_CAPABILITIES)
    units = plan_units(plan, ["The level broke.", "That changes things."])
    assert units[0].text == "Wait."
    assert units[0].gap_after_ms >= REACTION_GAP_MS, "the beat is the point"
    assert [u.text for u in units[1:]] == [
        "The level broke.",
        "That changes things.",
    ]


def test_a_reaction_never_becomes_the_whole_utterance():
    """A reaction with nothing behind it is the shape a runaway reaction loop
    would take, and it would sound like the host saying "Whoa." then stopping."""
    plan = DeliveryPlan(reaction="Whoa.").renderable_with(PIPER_CAPABILITIES)
    assert plan_units(plan, []) == []
    assert plan_units(plan, ["", "  "]) == []


async def test_a_reaction_is_spoken_before_the_words(tmp_path):
    tts, sink = ClockTTS(synth_s=0.01), ClockSink(play_s=0.02)
    r = AudioRouter("SESSION_001", tts, tmp_path, sink=sink)
    r._running = True

    plan = DeliveryPlan(reaction="Whoa.").renderable_with(PIPER_CAPABILITIES)
    await r._speak(audio_request(segments=["The level broke."]), plan)

    spoken = [text for kind, text, _ in tts.events if kind == "synth_start"]
    assert spoken == ["Whoa.", "The level broke."]


async def test_a_backend_without_reactions_never_renders_one(tmp_path):
    class PlainTTS(ClockTTS):
        capabilities = NULL_CAPABILITIES

    tts, sink = PlainTTS(synth_s=0.01), ClockSink(play_s=0.02)
    r = AudioRouter("SESSION_001", tts, tmp_path, sink=sink)
    r._running = True

    plan = DeliveryPlan(reaction="Whoa.", pause_before_ms=500)
    await r._speak(audio_request(segments=["The level broke."]), plan)

    spoken = [text for kind, text, _ in tts.events if kind == "synth_start"]
    assert spoken == ["The level broke."]


# -- the existing guarantees still hold -------------------------------------


async def test_barge_in_still_discards_the_rest_of_the_utterance(tmp_path):
    tts, sink = ClockTTS(synth_s=0.01), ClockSink(play_s=0.1)
    r = AudioRouter("SESSION_001", tts, tmp_path, sink=sink)
    r._running = True

    a = audio_request(segments=["a1", "a2", "a3", "a4"])
    speaking = asyncio.create_task(r._speak(a))
    await asyncio.sleep(0.03)
    r._preempted = a.utterance_id
    sink.stop()
    await speaking

    assert len(sink.played) < 3
    assert sink.stop_calls >= 1


async def test_delivery_never_changes_what_words_are_spoken(tmp_path):
    """Expression is delivery. It must not add, drop or reorder content."""
    segments = ["First sentence.", "Second sentence.", "Third sentence."]
    for plan in (
        None,
        DeliveryPlan.neutral(),
        DeliveryPlan(pace=1.05, gap_ms=200, pause_before_ms=300),
    ):
        tts, sink = ClockTTS(synth_s=0.005), ClockSink(play_s=0.01)
        r = AudioRouter("SESSION_001", tts, tmp_path / str(id(plan)), sink=sink)
        r._running = True
        await r._speak(audio_request(segments=segments), plan)
        spoken = [t for k, t, _ in tts.events if k == "synth_start"]
        assert " ".join(spoken) == " ".join(segments)


@pytest.mark.parametrize("pace", [0.5, 1.0, 3.0])
async def test_pace_never_reaches_the_synthesiser_unbounded(tmp_path, pace):
    seen = []

    class Recorder(ClockTTS):
        async def synthesize(self, text, voice_id, out_path, *, prosody=None):
            seen.append(prosody)
            return await super().synthesize(text, voice_id, out_path, prosody=prosody)

    r = AudioRouter("SESSION_001", Recorder(0.005), tmp_path, sink=ClockSink(0.01))
    r._running = True
    await r._speak(
        audio_request(segments=["One."]), DeliveryPlan(pace=pace)
    )
    assert seen and all(0.80 <= p.length_scale <= 1.25 for p in seen)
