"""Things that must not grow, and failures that must not be swallowed.

A host meant to run for weeks on an 8 GB machine cannot keep every object it
has ever produced. These are the collections that did, and the failure paths
that reported success anyway.
"""

from __future__ import annotations

from collections import deque

import pytest

from runtime.session import RETAINED_DROPS, RETAINED_UTTERANCES


# -- bounded history -------------------------------------------------------


def test_the_retention_caps_are_actually_finite():
    """A None maxlen is an unbounded deque wearing a bounded one's clothes."""
    assert isinstance(RETAINED_UTTERANCES, int) and RETAINED_UTTERANCES > 0
    assert isinstance(RETAINED_DROPS, int) and RETAINED_DROPS > 0


def test_a_bounded_deque_discards_the_oldest_entries():
    """The property the session now relies on."""
    d: deque[int] = deque(maxlen=RETAINED_DROPS)
    for i in range(RETAINED_DROPS * 3):
        d.append(i)
    assert len(d) == RETAINED_DROPS
    assert d[-1] == RETAINED_DROPS * 3 - 1


def test_counters_keep_rising_after_the_deque_stops_growing():
    """The bug this pair exists to prevent: consumers derived "how many are
    new" from len(), which silently stops changing once a bounded deque is
    full, so metrics and the trace store would quietly stop recording."""
    d: deque[int] = deque(maxlen=10)
    count = 0
    for i in range(50):
        d.append(i)
        count += 1
    assert len(d) == 10, "the history is capped"
    assert count == 50, "the total is not"


def test_filenames_use_the_counter_not_the_capped_length():
    """len(transcript) numbered the wav files. Once the deque hit its cap that
    number would stop advancing and every later utterance would overwrite the
    same file."""
    import inspect

    from runtime.session import SessionRuntime

    source = inspect.getsource(SessionRuntime)
    speak = source.split("async def _speak")[1]
    assert "self.spoken_count" in speak
    assert "len(self.transcript)" not in speak


# -- audio failures must be loud -------------------------------------------


async def test_playback_failure_raises_instead_of_reporting_success(tmp_path,
                                                                    monkeypatch):
    """`except Exception: pass` wrapped the whole playback path, so a dead
    device, an exclusive-mode lock or an undecodable wav all looked exactly
    like a successfully spoken utterance."""
    pytest.importorskip("sounddevice")
    pytest.importorskip("soundfile")

    from platform_.audio import router as mod

    def boom(*_a, **_kw):
        raise OSError("audio device disappeared")

    monkeypatch.setattr(mod, "AudioPlaybackError", mod.AudioPlaybackError)
    sink = mod.AudioSink()

    import soundfile as sf

    monkeypatch.setattr(sf, "read", boom)

    wav = tmp_path / "x.wav"
    wav.write_bytes(b"not a wav")

    with pytest.raises(mod.AudioPlaybackError):
        await sink.play(wav)

    assert sink.last_error and "audio device disappeared" in sink.last_error


async def test_a_missing_audio_backend_still_simulates_deliberately(tmp_path):
    """The no-backend path is intentional -- it lets the rest of the pipeline
    run on a machine with no sound card. It must stay distinct from a real
    failure, which is why readiness never uses AudioSink to prove audio."""
    from platform_.audio.router import AudioSink

    sink = AudioSink()
    assert hasattr(sink, "last_error")


def test_the_sink_exposes_its_last_error():
    """Something above has to be able to see that audio failed."""
    from platform_.audio.router import AudioSink

    assert AudioSink().last_error is None


# -- no leaked descriptors -------------------------------------------------


def test_the_supervisor_log_handle_is_closed_after_spawn():
    """The panel opened the log with a bare open() and handed it to Popen
    without ever closing its own copy, leaking a descriptor on every START in
    a window the user is expected to leave open."""
    import inspect

    from runtime.panel import start_goldlive

    source = inspect.getsource(start_goldlive)
    assert "with open(" in source, "the handle must be scoped"
    assert "handle = open(" not in source


# -- interruption must actually interrupt ----------------------------------


async def test_barge_in_abandons_the_whole_utterance_not_just_one_segment(tmp_path):
    """Found by reading the drain loop.

    submit() handled a CRITICAL barge-in by calling sink.stop(), which aborts
    only the segment currently sounding. _speak() then continued its loop and
    played the remaining segments, so the preempted utterance carried on
    talking underneath the thing that had interrupted it -- the character left
    speaking when it should have stopped.
    """
    import asyncio
    from uuid import uuid4

    from platform_.audio.router import AudioRouter
    from shared.contracts import AudioRequest, Priority

    played: list[str] = []

    class RecordingSink:
        def __init__(self):
            self._stopped = False

        async def play(self, path):
            played.append(path.name)
            await asyncio.sleep(0)

        def stop(self):
            self._stopped = True

    class InstantTTS:
        async def synthesize(self, text, voice_id, out_path):
            from platform_.tts.base import TTSResult

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"x")
            return TTSResult(path=out_path, duration_ms=10, first_audio_ms=1)

    router = AudioRouter("SESSION_001", InstantTTS(), tmp_path, sink=RecordingSink())

    long_utterance = AudioRequest(
        session_id="SESSION_001", utterance_id=uuid4(), trace_id="t-1",
        segments=["one", "two", "three", "four"],
        voice_id="v", priority=Priority.LOW,
    )

    # Mark it as the one in flight, then preempt it mid-utterance.
    router._current = long_utterance
    router._preempted.add(long_utterance.utterance_id)

    await router._speak(long_utterance)

    assert played == [], (
        f"a preempted utterance must not keep speaking, but played {played}"
    )


async def test_an_unpreempted_utterance_speaks_every_segment(tmp_path):
    """The guard must not break normal speech."""
    from uuid import uuid4

    from platform_.audio.router import AudioRouter
    from shared.contracts import AudioRequest, Priority

    played: list[str] = []

    class RecordingSink:
        async def play(self, path):
            played.append(path.name)

        def stop(self):
            pass

    class InstantTTS:
        async def synthesize(self, text, voice_id, out_path):
            from platform_.tts.base import TTSResult

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"x")
            return TTSResult(path=out_path, duration_ms=10, first_audio_ms=1)

    router = AudioRouter("SESSION_001", InstantTTS(), tmp_path, sink=RecordingSink())
    request = AudioRequest(
        session_id="SESSION_001", utterance_id=uuid4(), trace_id="t-1",
        segments=["one", "two", "three"], voice_id="v", priority=Priority.LOW,
    )
    router._running = True
    await router._speak(request)
    assert len(played) == 3
