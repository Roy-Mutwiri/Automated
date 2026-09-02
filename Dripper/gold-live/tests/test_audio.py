"""Audio router: queueing, barge-in, deadlines, isolation and failure."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from platform_.audio.devices import AudioDevice
from platform_.audio.router import AudioRouter, AudioSink, PlaybackState
from platform_.tts.base import TTSProvider, TTSResult
from shared.contracts import AudioRequest, Priority, utcnow


class RecordingTTS(TTSProvider):
    def __init__(self, fail_on: str | None = None, delay_s: float = 0.0) -> None:
        self.spoken: list[str] = []
        self.fail_on = fail_on
        self.delay_s = delay_s

    async def synthesize(self, text: str, voice_id: str, out_path: Path) -> TTSResult:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("tts exploded")
        self.spoken.append(text)
        return TTSResult(path=None, duration_ms=100, first_audio_ms=25)


class SilentSink(AudioSink):
    def __init__(self) -> None:
        super().__init__()
        self.played: list[Path] = []
        self.stops = 0

    async def play(self, path: Path) -> None:
        self.played.append(path)

    def stop(self) -> None:
        self.stops += 1
        super().stop()


def request(
    session_id: str = "SESSION_001",
    segments: list[str] | None = None,
    priority: Priority = Priority.MEDIUM,
    deadline_ms: int = 30_000,
) -> AudioRequest:
    return AudioRequest(
        utterance_id=uuid4(), session_id=session_id, trace_id="t",
        segments=segments or ["First sentence.", "Second sentence."],
        voice_id="v", priority=priority, deadline_ms=deadline_ms,
    )


async def router(tts: TTSProvider, tmp_path: Path) -> tuple[AudioRouter, SilentSink]:
    sink = SilentSink()
    r = AudioRouter("SESSION_001", tts, tmp_path, sink=sink)
    await r.start()
    return r, sink


async def drain(r: AudioRouter, timeout: float = 1.0) -> None:
    """Wait for the queue to empty and playback to finish."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if r.queue_depth == 0 and not r.is_speaking:
            await asyncio.sleep(0.02)
            if r.queue_depth == 0 and not r.is_speaking:
                return
        await asyncio.sleep(0.01)


# -- basic playback --------------------------------------------------------


async def test_speaks_every_segment_in_order(tmp_path):
    tts = RecordingTTS()
    r, _sink = await router(tts, tmp_path)
    await r.submit(request(segments=["One.", "Two.", "Three."]))
    await drain(r)
    await r.stop()
    assert tts.spoken == ["One.", "Two.", "Three."]
    assert r.stats.played == 1


async def test_state_returns_to_idle(tmp_path):
    r, _ = await router(RecordingTTS(), tmp_path)
    await r.submit(request())
    await drain(r)
    assert r.state is PlaybackState.IDLE
    await r.stop()
    assert r.state is PlaybackState.STOPPED


# -- isolation -------------------------------------------------------------


async def test_router_rejects_another_sessions_audio(tmp_path):
    r, _ = await router(RecordingTTS(), tmp_path)
    with pytest.raises(ValueError, match=r"SESSION_009.*SESSION_001"):
        await r.submit(request(session_id="SESSION_009"))
    await r.stop()


# -- deadlines -------------------------------------------------------------


async def test_expired_utterance_is_dropped_not_spoken(tmp_path):
    """A reaction that arrives forty seconds late is worse than silence."""
    tts = RecordingTTS()
    r, _ = await router(tts, tmp_path)

    stale = request(deadline_ms=1)
    object.__setattr__(stale, "created_at", utcnow() - timedelta(seconds=30))
    await r.submit(stale)
    await drain(r)
    await r.stop()

    assert tts.spoken == []
    assert r.stats.dropped_expired == 1
    assert r.stats.played == 0


async def test_fresh_utterance_is_not_dropped(tmp_path):
    tts = RecordingTTS()
    r, _ = await router(tts, tmp_path)
    await r.submit(request(deadline_ms=30_000))
    await drain(r)
    await r.stop()
    assert tts.spoken and r.stats.dropped_expired == 0


# -- barge-in --------------------------------------------------------------


async def test_critical_utterance_preempts_playback(tmp_path):
    tts = RecordingTTS(delay_s=0.05)
    r, sink = await router(tts, tmp_path)

    await r.submit(request(segments=["Long one."] * 4))
    await asyncio.sleep(0.06)  # let it start
    await r.submit(request(segments=["Urgent."], priority=Priority.CRITICAL))
    await drain(r, timeout=2.0)
    await r.stop()

    assert r.stats.preempted == 1
    assert sink.stops >= 1


async def test_non_critical_does_not_preempt(tmp_path):
    tts = RecordingTTS(delay_s=0.05)
    r, _sink = await router(tts, tmp_path)

    await r.submit(request(segments=["Long one."] * 3))
    await asyncio.sleep(0.06)
    await r.submit(request(segments=["Also important."], priority=Priority.HIGH))
    await drain(r, timeout=2.0)
    await r.stop()

    assert r.stats.preempted == 0, "only CRITICAL may interrupt"


# -- failure ---------------------------------------------------------------


async def test_tts_failure_does_not_stop_the_router(tmp_path):
    """One bad utterance must not take audio down for the session."""
    tts = RecordingTTS(fail_on="BOOM")
    r, _ = await router(tts, tmp_path)

    await r.submit(request(segments=["BOOM"]))
    await drain(r)
    await r.submit(request(segments=["Recovered."]))
    await drain(r)
    await r.stop()

    assert r.stats.tts_failures == 1
    assert "Recovered." in tts.spoken


async def test_queue_full_drops_rather_than_blocking(tmp_path):
    sink = SilentSink()
    r = AudioRouter("SESSION_001", RecordingTTS(delay_s=0.3), tmp_path,
                    sink=sink, max_queue=2)
    await r.start()
    accepted = [await r.submit(request()) for _ in range(6)]
    await r.stop()
    assert accepted.count(False) >= 1, "must shed load rather than block the caller"


# -- device discovery ------------------------------------------------------


@pytest.mark.parametrize(
    "name,virtual",
    [
        ("CABLE Input (VB-Audio Virtual Cable)", True),
        ("VoiceMeeter Aux Input", True),
        ("Speakers (Realtek High Definition Audio)", False),
        ("Headphones", False),
    ],
)
def test_virtual_cable_detection(name, virtual):
    d = AudioDevice(index=0, name=name, channels=2, sample_rate=48000)
    assert d.looks_virtual is virtual


def test_snapshot_shape(tmp_path):
    r = AudioRouter("SESSION_001", RecordingTTS(), tmp_path, sink=SilentSink())
    snap = r.snapshot()
    assert snap["session_id"] == "SESSION_001"
    assert {"state", "queue_depth", "played", "tts_failures"} <= snap.keys()
