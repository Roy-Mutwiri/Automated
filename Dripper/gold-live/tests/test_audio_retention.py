"""Spoken audio must not accumulate forever.

The defect these cover was measured, not imagined: 5,446 wav files and 1.75 GB
in three days, because every utterance was written to disk and nothing ever
deleted one. At the production cadence that is roughly 28 GB a month on a
system whose whole point is running continuously.

The risk in fixing it is deleting a file someone is still using, so most of
these tests are about what must *survive* a sweep rather than what must go.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fakes import FakeSink, FakeTTS, audio_request

from platform_.audio.retention import (
    AudioRetention,
    RetentionPolicy,
)
from platform_.audio.router import RETAINED_LATENCIES, AudioRouter
from platform_.tts.base import TTSResult
from shared.contracts import Priority


def wav(directory: Path, name: str, size: int = 1024, age_s: float = 0.0) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"\0" * size)
    if age_s:
        old = time.time() - age_s
        import os

        os.utime(path, (old, old))
    return path


# -- the policy in isolation ------------------------------------------------


def test_files_older_than_the_age_bound_are_removed(tmp_path):
    wav(tmp_path, "old.wav", age_s=10_000)
    wav(tmp_path, "new.wav", age_s=10)
    result = AudioRetention(tmp_path, RetentionPolicy(max_age_s=3600)).sweep()

    assert result.deleted == 1
    assert (tmp_path / "new.wav").exists()
    assert not (tmp_path / "old.wav").exists()


def test_the_size_bound_evicts_oldest_first(tmp_path):
    for i in range(10):
        wav(tmp_path, f"{i:02d}.wav", size=1000, age_s=1000 - i * 10)
    policy = RetentionPolicy(max_age_s=10_000, max_bytes=5000)
    result = AudioRetention(tmp_path, policy).sweep()

    survivors = sorted(p.name for p in tmp_path.glob("*.wav"))
    assert result.kept_bytes <= 5000
    # The newest five survive; the oldest five are the ones evicted.
    assert survivors == ["05.wav", "06.wav", "07.wav", "08.wav", "09.wav"]


def test_a_protected_file_is_never_deleted_by_either_bound(tmp_path):
    """The file being written or played must survive, however old or large."""
    active = wav(tmp_path, "active.wav", size=10_000, age_s=99_999)
    wav(tmp_path, "other.wav", size=10_000, age_s=99_999)

    policy = RetentionPolicy(max_age_s=1, max_bytes=1)
    result = AudioRetention(tmp_path, policy).sweep(protect={active})

    assert active.exists(), "an in-use file was deleted"
    assert not (tmp_path / "other.wav").exists()
    assert result.protected == 1


def test_a_missing_directory_is_not_an_error(tmp_path):
    result = AudioRetention(tmp_path / "nope", RetentionPolicy()).sweep()
    assert result.deleted == 0 and result.kept == 0


def test_a_file_that_cannot_be_removed_does_not_raise(tmp_path, monkeypatch):
    """Reclaiming disk must never take the session down."""
    wav(tmp_path, "locked.wav", age_s=99_999)

    def refuse(self):
        raise PermissionError("file is in use by another process")

    monkeypatch.setattr(Path, "unlink", refuse)
    result = AudioRetention(tmp_path, RetentionPolicy(max_age_s=1)).sweep()

    assert result.errors, "the failure must be recorded, not swallowed"
    assert result.deleted == 0


def test_sweeping_is_idempotent(tmp_path):
    wav(tmp_path, "keep.wav", age_s=10)
    retention = AudioRetention(tmp_path, RetentionPolicy(max_age_s=3600))
    first = retention.sweep()
    second = retention.sweep()
    assert first.kept == second.kept == 1
    assert second.deleted == 0


def test_only_wav_files_are_touched(tmp_path):
    wav(tmp_path, "audio.wav", age_s=99_999)
    (tmp_path / "transcript.jsonl").write_text("{}", encoding="utf-8")
    AudioRetention(tmp_path, RetentionPolicy(max_age_s=1)).sweep()
    assert (tmp_path / "transcript.jsonl").exists()


# -- the router applying it -------------------------------------------------


async def drain(router: AudioRouter, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if router.queue_depth == 0 and not router.is_speaking:
            return
        await asyncio.sleep(0.01)


async def test_sustained_speech_does_not_grow_the_directory_without_bound(tmp_path):
    """The exact failure that was measured: speak a lot, watch the disk.

    Samples throughout rather than only at the end. An earlier version of this
    test checked the final size alone and passed while the frozen build was
    overshooting a 4 MB ceiling to 9.21 MB between sweeps -- the peak is the
    thing the cap is supposed to bound, not the value it happens to land on.
    """
    policy = RetentionPolicy(max_age_s=86_400, max_bytes=20_000)
    router = AudioRouter("SESSION_001", FakeTTS(), tmp_path,
                         sink=FakeSink(), retention=policy)
    await router.start()

    peak = 0
    for i in range(300):
        await router.submit(audio_request(segments=[f"line {i}"]))
        await drain(router)
        peak = max(peak, router.retention.usage()[1])
    await router.stop()

    count, total = router.retention.usage()
    # The cap plus one utterance in flight: a file cannot be evicted before it
    # exists. Anything beyond that is the policy failing to keep up.
    one_utterance = 4096
    assert peak <= policy.max_bytes + one_utterance, f"peaked at {peak} bytes"
    assert total <= policy.max_bytes, f"{count} files, {total} bytes on disk"


async def test_startup_reclaims_audio_left_by_a_previous_session(tmp_path):
    """A hard-killed supervisor cannot clean up; the next start must."""
    out = tmp_path / "SESSION_001"
    wav(out, "00001_dead_00.wav", age_s=99_999)
    wav(out, "00002_dead_01.wav", age_s=99_999)

    router = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=FakeSink(),
                         retention=RetentionPolicy(max_age_s=3600))
    await router.start()
    await router.stop()

    assert list(out.glob("*.wav")) == []


async def test_shutdown_sweeps_what_the_last_utterances_left(tmp_path):
    out = tmp_path / "SESSION_001"
    router = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=FakeSink(),
                         retention=RetentionPolicy(max_age_s=0.0))
    await router.start()
    await router.submit(audio_request(segments=["one"]))
    await drain(router)
    await router.stop()

    assert list(out.glob("*.wav")) == [], "shutdown left audio behind"


async def test_a_failed_synthesis_leaves_no_permanent_file(tmp_path):
    class Failing(FakeTTS):
        async def synthesize(self, text, voice_id, out_path, *, prosody=None):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"\0" * 512)   # a partial render
            raise RuntimeError("synthesis died")

    out = tmp_path / "SESSION_001"
    router = AudioRouter("SESSION_001", Failing(), tmp_path, sink=FakeSink(),
                         retention=RetentionPolicy(max_age_s=0.0))
    await router.start()
    await router.submit(audio_request(segments=["one"]))
    await drain(router)
    await router.stop()

    assert router.stats.tts_failures == 1
    assert list(out.glob("*.wav")) == [], "a failed render left a file behind"


async def test_a_failed_playback_leaves_no_permanent_file(tmp_path):
    class BadSink(FakeSink):
        async def play(self, path):
            raise RuntimeError("device gone")

    out = tmp_path / "SESSION_001"
    router = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=BadSink(),
                         retention=RetentionPolicy(max_age_s=0.0))
    await router.start()
    await router.submit(audio_request(segments=["one"]))
    await drain(router)
    await router.stop()

    assert list(out.glob("*.wav")) == []


async def test_an_interrupted_utterance_leaves_no_permanent_file(tmp_path):
    out = tmp_path / "SESSION_001"
    router = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=FakeSink(),
                         retention=RetentionPolicy(max_age_s=0.0))
    await router.start()
    for i in range(20):
        await router.submit(audio_request(segments=[f"long {i}", "second", "third"]))
        await router.submit(audio_request(segments=[f"urgent {i}"],
                                          priority=Priority.CRITICAL))
    await drain(router, timeout_s=10)
    await router.stop()

    assert list(out.glob("*.wav")) == []


async def test_an_empty_segment_creates_no_output_file(tmp_path):
    class EmptyResult(FakeTTS):
        async def synthesize(self, text, voice_id, out_path, *, prosody=None):
            return TTSResult(path=None, duration_ms=0, first_audio_ms=0)

    out = tmp_path / "SESSION_001"
    router = AudioRouter("SESSION_001", EmptyResult(), tmp_path, sink=FakeSink(),
                         retention=RetentionPolicy(max_age_s=86_400))
    await router.start()
    await router.submit(audio_request(segments=[""]))
    await drain(router)
    await router.stop()

    assert list(out.glob("*.wav")) == []


async def test_one_session_cannot_delete_another_sessions_audio(tmp_path):
    other = tmp_path / "SESSION_002"
    keeper = wav(other, "00001_other_00.wav", age_s=99_999)

    router = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=FakeSink(),
                         retention=RetentionPolicy(max_age_s=1))
    await router.start()
    await router.stop()

    assert keeper.exists(), "a session swept another session's directory"


async def test_the_file_being_played_is_protected_during_playback(tmp_path):
    """A sweep landing mid-playback must not unlink the file in flight."""
    swept: list[int] = []

    class SweepingSink(FakeSink):
        async def play(self, path):
            result = router.retention.sweep(protect=router._in_use)
            swept.append(result.deleted)
            assert path.exists(), "the file being played was deleted"
            await super().play(path)

    router = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=SweepingSink(),
                         retention=RetentionPolicy(max_age_s=0.0, max_bytes=0))
    await router.start()
    await router.submit(audio_request(segments=["one"]))
    await drain(router)
    await router.stop()

    assert swept, "the sweep never ran during playback"


# -- the latency metric -----------------------------------------------------


async def test_latency_samples_are_bounded(tmp_path):
    router = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=FakeSink())
    await router.start()
    for i in range(RETAINED_LATENCIES * 3):
        await router.submit(audio_request(segments=[f"line {i}"]))
        await drain(router)
    await router.stop()

    assert len(router.stats.first_audio_ms) <= RETAINED_LATENCIES


def test_the_median_still_works_on_a_bounded_window():
    from platform_.audio.router import RouterStats

    stats = RouterStats()
    for value in (30, 10, 20):
        stats.first_audio_ms.append(value)
    assert stats.median_first_audio_ms == 20


def test_the_median_of_no_samples_is_zero():
    from platform_.audio.router import RouterStats

    assert RouterStats().median_first_audio_ms == 0.0


@pytest.mark.parametrize("count", [1, 10, RETAINED_LATENCIES * 2])
def test_latency_collection_never_exceeds_its_cap(count):
    from platform_.audio.router import RouterStats

    stats = RouterStats()
    for i in range(count):
        stats.first_audio_ms.append(float(i))
    assert len(stats.first_audio_ms) == min(count, RETAINED_LATENCIES)
