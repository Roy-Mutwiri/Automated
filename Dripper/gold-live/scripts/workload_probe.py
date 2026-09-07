"""A long mixed workload through one persistent loop, sampled as it runs.

    python -m scripts.workload_probe [--ops 1200] [--interrupts 400]

scripts/mem_probe.py answers "where does the memory go" for synthesis alone.
The accelerated soak asserts that structures are empty when it finishes. This
sits between them: many realistic mixed operations -- normal,
duplicate, empty, short, long, dropped, interrupted, failing, idle -- pushed
through a real AudioRouter and real Piper on a single long-lived event loop,
with RSS, handles, queue depth and preemption state recorded at intervals
rather than only at the end.

A structure that is empty at the end can still have been unbounded in the
middle, and an end-state assertion cannot see that. Sampling can.

Nothing is restarted mid-run and gc runs only at the marks: the question is
what this process does when it is left alone.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import gc
import shutil
import sys
import tempfile
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from platform_.audio.router import AudioRouter, AudioSink  # noqa: E402
from platform_.tts.base import TTSResult  # noqa: E402
from shared.contracts import AudioRequest, Priority  # noqa: E402


# -- process measurement ----------------------------------------------------


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_k32 = ctypes.windll.kernel32
_k32.GetCurrentProcess.restype = ctypes.c_void_p
_gpmi = getattr(_k32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
_gpmi.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wintypes.DWORD]
_gpmi.restype = wintypes.BOOL

# Without argtypes the pseudo-handle is marshalled as a C int and overflows.
_k32.GetProcessHandleCount.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
_k32.GetProcessHandleCount.restype = wintypes.BOOL


def rss_mb() -> float:
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    if not _gpmi(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError(f"GetProcessMemoryInfo failed: {ctypes.GetLastError()}")
    return c.WorkingSetSize / 1e6


def handle_count() -> int:
    n = wintypes.DWORD()
    if not _k32.GetProcessHandleCount(_k32.GetCurrentProcess(), ctypes.byref(n)):
        return -1
    return int(n.value)


# -- doubles that fail on purpose -------------------------------------------


class SilentSink(AudioSink):
    """Real router, no speaker.

    Playback latency is not what is being measured here, and 3,000 utterances
    through the speakers would make the run unusable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.played = 0

    async def play(self, path: Path) -> None:
        self.played += 1
        await asyncio.sleep(0)

    def stop(self) -> None:
        return None


class FlakyPiper:
    """Real Piper, made to fail every `fail_every` calls.

    Failure is a parameter, not an accident: the point is to drive the
    router's failure path thousands of times and see whether anything is
    retained when it does.

    It raises, because that is how PiperTTS actually fails. An earlier
    version returned TTSResult(path=None), which the router treats as "this
    segment was empty, say nothing" -- a legitimate path, not a failure. That
    double measured the wrong branch and made the failure counter look broken
    when it was not.
    """

    def __init__(self, inner, fail_every: int = 25) -> None:
        self.inner = inner
        self.fail_every = fail_every
        self.calls = 0
        self.raised = 0
        self.empty = 0

    async def synthesize(self, text: str, voice_id: str, out_path: Path) -> TTSResult:
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            self.raised += 1
            raise RuntimeError("synthetic Piper failure")
        # Half as often, the genuine empty-segment result.
        if self.fail_every and self.calls % (self.fail_every * 2) == 1:
            self.empty += 1
            return TTSResult(path=None, duration_ms=0, first_audio_ms=0)
        return await self.inner.synthesize(text, voice_id, out_path)


# -- sampling ---------------------------------------------------------------


@dataclass
class Sample:
    label: str
    rss: float
    handles: int
    queue: int
    preempted: bool
    stats_len: int
    wavs: int
    wav_bytes: int
    tmp_files: int

    def line(self) -> str:
        return (f"  {self.label:<30} rss {self.rss:>7.1f} MB  hnd {self.handles:>5}  "
                f"q {self.queue:>2}  preempt {'Y' if self.preempted else 'n'}  "
                f"lat {self.stats_len:>4}  wav {self.wavs:>4} "
                f"({self.wav_bytes / 1e6:>6.1f} MB)  tmp {self.tmp_files:>3}")


@dataclass
class Probe:
    router: AudioRouter
    out_dir: Path
    tmp_dir: Path
    samples: list[Sample] = field(default_factory=list)

    def take(self, label: str) -> Sample:
        s = Sample(
            label=label,
            rss=rss_mb(),
            handles=handle_count(),
            queue=self.router.queue_depth,
            preempted=self.router._preempted is not None,
            stats_len=len(self.router.stats.first_audio_ms),
            wavs=len(files := (list(self.out_dir.glob("*.wav"))
                               if self.out_dir.is_dir() else [])),
            wav_bytes=sum(f.stat().st_size for f in files if f.exists()),
            tmp_files=sum(1 for _ in self.tmp_dir.rglob("*")) if self.tmp_dir.is_dir() else 0,
        )
        self.samples.append(s)
        print(s.line(), flush=True)
        return s


async def settle(ticks: int = 60) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


async def drain(router: AudioRouter, timeout_s: float = 120.0) -> bool:
    """Wait until the router has actually finished speaking.

    settle() only yields the loop; synthesis needs real time. Sampling
    "after the workload" without waiting here would measure a queue still
    full of work and call it drained.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if router.queue_depth == 0 and not router.is_speaking:
            return True
        await asyncio.sleep(0.05)
    return False


async def room_for_more(router: AudioRouter, headroom: int = 6,
                        timeout_s: float = 60.0) -> None:
    """Backpressure, so the workload completes rather than being dropped."""
    deadline = time.monotonic() + timeout_s
    while router.queue_depth >= headroom and time.monotonic() < deadline:
        await asyncio.sleep(0.02)


# -- the workload -----------------------------------------------------------

LONG = ("Gold has spent the better part of this session grinding against the "
        "same shelf of supply that capped it yesterday, and every attempt to "
        "push through has been met by sellers who are in no hurry at all.")
SHORT = "Still holding."
REPEAT = "Gold is holding just under that level."


def make_request(kind: str, i: int) -> AudioRequest:
    segments = {
        "normal": [f"Gold is trading around the level we flagged, note {i}."],
        "repeat": [REPEAT],
        "empty": [""],
        "short": [SHORT],
        "long": [LONG],
        "multi": [f"First point {i}.", f"Second point {i}.", f"Third point {i}."],
    }[kind]
    return AudioRequest(
        session_id="SESSION_001",
        utterance_id=uuid4(),
        trace_id=f"trace-{i}",
        segments=segments,
        voice_id="en_US-john-medium",
        priority=Priority.LOW,
    )


async def mixed_workload(probe: Probe, ops: int, every: int) -> dict:
    """Section 6: thousands of realistic mixed operations."""
    kinds = ["normal", "repeat", "short", "long", "multi", "empty"]
    dropped = 0
    submitted = 0

    for i in range(1, ops + 1):
        # Every 200th op, deliberately burst past the queue so the drop path
        # is exercised too -- dropped responses are part of the workload.
        if i % 200 == 0:
            for b in range(12):
                if not await probe.router.submit(make_request("short", i * 100 + b)):
                    dropped += 1
                submitted += 1
            continue

        await room_for_more(probe.router)
        accepted = await probe.router.submit(make_request(kinds[i % len(kinds)], i))
        submitted += 1
        if not accepted:
            dropped += 1

        # An idle beat now and then: a real session is not saturated.
        if i % 37 == 0:
            await asyncio.sleep(0.05)

        if i % every == 0:
            await drain(probe.router)
            probe.take(f"{i} mixed ops")

    await drain(probe.router, timeout_s=180)
    return {"submitted": submitted, "dropped": dropped}


async def interruption_workload(probe: Probe, rounds: int, every: int) -> dict:
    """Section 5: barge-in on nearly every utterance, sampled throughout.

    The accelerated soak proves `_preempted` is None at the end. This proves
    it never accumulates *during*, which is the property the old set-based
    design actually violated.
    """
    max_preempt_state = 0
    for i in range(1, rounds + 1):
        await room_for_more(probe.router, headroom=3)
        await probe.router.submit(make_request("long", i))
        # Let it actually start speaking, otherwise there is nothing to preempt
        # and the test quietly measures nothing.
        await asyncio.sleep(0.12)
        await probe.router.submit(
            AudioRequest(
                session_id="SESSION_001", utterance_id=uuid4(),
                trace_id=f"barge-{i}", segments=[f"Breaking, {i}."],
                voice_id="en_US-john-medium", priority=Priority.CRITICAL,
            )
        )
        await asyncio.sleep(0.05)
        if probe.router._preempted is not None:
            max_preempt_state = max(max_preempt_state, 1)
        if i % every == 0:
            await drain(probe.router)
            probe.take(f"{i} interruptions")

    await drain(probe.router, timeout_s=180)
    return {"rounds": rounds, "max_preempt_state": max_preempt_state}


async def run(ops: int, interrupts: int) -> int:
    from platform_.tts.piper import PiperTTS
    from shared.paths import data_root

    work = Path(tempfile.mkdtemp(prefix="goldlive-workload-"))
    out_dir = work / "out"
    tmp_dir = work / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  workspace : {work}")
    print(f"  mixed ops : {ops}   interruptions: {interrupts}\n")

    gc.collect()
    print(f"  {'before piper':<30} rss {rss_mb():>7.1f} MB  hnd {handle_count():>5}")

    piper = PiperTTS(voices_dir=data_root() / "voices")
    await piper.synthesize("Warming up.", "en_US-john-medium", tmp_dir / "w.wav")
    gc.collect()
    print(f"  {'piper loaded + 1 call':<30} rss {rss_mb():>7.1f} MB  hnd {handle_count():>5}\n")

    tts = FlakyPiper(piper, fail_every=25)
    sink = SilentSink()
    router = AudioRouter("SESSION_001", tts, out_dir, sink=sink, max_queue=8)
    sweeps = {"n": 0, "deleted": 0, "errors": 0}
    real_sweep = router.sweep_audio

    def counted_sweep():
        result = real_sweep()
        sweeps["n"] += 1
        sweeps["deleted"] += result.deleted
        sweeps["errors"] += len(result.errors)
        return result

    router.sweep_audio = counted_sweep
    await router.start()
    probe = Probe(router=router, out_dir=out_dir / "SESSION_001", tmp_dir=tmp_dir)

    probe.take("router started")
    mixed = await mixed_workload(probe, ops, max(1, ops // 10))
    after_mixed = probe.take("mixed workload drained")
    interrupt = await interruption_workload(probe, interrupts, max(1, interrupts // 10))
    after_interrupt = probe.take("interruptions drained")

    await router.stop()
    gc.collect()
    final = probe.take("router stopped")

    print("\n" + "=" * 98)
    print(f"  submitted            : {mixed['submitted']}")
    print(f"  dropped (queue full) : {mixed['dropped']}")
    print(f"  tts calls: {tts.calls}  raised: {tts.raised}  empty-result: {tts.empty}")
    print(f"  sink playbacks       : {sink.played}")
    print(f"  router stats         : queued={router.stats.queued} played={router.stats.played} "
          f"preempted={router.stats.preempted} tts_failures={router.stats.tts_failures} "
          f"dropped_expired={router.stats.dropped_expired}")
    print(f"  interruptions        : {interrupt['rounds']}")
    print(f"  max preemption state : {interrupt['max_preempt_state']} (design maximum is 1)")
    print(f"  first_audio_ms list  : {len(router.stats.first_audio_ms)} entries retained")
    print(f"  final queue depth    : {router.queue_depth}")
    print(f"  final preempted      : {router._preempted}")
    files_now, bytes_now = router.retention.usage()
    peak_files = max((s.wavs for s in probe.samples), default=0)
    peak_bytes = max((s.wav_bytes for s in probe.samples), default=0)
    print(f"  retention policy     : {router.retention.policy.describe()}")
    print(f"  wav peak / final     : {peak_files} files "
          f"({peak_bytes / 1e6:.1f} MB) -> {files_now} files "
          f"({bytes_now / 1e6:.1f} MB)")
    print(f"  sweeps / deleted / errors : {sweeps['n']} / {sweeps['deleted']} / "
          f"{sweeps['errors']}")

    mid = [s for s in probe.samples if "mixed ops" in s.label]
    if len(mid) >= 2:
        print(f"  mixed-phase RSS      : {mid[0].rss:.1f} -> {mid[-1].rss:.1f} MB "
              f"({mid[-1].rss - mid[0].rss:+.1f})")
        print(f"  mixed-phase handles  : {mid[0].handles} -> {mid[-1].handles} "
              f"({mid[-1].handles - mid[0].handles:+d})")
    bar = [s for s in probe.samples if "interruptions" in s.label]
    if len(bar) >= 2:
        print(f"  barge-in RSS         : {bar[0].rss:.1f} -> {bar[-1].rss:.1f} MB "
              f"({bar[-1].rss - bar[0].rss:+.1f})")
        print(f"  barge-in handles     : {bar[0].handles} -> {bar[-1].handles} "
              f"({bar[-1].handles - bar[0].handles:+d})")
    print(f"  after mixed / after barge-in / stopped : {after_mixed.rss:.1f} / "
          f"{after_interrupt.rss:.1f} / {final.rss:.1f} MB")
    print("=" * 98 + "\n")

    shutil.rmtree(work, ignore_errors=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ops", type=int, default=1200)
    p.add_argument("--interrupts", type=int, default=400)
    a = p.parse_args()
    started = time.monotonic()
    rc = asyncio.run(run(a.ops, a.interrupts))
    print(f"  elapsed {time.monotonic() - started:.0f}s\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
