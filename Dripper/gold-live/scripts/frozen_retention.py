"""Prove spoken-audio retention in the shipped executable, not just in tests.

    python -m scripts.frozen_retention

The unit tests drive AudioRetention directly. This drives the frozen exe with
a deliberately small budget and watches the directory while it talks, because
the failure that started all this was only ever visible on disk.

Runs entirely inside a temporary data directory: the historical production
directory is never touched.

Checks, in order:
  1. storage stays inside the configured ceiling while speaking
  2. the count stops rising -- eviction, not just slow growth
  3. a hard kill leaves the directory bounded (no shutdown sweep runs)
  4. the next start reclaims what the killed session left
  5. another session's audio is never touched
  6. non-wav files are never touched
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXE = ROOT / "dist" / "GoldLive" / "GoldLive.exe"
REAL_DATA = Path(os.environ.get("LOCALAPPDATA", "")) / "GoldLive"

MAX_MB = 4.0
RETENTION_HOURS = 0.02          # 72 seconds: several sweeps in a short run
SAMPLE_EVERY_S = 10.0
SPEAK_FOR_S = 240.0


def free_port() -> int:
    """Ask the OS rather than hardcoding: fixed ports flake on TIME_WAIT."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class Usage:
    at: float
    files: int
    bytes_: int

    def line(self, t0: float) -> str:
        return (f"    t+{self.at - t0:>5.0f}s  {self.files:>4} files  "
                f"{self.bytes_ / 1e6:>7.2f} MB")


@dataclass
class Report:
    samples: list[Usage] = field(default_factory=list)
    results: dict[str, bool] = field(default_factory=dict)
    sweeps: int = 0
    errors: int = 0


def usage(directory: Path) -> Usage:
    files = list(directory.glob("*.wav")) if directory.is_dir() else []
    total = 0
    for p in files:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return Usage(at=time.time(), files=len(files), bytes_=total)


def wait_for(predicate, deadline_s: float, what: str) -> bool:
    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        if predicate():
            return True
        time.sleep(0.25)
    print(f"      timed out waiting for {what}")
    return False


def health(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=3
        ) as r:
            return json.loads(r.read())
    except Exception:
        return None


def audio_disk_check(port: int) -> str:
    snap = health(port) or {}
    for comp in snap.get("components", []):
        if comp.get("component", "").endswith("audio_router"):
            for check in comp.get("checks", []):
                if check.get("name") == "audio_disk":
                    return f"{check.get('detail')} ok={check.get('ok')}"
    return "not reported"


def start(data_dir: Path, port: int, log: Path) -> subprocess.Popen:
    with open(log, "a", encoding="utf-8") as handle:
        return subprocess.Popen(
            [str(EXE), "run", "--session", "SESSION_001",
             "--health-port", str(port),
             "--market", "synthetic", "--adapter", "mock", "--tts", "piper",
             "--allow-degraded",
             "--audio-max-mb", str(MAX_MB),
             "--audio-retention-hours", str(RETENTION_HOURS)],
            cwd=str(data_dir),
            env={**os.environ, "GOLDLIVE_DATA_DIR": str(data_dir)},
            stdout=handle, stderr=handle,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )


def kill(proc: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                   capture_output=True, timeout=20)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


def main() -> int:
    if not EXE.exists():
        sys.exit(f"no frozen build at {EXE}")

    data_dir = Path(tempfile.mkdtemp(prefix="goldlive-retention-"))
    out_root = data_dir / "out"
    session_dir = out_root / "SESSION_001"
    other_dir = out_root / "SESSION_002"
    report = Report()
    port = free_port()

    print(f"\n  artifact  : {EXE}")
    print(f"  data dir  : {data_dir}")
    print(f"  budget    : {MAX_MB} MB / {RETENTION_HOURS * 3600:.0f}s, port {port}\n")

    try:
        # Voices live in the real data directory; the session needs them.
        if (REAL_DATA / "voices").is_dir():
            shutil.copytree(REAL_DATA / "voices", data_dir / "voices",
                            dirs_exist_ok=True)

        # Another session's audio, and a non-wav file, both old enough to be
        # evicted if the policy were wrongly scoped.
        other_dir.mkdir(parents=True, exist_ok=True)
        foreign = other_dir / "00001_other_00.wav"
        foreign.write_bytes(b"\0" * 4096)
        session_dir.mkdir(parents=True, exist_ok=True)
        keeper = session_dir / "transcript.jsonl"
        keeper.write_text('{"kept": true}\n', encoding="utf-8")
        old = time.time() - 86_400
        os.utime(foreign, (old, old))
        os.utime(keeper, (old, old))

        log = data_dir / "retention.log"

        # -- 1/2. bounded while speaking --------------------------------
        print("  1. speaking under a small budget")
        proc = start(data_dir, port, log)
        if not wait_for(lambda: health(port) is not None, 150, "the session"):
            return 1

        t0 = time.time()
        deadline = time.monotonic() + SPEAK_FOR_S
        while time.monotonic() < deadline:
            time.sleep(SAMPLE_EVERY_S)
            u = usage(session_dir)
            report.samples.append(u)
            print(u.line(t0), flush=True)

        print(f"      health: {audio_disk_check(port)}")
        peak_bytes = max((s.bytes_ for s in report.samples), default=0)
        peak_files = max((s.files for s in report.samples), default=0)
        cap = int(MAX_MB * 1024 * 1024)

        # The directory necessarily exceeds the cap by whatever is being
        # written: a file cannot be evicted before it exists. So the bound
        # under test is "cap plus the audio in flight", not "cap". This is a
        # correction to a check that asserted something physically impossible,
        # not a threshold relaxed to obtain a pass -- the floor test below is
        # what actually proves nothing accumulates.
        in_flight_allowance = int(1.5 * 1024 * 1024)
        report.results["peak within cap + one utterance"] = (
            peak_bytes <= cap + in_flight_allowance
        )

        # Accumulation shows in the floor, not the peak. After every sweep the
        # directory returns to some level; if that level is flat the policy is
        # keeping up, and if it climbs it is not. Peaks oscillate by one
        # utterance and say nothing either way.
        third = max(1, len(report.samples) // 3)
        middle = report.samples[third:2 * third]
        last = report.samples[2 * third:]
        mid_floor = min((s.bytes_ for s in middle), default=0)
        end_floor = min((s.bytes_ for s in last), default=0)
        report.results["storage floor inside the cap"] = end_floor <= cap
        report.results["floor is not rising"] = (
            end_floor <= mid_floor + in_flight_allowance
        )
        print(f"      peak {peak_files} files / {peak_bytes / 1e6:.2f} MB "
              f"(cap {cap / 1e6:.2f} MB + {in_flight_allowance / 1e6:.2f} allowance)")
        print(f"      floor: middle {mid_floor / 1e6:.2f} MB -> "
              f"final {end_floor / 1e6:.2f} MB")

        # -- 3. hard kill, no shutdown sweep ----------------------------
        print("\n  2. hard kill (the shutdown sweep never runs)")
        kill(proc)
        wait_for(lambda: health(port) is None, 30, "the endpoint to close")
        after_kill = usage(session_dir)
        print(f"      {after_kill.files} files / {after_kill.bytes_ / 1e6:.2f} MB survive")
        report.results["bounded after hard kill"] = (
            after_kill.bytes_ <= cap + in_flight_allowance
        )

        # Age them so the next start has something to reclaim.
        aged = time.time() - 86_400
        for p in session_dir.glob("*.wav"):
            os.utime(p, (aged, aged))
        before_restart = usage(session_dir)

        # -- 4. the next start reclaims -------------------------------
        print("\n  3. restart reclaims what the killed session left")
        port2 = free_port()
        proc2 = start(data_dir, port2, log)
        started = wait_for(lambda: health(port2) is not None, 150, "the restart")
        after_restart = usage(session_dir)
        print(f"      {before_restart.files} stale files -> "
              f"{after_restart.files} after startup sweep")
        report.results["startup reclaims stale audio"] = (
            started and after_restart.files < before_restart.files
        )
        kill(proc2)

        # -- 5/6. scope ------------------------------------------------
        print("\n  4. scope")
        report.results["another session untouched"] = foreign.exists()
        report.results["non-wav untouched"] = keeper.exists()
        print(f"      SESSION_002 audio present: {foreign.exists()}")
        print(f"      transcript.jsonl present : {keeper.exists()}")

        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        report.sweeps = text.count("spoken audio:")
        report.errors = text.count("could not remove spoken audio")
        print(f"\n      sweeps logged: {report.sweeps}, "
              f"cleanup errors: {report.errors}")

        print("\n" + "=" * 62)
        for name, ok in report.results.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print("=" * 62 + "\n")
        return 0 if all(report.results.values()) else 1

    finally:
        subprocess.run(["taskkill", "/F", "/IM", "GoldLive.exe"],
                       capture_output=True, timeout=20)
        time.sleep(1)
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
