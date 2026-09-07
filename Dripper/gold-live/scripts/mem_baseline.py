"""Resource baseline for the frozen artifact, measured from outside it.

    python -m scripts.mem_baseline [--runs 3] [--minutes 6]

The in-process probe (scripts/mem_probe.py) breaks down where memory goes
inside one interpreter. This measures what the shipped executable actually
does: RSS, handles and threads across every GoldLive process, sampled while
the session runs a real workload, then again after shutdown.

Deliberately does not force garbage collection or restart anything to flatter
the numbers. Windows keeps memory in the allocator after it is freed, so the
question is never "did RSS return to zero" -- it is "does it keep climbing
under equivalent work, or does it settle".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXE = ROOT / "dist" / "GoldLive" / "GoldLive.exe"
DATA = Path(os.environ.get("LOCALAPPDATA", "")) / "GoldLive"
PORT = 9101


# -- observation ------------------------------------------------------------


@dataclass
class Sample:
    at: float
    rss_mb: float = 0.0
    handles: int = 0
    threads: int = 0
    processes: int = 0
    queue_depth: float = 0.0
    utterances: float = 0.0
    blocked: float = 0.0
    state: str = "-"

    def line(self, label: str) -> str:
        return (f"  {label:<26} rss {self.rss_mb:>7.1f} MB  hnd {self.handles:>5}  "
                f"thr {self.threads:>4}  proc {self.processes}  q {self.queue_depth:.0f}  "
                f"utt {self.utterances:.0f}  {self.state}")


def processes() -> dict:
    ps = (
        "$p = Get-Process GoldLive -ErrorAction SilentlyContinue; "
        "if ($p) { [pscustomobject]@{ "
        "count=@($p).Count; "
        "rss=[math]::Round((($p | Measure-Object WorkingSet64 -Sum).Sum/1MB),1); "
        "handles=(($p | Measure-Object HandleCount -Sum).Sum); "
        "threads=(($p | ForEach-Object { $_.Threads.Count } | Measure-Object -Sum).Sum) "
        "} | ConvertTo-Json -Compress } else { '{}' }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=25,
        )
        return json.loads((out.stdout or "{}").strip() or "{}")
    except Exception:
        return {}


def metrics() -> dict:
    wanted = ("goldlive_queue_depth", "goldlive_utterances_total",
              "goldlive_blocked_total")
    out: dict = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=4) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                name, _, value = line.rpartition(" ")
                base = name.split("{")[0]
                if base in wanted:
                    with_suppress(out, base, value)
    except Exception:
        pass
    return out


def with_suppress(out: dict, key: str, value: str) -> None:
    try:
        out[key] = out.get(key, 0.0) + float(value)
    except ValueError:
        pass


def health_state() -> str:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
            return str(json.loads(r.read()).get("state", "-"))
    except Exception:
        return "down"


def sample() -> Sample:
    p = processes()
    m = metrics()
    return Sample(
        at=time.time(),
        rss_mb=float(p.get("rss") or 0), handles=int(p.get("handles") or 0),
        threads=int(p.get("threads") or 0), processes=int(p.get("count") or 0),
        queue_depth=m.get("goldlive_queue_depth", 0.0),
        utterances=m.get("goldlive_utterances_total", 0.0),
        blocked=m.get("goldlive_blocked_total", 0.0),
        state=health_state(),
    )


def wait_for(predicate, deadline_s: float, what: str) -> bool:
    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        if predicate():
            return True
        time.sleep(0.25)
    print(f"      timed out waiting for {what}")
    return False


def cleanup() -> None:
    for _ in range(2):
        subprocess.run(["taskkill", "/F", "/IM", "GoldLive.exe"],
                       capture_output=True, timeout=20)
        time.sleep(1)
    wait_for(lambda: (processes().get("count") or 0) == 0, 20, "cleanup")


def start() -> subprocess.Popen:
    log = DATA / "logs" / "mem_baseline.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle:
        return subprocess.Popen(
            [str(EXE), "supervise", "--sessions", "SESSION_001",
             "--market", "synthetic", "--adapter", "mock", "--tts", "piper",
             "--allow-degraded"],
            cwd=str(DATA), env={**os.environ, "GOLDLIVE_DATA_DIR": str(DATA)},
            stdout=handle, stderr=handle,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )


def skip() -> None:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/control/skip", data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass


# -- one measured run -------------------------------------------------------


@dataclass
class RunResult:
    samples: list[Sample] = field(default_factory=list)
    post_shutdown: Sample | None = None
    interruptions: int = 0

    @property
    def peak_rss(self) -> float:
        return max((s.rss_mb for s in self.samples), default=0.0)

    @property
    def final_rss(self) -> float:
        return self.samples[-1].rss_mb if self.samples else 0.0

    @property
    def warm_rss(self) -> float:
        """RSS once initialisation has settled: the third sample."""
        return self.samples[2].rss_mb if len(self.samples) > 2 else self.final_rss


def measure_run(index: int, minutes: float, interrupt_every: int = 4) -> RunResult:
    print(f"\n  run {index}: {minutes:.0f} minutes under load")
    cleanup()
    result = RunResult()

    before = sample()
    print(before.line("pre-launch"))

    proc = start()
    if not wait_for(lambda: health_state() != "down", 120, "the session"):
        cleanup()
        return result

    result.samples.append(sample())
    print(result.samples[-1].line("session up"))

    deadline = time.monotonic() + minutes * 60
    tick = 0
    while time.monotonic() < deadline:
        time.sleep(20)
        tick += 1
        # Interrupt periodically: barge-in is the path that leaked before.
        if tick % interrupt_every == 0:
            skip()
            result.interruptions += 1
        s = sample()
        result.samples.append(s)
        print(s.line(f"t+{tick * 20 // 60}m{tick * 20 % 60:02d}s"))

    from runtime.panel import stop_goldlive

    os.environ["GOLDLIVE_DATA_DIR"] = str(DATA)
    stop_goldlive(timeout_s=60)
    wait_for(lambda: (processes().get("count") or 0) == 0, 60, "shutdown")
    if proc.poll() is None:
        proc.kill()
    cleanup()

    result.post_shutdown = sample()
    print(result.post_shutdown.line("after shutdown"))
    return result


def main() -> int:
    if not EXE.exists():
        sys.exit(f"no frozen build at {EXE}")

    runs = 3
    minutes = 6.0
    argv = sys.argv[1:]
    if "--runs" in argv:
        runs = int(argv[argv.index("--runs") + 1])
    if "--minutes" in argv:
        minutes = float(argv[argv.index("--minutes") + 1])

    print(f"\n  artifact : {EXE}")
    print(f"  runs     : {runs} x {minutes:.0f} minutes\n")

    results = [measure_run(i + 1, minutes) for i in range(runs)]

    print("\n" + "=" * 78)
    print(f"  {'run':<6}{'warm MB':>10}{'peak MB':>10}{'final MB':>11}"
          f"{'after stop':>12}{'handles':>10}{'proc':>7}{'skips':>7}")
    for i, r in enumerate(results, 1):
        last = r.samples[-1] if r.samples else Sample(at=0)
        post = r.post_shutdown or Sample(at=0)
        print(f"  {i:<6}{r.warm_rss:>10.1f}{r.peak_rss:>10.1f}{r.final_rss:>11.1f}"
              f"{post.rss_mb:>12.1f}{last.handles:>10}{last.processes:>7}"
              f"{r.interruptions:>7}")

    peaks = [r.peak_rss for r in results if r.peak_rss]
    finals = [r.final_rss for r in results if r.final_rss]
    if peaks:
        spread = max(peaks) - min(peaks)
        print(f"\n  peak spread across runs : {spread:.1f} MB "
              f"({min(peaks):.1f} - {max(peaks):.1f})")
    if finals and len(finals) > 1:
        drift = finals[-1] - finals[0]
        print(f"  final RSS drift run1->runN : {drift:+.1f} MB")

    # Within-run growth after warm-up is the number that matters most.
    for i, r in enumerate(results, 1):
        if len(r.samples) > 3:
            after_warm = r.samples[2].rss_mb
            growth = r.final_rss - after_warm
            per_min = growth / max(1.0, minutes - 1)
            print(f"  run {i} growth after warm-up : {growth:+.1f} MB "
                  f"({per_min:+.1f} MB/min)")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
