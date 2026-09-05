"""Start and stop the real process topology, many times, and check it is clean.

    python -m scripts.process_cycles [cycles] [--frozen]

The accelerated soak exercises objects inside one interpreter. This exercises
what actually ships: a supervisor process that spawns a session process, a
health endpoint that has to appear and then disappear, and a lifecycle file
that has to be right afterwards.

Deterministic by construction -- every wait is a poll for a condition with a
deadline, never a fixed sleep, so it neither races on a slow machine nor
wastes time on a fast one.

Runs in development mode (synthetic market, mock adapter, file TTS) on
purpose: this measures process lifecycle, not content. The real market and
model are covered by the readiness gates and the wall-clock soak.
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

from runtime import lifecycle  # noqa: E402
from runtime.panel import stop_goldlive  # noqa: E402
from runtime.supervisor import child_env  # noqa: E402
from shared.paths import data_path, data_root  # noqa: E402

HEALTH_PORT = 9101
START_DEADLINE_S = 90.0
STOP_DEADLINE_S = 45.0
POLL_S = 0.25


# -- observation ------------------------------------------------------------


def goldlive_processes() -> list[dict]:
    """Every GoldLive process, with its parent, so duplicates can be collapsed.

    Matching loosely on the command line was wrong: the first version matched
    any process merely *mentioning* the pattern, including the shell running
    the query. Wildcards on the actual `-m runtime.supervisor` / `-m
    runtime.live` invocation are exact and need no regex escaping, which
    produced a literal backspace when attempted.
    """
    me = os.getpid()
    ps = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "($_.Name -eq 'python.exe' -and ("
        "$_.CommandLine -like '* -m runtime.supervisor *' -or "
        "$_.CommandLine -like '* -m runtime.live *')) "
        "-or $_.Name -eq 'GoldLive.exe' } | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=25,
        )
        raw = (out.stdout or "").strip()
        if not raw:
            return []
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        return [p for p in data if int(p["ProcessId"]) != me]
    except Exception:
        return []


def goldlive_pids() -> set[int]:
    return {int(p["ProcessId"]) for p in goldlive_processes()}


def logical_process_count() -> int:
    """Processes that are really separate roles, not launcher re-execs.

    A venv's python.exe is a small launcher that re-executes the real
    interpreter as a child with an identical command line, so every logical
    process appears twice in source mode. The frozen build has no launcher.
    Collapsing a child onto a parent with the same command line counts the
    topology that actually exists: one supervisor, one session.
    """
    procs = goldlive_processes()
    by_pid = {int(p["ProcessId"]): p for p in procs}
    launchers = {
        int(p["ProcessId"])
        for p in procs
        if (parent := by_pid.get(int(p["ParentProcessId"])))
        and parent.get("CommandLine") == p.get("CommandLine")
    }
    return len(procs) - len(launchers)


def health() -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{HEALTH_PORT}/health", timeout=2
        ) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def wait_for(predicate, deadline_s: float, what: str) -> float:
    """Poll until true. Returns elapsed seconds, raises on timeout."""
    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        if predicate():
            return time.monotonic() - started
        time.sleep(POLL_S)
    raise TimeoutError(f"timed out after {deadline_s:.0f}s waiting for {what}")


# -- one cycle --------------------------------------------------------------


@dataclass
class CycleResult:
    index: int
    start_s: float = 0.0
    stop_s: float = 0.0
    peak_processes: int = 0
    leftover: set[int] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    ok: bool = True

    def fail(self, note: str) -> None:
        self.ok = False
        self.notes.append(note)


def supervisor_command(frozen: bool) -> list[str]:
    if frozen:
        exe = ROOT / "dist" / "GoldLive" / "GoldLive.exe"
        base = [str(exe), "supervise"]
    else:
        base = [sys.executable, "-m", "runtime.supervisor"]
    return [
        *base, "--sessions", "SESSION_001",
        "--market", "synthetic", "--adapter", "mock", "--tts", "file",
        "--allow-degraded",
    ]


def run_cycle(index: int, frozen: bool, log: Path,
              stop_during_warmup: bool = False) -> CycleResult:
    result = CycleResult(index=index)

    before = goldlive_pids()
    if before:
        result.fail(f"processes already running before the cycle: {sorted(before)}")

    with open(log, "a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            supervisor_command(frozen), cwd=str(data_root()),
            stdout=handle, stderr=handle,
            env={**child_env(), "GOLDLIVE_DATA_DIR": str(data_root())},
            # As the panel does: its own group, so a CTRL_BREAK aimed at the
            # supervisor cannot travel back to whatever started it.
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    try:
        if stop_during_warmup:
            # Do not wait for health: STOP arriving mid-startup is the case
            # most likely to leave a half-built session behind.
            time.sleep(1.0)
            result.notes.append("stopped during warm-up")
        else:
            result.start_s = wait_for(
                lambda: health() is not None, START_DEADLINE_S, "the health endpoint"
            )
            snapshot = health() or {}
            if snapshot.get("state") not in ("ok", "degraded"):
                result.fail(f"unexpected health state {snapshot.get('state')!r}")

            live = lifecycle.load(reconcile=False)
            if live.state is not lifecycle.RunState.RUNNING:
                result.fail(f"lifecycle says {live.state.value}, expected running")

        result.peak_processes = logical_process_count()
        if not stop_during_warmup and result.peak_processes != 2:
            result.fail(
                f"expected supervisor + session = 2 processes, saw "
                f"{result.peak_processes}"
            )

        # Stop exactly the way the product does. Popen.terminate() maps to
        # TerminateProcess on Windows, which gives the supervisor no chance to
        # stop anything -- using it here would test a path no user can invoke
        # and would orphan every session.
        started = time.monotonic()
        stopped, why = stop_goldlive(timeout_s=STOP_DEADLINE_S)
        if not stopped:
            result.fail(f"production stop reported failure: {why}")
        result.stop_s = time.monotonic() - started

    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    try:
        wait_for(lambda: health() is None, STOP_DEADLINE_S, "the health endpoint to close")
    except TimeoutError:
        result.fail("the health endpoint was still answering after shutdown")

    try:
        wait_for(lambda: not goldlive_pids(), STOP_DEADLINE_S, "child processes to exit")
    except TimeoutError:
        result.leftover = goldlive_pids()
        result.fail(f"orphaned processes remained: {sorted(result.leftover)}")

    final = lifecycle.load()
    if final.may_run:
        result.fail(f"lifecycle still authorises running: {final.state.value}")

    return result


# -- the run ----------------------------------------------------------------


def main() -> int:
    cycles = 10
    frozen = "--frozen" in sys.argv
    for arg in sys.argv[1:]:
        if arg.isdigit():
            cycles = int(arg)

    log = data_path("logs", "process_cycles.log")
    mode = "frozen" if frozen else "source"
    print(f"\n  {cycles} start/stop cycles ({mode} mode)")
    print(f"  log: {log}\n")

    # Leave nothing from a previous run to confuse the first cycle.
    for pid in goldlive_pids():
        subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                       capture_output=True, timeout=15)
    lifecycle.mark_stopped()

    results: list[CycleResult] = []
    for i in range(1, cycles + 1):
        # Every fourth cycle stops during warm-up instead of when healthy.
        warmup = (i % 4 == 0)
        result = run_cycle(i, frozen, log, stop_during_warmup=warmup)
        results.append(result)
        flag = "ok " if result.ok else "FAIL"
        extra = f"  {'; '.join(result.notes)}" if result.notes else ""
        print(f"  [{i:>3}] {flag}  start {result.start_s:5.1f}s  "
              f"stop {result.stop_s:5.1f}s  procs {result.peak_processes}{extra}",
              flush=True)

    peaks = [r.peak_processes for r in results if r.peak_processes]
    failed = [r for r in results if not r.ok]

    print("\n" + "=" * 62)
    print(f"  cycles              : {len(results)}")
    print(f"  failed              : {len(failed)}")
    print(f"  max process count   : {max(peaks) if peaks else 0}")
    print(f"  first / last peak   : {peaks[0] if peaks else 0} / "
          f"{peaks[-1] if peaks else 0}")
    starts = [r.start_s for r in results if r.start_s]
    if starts:
        print(f"  start time min/max  : {min(starts):.1f}s / {max(starts):.1f}s")
    print(f"  leftover processes  : {sorted(set().union(*(r.leftover for r in results)))}")
    print("=" * 62 + "\n")

    for r in failed:
        print(f"  cycle {r.index}: {'; '.join(r.notes)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
