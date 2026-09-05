"""Regress the three Phase 1 defects against the frozen artifact.

    python -m scripts.frozen_checks

Source-mode passing proves nothing about the packaged build: every one of
these defects lived in code that behaves differently once frozen -- signal
delivery, parent process identity, and module resolution.

  A  STOP must not kill whoever called it.
  B  A hard-killed supervisor must not leave its session running.
  C  A data directory outside the checkout must work.

Each check leaves the machine clean whether it passes or fails.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXE = ROOT / "dist" / "GoldLive" / "GoldLive.exe"
HEALTH_PORT = 9101
POLL_S = 0.25

DEV_ARGS = ["--market", "synthetic", "--adapter", "mock", "--tts", "file",
            "--allow-degraded"]


def health(timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{HEALTH_PORT}/health", timeout=timeout
        ) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def wait_for(predicate, deadline_s: float, what: str) -> bool:
    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        if predicate():
            return True
        time.sleep(POLL_S)
    print(f"      timed out after {deadline_s:.0f}s waiting for {what}")
    return False


def goldlive_pids() -> set[int]:
    me = os.getpid()
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'GoldLive.exe' } | "
        "Select-Object ProcessId | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=25,
        )
        raw = (out.stdout or "").strip()
        if not raw:
            return set()
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        return {int(p["ProcessId"]) for p in data if int(p["ProcessId"]) != me}
    except Exception:
        return set()


def cleanup() -> None:
    for pid in goldlive_pids():
        subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                       capture_output=True, timeout=15)
    wait_for(lambda: not goldlive_pids(), 20, "cleanup")


def start_supervisor(data_dir: Path, log: Path) -> subprocess.Popen:
    env = {**os.environ, "GOLDLIVE_DATA_DIR": str(data_dir)}
    with open(log, "a", encoding="utf-8") as handle:
        return subprocess.Popen(
            [str(EXE), "supervise", "--sessions", "SESSION_001", *DEV_ARGS],
            cwd=str(data_dir), env=env, stdout=handle, stderr=handle,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )


# -- A ----------------------------------------------------------------------


def check_a(data_dir: Path, log: Path) -> bool:
    """STOP must terminate GoldLive and leave the caller alive.

    The defect: CTRL_BREAK_EVENT is routed by console, so it reached the
    process that asked for the stop. `GoldLive.exe stop` from a terminal
    killed that terminal.
    """
    print("\n  A. STOP must not kill the caller")
    cleanup()
    proc = start_supervisor(data_dir, log)
    if not wait_for(lambda: health() is not None, 90, "the session to come up"):
        cleanup()
        return False
    print(f"      session up, {len(goldlive_pids())} GoldLive process(es)")

    # A sacrificial child issues the stop. If the signal escapes to the
    # caller, this process dies and its exit code says so.
    caller = subprocess.run(
        [str(EXE), "stop"],
        cwd=str(data_dir), env={**os.environ, "GOLDLIVE_DATA_DIR": str(data_dir)},
        capture_output=True, text=True, timeout=120,
    )
    print(f"      `GoldLive.exe stop` returned {caller.returncode}")

    alive = proc.poll() is None
    stopped = wait_for(lambda: health() is None, 45, "the health endpoint to close")
    gone = wait_for(lambda: not goldlive_pids(), 45, "processes to exit")

    # This interpreter still running is itself part of the evidence.
    print(f"      this harness is still alive: pid {os.getpid()}")
    if alive:
        print("      supervisor still running after stop")
    cleanup()

    ok = caller.returncode == 0 and stopped and gone
    print(f"      -> {'PASS' if ok else 'FAIL'}")
    return ok


# -- B ----------------------------------------------------------------------


def check_b(data_dir: Path, log: Path) -> bool:
    """A hard-killed supervisor must not leave its session behind."""
    print("\n  B. Hard-killed supervisor must not orphan its session")
    cleanup()
    proc = start_supervisor(data_dir, log)
    if not wait_for(lambda: health() is not None, 90, "the session to come up"):
        cleanup()
        return False

    before = goldlive_pids()
    print(f"      {len(before)} GoldLive process(es) before the kill")

    # /F, no /T: kill only the supervisor, exactly as a crash would.
    subprocess.run(["taskkill", "/F", "/PID", str(proc.pid)],
                   capture_output=True, timeout=15)
    print(f"      hard-killed supervisor pid {proc.pid}")

    # The watchdog polls every 5s by default; allow a few cycles.
    session_gone = wait_for(lambda: not goldlive_pids(), 60,
                            "the orphaned session to stop itself")
    endpoint_gone = wait_for(lambda: health() is None, 30,
                             "the health endpoint to close")

    from runtime import lifecycle

    os.environ["GOLDLIVE_DATA_DIR"] = str(data_dir)
    state = lifecycle.load(path=data_dir / "lifecycle.json")
    print(f"      lifecycle after the crash: {state.state.value}")
    crash_recorded = not state.may_run

    # Recovery is only proved by the next operation succeeding.
    cleanup()
    again = start_supervisor(data_dir, log)
    restarted = wait_for(lambda: health() is not None, 90, "the next START")
    print(f"      next START {'succeeded' if restarted else 'FAILED'}")
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(again.pid)],
                   capture_output=True, timeout=15)
    cleanup()

    ok = session_gone and endpoint_gone and crash_recorded and restarted
    print(f"      -> {'PASS' if ok else 'FAIL'}")
    return ok


# -- C ----------------------------------------------------------------------


def check_c(log: Path) -> bool:
    """A data directory outside the checkout must work, from any cwd."""
    print("\n  C. Non-standard GOLDLIVE_DATA_DIR, launched from elsewhere")
    cleanup()
    temp = Path(tempfile.mkdtemp(prefix="goldlive-isolated-"))
    try:
        # Seed the isolated directory the way Setup would.
        subprocess.run(
            [str(EXE), "setup"],
            cwd=str(Path(tempfile.gettempdir())),
            env={**os.environ, "GOLDLIVE_DATA_DIR": str(temp)},
            capture_output=True, text=True, timeout=120,
        )
        # Voices live in the real data dir; link what the session needs.
        real_voices = Path(os.environ.get("LOCALAPPDATA", "")) / "GoldLive" / "voices"
        if real_voices.is_dir():
            shutil.copytree(real_voices, temp / "voices", dirs_exist_ok=True)

        env = {**os.environ, "GOLDLIVE_DATA_DIR": str(temp)}
        with open(log, "a", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                [str(EXE), "supervise", "--sessions", "SESSION_001", *DEV_ARGS],
                cwd=str(Path(tempfile.gettempdir())),  # deliberately not the checkout
                env=env, stdout=handle, stderr=handle,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        up = wait_for(lambda: health() is not None, 90, "the isolated session")
        print(f"      started from {Path(tempfile.gettempdir())} with data in {temp.name}")

        text = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        module_error = "ModuleNotFoundError" in text
        if module_error:
            print("      ModuleNotFoundError appeared in the log")

        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, timeout=15)
        stopped = wait_for(lambda: not goldlive_pids(), 45, "shutdown")
        cleanup()

        ok = up and stopped and not module_error
        print(f"      -> {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def main() -> int:
    if not EXE.exists():
        sys.exit(f"no frozen build at {EXE}; run PyInstaller first")

    from shared.paths import data_path

    log = data_path("logs", "frozen_checks.log")
    log.write_text("", encoding="utf-8")
    data_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "GoldLive"

    print(f"\n  frozen artifact: {EXE}")
    print(f"  log: {log}")

    results = {
        "A caller survives STOP": check_a(data_dir, log),
        "B hard-kill self-cleanup": check_b(data_dir, log),
        "C non-standard data dir": check_c(log),
    }

    print("\n" + "=" * 58)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 58 + "\n")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
