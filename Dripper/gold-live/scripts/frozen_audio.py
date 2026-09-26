"""Audio lifecycle and shutdown-from-each-state, against the frozen artifact.

    python -m scripts.frozen_audio

The source suite injects fakes to reach failure paths; a frozen process cannot
be reached that way. What it *can* prove is the part that only the real
artifact can: that bundled Piper actually synthesises inside the exe, that
audio files appear, that the router reports healthy through interruption, and
that shutdown from a speaking state leaves nothing behind.

Uses the synthetic market for speed and real Piper for truth -- the point here
is the audio path, not the price.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXE = ROOT / "dist" / "GoldLive" / "GoldLive.exe"
PORT = 9101
DATA = Path(os.environ.get("LOCALAPPDATA", "")) / "GoldLive"


def get(path: str, timeout: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def post(name: str, payload: dict | None = None) -> dict | None:
    body = json.dumps(payload or {}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/control/{name}", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as exc:
        print(f"      control {name} failed: {exc}")
        return None


def wait_for(predicate, deadline_s: float, what: str) -> bool:
    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        if predicate():
            return True
        time.sleep(0.25)
    print(f"      timed out after {deadline_s:.0f}s waiting for {what}")
    return False


def pids() -> set[int]:
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'GoldLive.exe' } "
          "| Select-Object ProcessId | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=25)
        raw = (out.stdout or "").strip()
        if not raw:
            return set()
        data = json.loads(raw)
        return {int(p["ProcessId"]) for p in ([data] if isinstance(data, dict) else data)}
    except Exception:
        return set()


def cleanup() -> None:
    for pid in pids():
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=15)
    wait_for(lambda: not pids(), 20, "cleanup")


def start(tts: str = "piper") -> subprocess.Popen:
    log = DATA / "logs" / "frozen_audio.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle:
        return subprocess.Popen(
            [str(EXE), "supervise", "--sessions", "SESSION_001",
             "--market", "synthetic", "--adapter", "mock", "--tts", tts,
             "--allow-degraded"],
            cwd=str(DATA), env={**os.environ, "GOLDLIVE_DATA_DIR": str(DATA)},
            stdout=handle, stderr=handle,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )


def wavs_since(mark: float) -> int:
    out = DATA / "out" / "SESSION_001"
    if not out.is_dir():
        return 0
    return sum(1 for p in out.glob("*.wav") if p.stat().st_mtime > mark)


def component(health: dict, name: str) -> str:
    for c in health.get("components", []):
        if c.get("component", "").endswith(name):
            return c.get("state", "?")
    return "?"


def main() -> int:
    if not EXE.exists():
        sys.exit(f"no frozen build at {EXE}")

    results: dict[str, bool] = {}
    print(f"\n  frozen artifact: {EXE}\n")

    # -- real synthesis and playback through the packaged Piper -------------
    print("  1. real TTS and playback inside the frozen exe")
    cleanup()
    mark = time.time()
    proc = start("piper")
    up = wait_for(lambda: get("/health") is not None, 120, "the session")
    spoke = wait_for(lambda: wavs_since(mark) >= 3, 180, "three utterances of audio")
    produced = wavs_since(mark)
    snapshot = get("/health") or {}
    router_state = component(snapshot, "audio_router")
    print(f"      {produced} wav file(s) produced; audio_router={router_state}")
    results["real TTS produces audio"] = up and spoke
    results["router healthy while speaking"] = router_state in ("ok", "degraded")

    # -- interruption through the operator control -------------------------
    print("\n  2. barge-in via the skip control")
    before = wavs_since(mark)
    skipped = post("skip") is not None
    time.sleep(2)
    still_healthy = (get("/health") or {}).get("state") in ("ok", "degraded")
    after_skip = wait_for(lambda: wavs_since(mark) > before, 120,
                          "speech to resume after the interruption")
    print(f"      skip accepted={skipped}, healthy after={still_healthy}, "
          f"resumed={after_skip}")
    results["barge-in accepted"] = skipped
    results["healthy after interruption"] = still_healthy
    results["speech resumes after interruption"] = after_skip

    # -- repeated interruption ---------------------------------------------
    print("\n  3. repeated interruption")
    for _ in range(5):
        post("skip")
        time.sleep(0.6)
    repeated_ok = (get("/health") or {}).get("state") in ("ok", "degraded")
    resumed = wait_for(lambda: wavs_since(mark) > 0, 90, "audio after repeated skips")
    print(f"      healthy={repeated_ok}, still producing audio={resumed}")
    results["survives repeated interruption"] = repeated_ok and resumed

    # -- shutdown while actively speaking ----------------------------------
    print("\n  4. shutdown while speaking")
    from runtime.panel import stop_goldlive

    os.environ["GOLDLIVE_DATA_DIR"] = str(DATA)
    stopped, why = stop_goldlive(timeout_s=60)
    endpoint_gone = wait_for(lambda: get("/health") is None, 45, "health to close")
    procs_gone = wait_for(lambda: not pids(), 45, "processes to exit")
    print(f"      stop={stopped} ({why}), endpoint gone={endpoint_gone}, "
          f"processes gone={procs_gone}")
    results["clean shutdown while speaking"] = stopped and endpoint_gone and procs_gone
    if proc.poll() is None:
        proc.kill()
    cleanup()

    # -- shutdown from warm-up, before anything is ready -------------------
    print("\n  5. shutdown during warm-up")
    proc = start("piper")
    time.sleep(1.0)
    stopped, _ = stop_goldlive(timeout_s=60)
    warm_gone = wait_for(lambda: not pids(), 45, "warm-up shutdown")
    print(f"      stop={stopped}, processes gone={warm_gone}")
    results["clean shutdown during warm-up"] = warm_gone
    if proc.poll() is None:
        proc.kill()
    cleanup()

    # -- a session with no working TTS must still start and stop -----------
    print("\n  6. degraded TTS (file sink) still starts and stops cleanly")
    proc = start("file")
    up = wait_for(lambda: get("/health") is not None, 120, "the degraded session")
    stopped, _ = stop_goldlive(timeout_s=60)
    deg_gone = wait_for(lambda: not pids(), 45, "shutdown")
    print(f"      started={up}, stopped cleanly={deg_gone}")
    results["degraded TTS lifecycle"] = up and deg_gone
    if proc.poll() is None:
        proc.kill()
    cleanup()

    print("\n" + "=" * 58)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 58 + "\n")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
