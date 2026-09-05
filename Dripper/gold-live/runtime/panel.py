"""The GoldLive control panel: an explicit START and an explicit STOP.

The project grew up as a background service, which is the wrong shape for
something a person installs on their own PC. A broadcasting application that
can speak through your microphone must be obvious about whether it is running,
and must never decide that for you. So:

  * opening this window starts nothing
  * closing it stops nothing and starts nothing
  * only the START button starts GoldLive
  * STOP means stop, and the supervisor is not allowed to undo it

tkinter on purpose: it ships with Python, adds no dependency, and packages
cleanly. This is a control panel, not a product surface -- it should be
legible and boring.

Status is read from the session's own /health endpoint rather than inferred
from "a process exists", because a process that is alive and mute is exactly
the failure this project has already been bitten by.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from runtime import lifecycle
from runtime.lifecycle import RunState
from shared.paths import data_path, data_root, is_frozen
from shared.version import app_version

log = logging.getLogger(__name__)

HEALTH_PORT = 9101
POLL_INTERVAL_MS = 1500
STOP_TIMEOUT_S = 30.0

# Panel states, which are a superset of the lifecycle states: the panel also
# has to describe a machine that has not been provisioned at all.
NOT_INSTALLED = "NOT INSTALLED"
READY_TO_START = "READY TO START"


@dataclass
class Snapshot:
    """Everything the panel draws, gathered off the UI thread."""

    panel_state: str = NOT_INSTALLED
    detail: str = ""
    session: str = "SESSION_001"
    market: str = "-"
    model: str = "-"
    tts: str = "-"
    audio: str = "-"
    comments: str = "-"
    broadcast: str = "NOT CONFIGURED"
    readiness: str = "-"
    uptime: str = "-"
    last_error: str = "None"
    can_start: bool = False
    can_stop: bool = False


def _health(port: int = HEALTH_PORT, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def _component(health: dict, name: str) -> str:
    """Find a component by exact name, then by suffix.

    The suffix match matters: the comment adapter publishes its own name, so
    it is file_comment_adapter, screen_comment_adapter or youtube_comment_adapter
    depending on which one is running. Matching only the exact string left the
    Comments row permanently blank whichever adapter was in use.
    """
    components = health.get("components", [])
    for c in components:
        if c.get("component") == name:
            return str(c.get("state", "-")).upper()
    for c in components:
        if str(c.get("component", "")).endswith(name):
            return str(c.get("state", "-")).upper()
    return "-"


def _provisioned() -> tuple[bool, str]:
    try:
        from runtime.provision import needs_provisioning

        needed, why = needs_provisioning()
        return (not needed), why
    except Exception as exc:
        return False, f"could not read provisioning state: {exc}"


def gather() -> Snapshot:
    """Build a snapshot from recorded intent plus live health.

    Intent first: what the user asked for. Health second: what is actually
    true. RUNNING is only ever reported when both agree.
    """
    snap = Snapshot()
    installed, why = _provisioned()
    if not installed:
        snap.panel_state = NOT_INSTALLED
        snap.detail = why
        return snap

    lc = lifecycle.load()
    snap.last_error = lc.last_error or "None"
    health = _health()

    if lc.state is RunState.RUNNING and health:
        snap.panel_state = "RUNNING"
        snap.can_stop = True
    elif lc.state is RunState.RUNNING and not health:
        # The supervisor is alive (reconcile would have said CRASHED otherwise)
        # but the session is not answering yet.
        snap.panel_state = "STARTING"
        snap.detail = "waiting for the session to become healthy"
        snap.can_stop = True
    elif lc.state is RunState.START_REQUESTED:
        snap.panel_state = "STARTING"
        snap.detail = "running readiness checks (this takes 10-30 seconds)"
        snap.can_stop = True
    elif lc.state is RunState.STOP_REQUESTED:
        snap.panel_state = "STOPPING"
        snap.can_stop = False
    elif lc.state is RunState.CRASHED:
        snap.panel_state = "ERROR"
        snap.detail = "GoldLive stopped unexpectedly."
        snap.can_start = True
    elif lc.state is RunState.ERROR:
        snap.panel_state = "ERROR"
        snap.detail = lc.last_error or "start failed"
        snap.can_start = True
    else:
        snap.panel_state = "STOPPED"
        snap.can_start = True

    if health:
        snap.market = _component(health, "market_engine")
        snap.model = _component(health, "generation")
        snap.audio = _component(health, "audio_router")
        snap.comments = _component(health, "comment_adapter")
        snap.tts = snap.audio
        secs = int(health.get("uptime_s", 0))
        snap.uptime = f"{secs // 3600}h {secs % 3600 // 60}m {secs % 60}s"
        if str(health.get("state", "")).lower() in ("degraded", "failing"):
            snap.panel_state = "DEGRADED" if snap.panel_state == "RUNNING" else snap.panel_state

    try:
        cached = data_path("readiness.json", create_parent=False)
        if cached.exists():
            snap.readiness = json.loads(cached.read_text(encoding="utf-8")).get("level", "-")
    except Exception:
        pass
    return snap


# -- starting and stopping -------------------------------------------------


def _supervisor_command() -> list[str]:
    if is_frozen():
        return [sys.executable, "supervise", "--sessions", "SESSION_001"]
    return [sys.executable, "-m", "runtime.supervisor", "--sessions", "SESSION_001"]


def start_goldlive() -> subprocess.Popen:
    """Launch the supervisor. The only place the panel starts anything."""
    lifecycle.request_start()

    log_path = data_path("logs", "supervisor.log")
    creation = 0
    if os.name == "nt":
        # No console window for the child, and its own process group so a
        # Ctrl-C in the panel's console cannot take the supervisor with it.
        creation = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

    # The handle is closed in this process as soon as the child has inherited
    # it. Holding it open leaked a descriptor on every START, and the panel is
    # meant to run for as long as the user leaves it open.
    with open(log_path, "a", encoding="utf-8", buffering=1) as handle:
        proc = subprocess.Popen(
            _supervisor_command(), cwd=str(data_root()),
            stdout=handle, stderr=handle, creationflags=creation,
        )
    log.info("supervisor started (pid %d); log: %s", proc.pid, log_path)
    return proc


def stop_goldlive(timeout_s: float = STOP_TIMEOUT_S) -> tuple[bool, str]:
    """Ask GoldLive to stop, then make sure it actually did.

    Recording the intent comes first and matters most: the supervisor checks it
    on every tick, so from this moment it will not restart anything, even if a
    child dies while we are still waiting for a graceful exit.
    """
    lc = lifecycle.request_stop()
    pid = lc.supervisor_pid

    if pid:
        try:
            os.kill(pid, signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        except (OSError, ValueError, AttributeError):
            with contextlib.suppress(OSError, ValueError):
                os.kill(pid, signal.SIGTERM)

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not lifecycle.load(reconcile=False).supervisor_alive():
            break
        time.sleep(0.5)

    remaining = _goldlive_children()
    if remaining:
        log.warning("forcing %d remaining GoldLive process(es) to stop", len(remaining))
        for child_pid in remaining:
            with contextlib.suppress(OSError, ValueError):
                os.kill(child_pid, signal.SIGTERM)
        time.sleep(2.0)
        remaining = _goldlive_children()
        for child_pid in remaining:
            with contextlib.suppress(OSError, ValueError):
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/PID", str(child_pid)],
                                   capture_output=True, timeout=10)
                else:
                    os.kill(child_pid, signal.SIGKILL)

    lifecycle.mark_stopped()
    leftover = _goldlive_children()
    if leftover:
        return False, f"{len(leftover)} GoldLive process(es) would not stop: {leftover}"
    return True, "stopped"


def _goldlive_children() -> list[int]:
    """PIDs of GoldLive processes other than this one.

    Deliberately matches only our own executable name -- STOP must never reach
    outside GoldLive.
    """
    me = os.getpid()
    name = "GoldLive.exe" if is_frozen() else None
    if name is None:
        return []
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {name}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return []
    pids = []
    for line in (out.stdout or "").splitlines():
        parts = [p.strip('" ') for p in line.split('","')]
        if len(parts) >= 2 and parts[1].isdigit():
            pid = int(parts[1])
            if pid != me:
                pids.append(pid)
    return pids


# -- the window ------------------------------------------------------------


def run_panel() -> int:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title(f"GoldLive {app_version()}")
    root.geometry("520x560")
    root.minsize(480, 520)

    results: queue.Queue[Snapshot] = queue.Queue()
    busy = {"working": False}

    header = tk.Label(root, text="GOLDLIVE", font=("Segoe UI", 20, "bold"))
    header.pack(pady=(16, 4))

    status_var = tk.StringVar(value="checking...")
    status = tk.Label(root, textvariable=status_var, font=("Segoe UI", 16, "bold"))
    status.pack(pady=(0, 2))

    detail_var = tk.StringVar(value="")
    tk.Label(root, textvariable=detail_var, font=("Segoe UI", 9),
             fg="#555", wraplength=460).pack(pady=(0, 10))

    buttons = tk.Frame(root)
    buttons.pack(pady=4)
    start_btn = tk.Button(buttons, text="START", width=16, height=2,
                          font=("Segoe UI", 11, "bold"), state=tk.DISABLED)
    stop_btn = tk.Button(buttons, text="STOP", width=16, height=2,
                         font=("Segoe UI", 11, "bold"), state=tk.DISABLED)
    start_btn.grid(row=0, column=0, padx=6)
    stop_btn.grid(row=0, column=1, padx=6)

    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=16, pady=12)

    grid = tk.Frame(root)
    grid.pack(fill="x", padx=24)
    fields = ("Session", "Market", "AI Model", "Piper", "Audio",
              "Comments", "Broadcast", "Readiness", "Uptime", "Last error")
    values = {}
    for i, name in enumerate(fields):
        tk.Label(grid, text=f"{name}:", font=("Segoe UI", 9),
                 anchor="w", width=12).grid(row=i, column=0, sticky="w", pady=1)
        var = tk.StringVar(value="-")
        tk.Label(grid, textvariable=var, font=("Consolas", 9),
                 anchor="w", wraplength=300, justify="left").grid(
            row=i, column=1, sticky="w", pady=1)
        values[name] = var

    footer = tk.Frame(root)
    footer.pack(side="bottom", pady=12)
    tk.Label(
        root,
        text="GoldLive does not start on its own and does not run after you close it.",
        font=("Segoe UI", 8), fg="#777", wraplength=460,
    ).pack(side="bottom", pady=(0, 4))

    def open_logs() -> None:
        path = data_root() / "logs"
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(path)

    tk.Button(footer, text="Logs", width=12, command=open_logs).grid(row=0, column=0, padx=4)

    def diagnostics() -> None:
        snap = gather()
        messagebox.showinfo(
            "Diagnostics",
            f"State: {snap.panel_state}\nData: {data_root()}\n"
            f"Readiness: {snap.readiness}\nLast error: {snap.last_error}\n\n"
            f"{snap.detail}",
        )

    tk.Button(footer, text="Diagnostics", width=12,
              command=diagnostics).grid(row=0, column=1, padx=4)

    COLOURS = {
        "RUNNING": "#1a7f37", "DEGRADED": "#9a6700", "STOPPED": "#57606a",
        "STARTING": "#0969da", "STOPPING": "#0969da", "ERROR": "#cf222e",
        NOT_INSTALLED: "#cf222e", READY_TO_START: "#1a7f37",
    }

    def draw(snap: Snapshot) -> None:
        dot = "●"
        status_var.set(f"{dot} {snap.panel_state}")
        status.config(fg=COLOURS.get(snap.panel_state, "#57606a"))
        detail_var.set(snap.detail)
        values["Session"].set(snap.session)
        values["Market"].set(snap.market)
        values["AI Model"].set(snap.model)
        values["Piper"].set(snap.tts)
        values["Audio"].set(snap.audio)
        values["Comments"].set(snap.comments)
        values["Broadcast"].set(snap.broadcast)
        values["Readiness"].set(snap.readiness)
        values["Uptime"].set(snap.uptime)
        values["Last error"].set(snap.last_error)
        if not busy["working"]:
            start_btn.config(state=tk.NORMAL if snap.can_start else tk.DISABLED)
            stop_btn.config(state=tk.NORMAL if snap.can_stop else tk.DISABLED)

    def poll() -> None:
        if not busy["working"]:
            threading.Thread(target=lambda: results.put(gather()), daemon=True).start()
        try:
            while True:
                draw(results.get_nowait())
        except queue.Empty:
            pass
        root.after(POLL_INTERVAL_MS, poll)

    def on_start() -> None:
        busy["working"] = True
        start_btn.config(state=tk.DISABLED)
        stop_btn.config(state=tk.DISABLED)
        status_var.set("● STARTING")
        detail_var.set("starting the supervisor and running readiness checks...")

        def work() -> None:
            try:
                start_goldlive()
            except Exception as exc:
                log.exception("start failed")
                lc = lifecycle.load(reconcile=False)
                lc.mark_error(str(exc))
                lifecycle.save(lc)
            finally:
                busy["working"] = False

        threading.Thread(target=work, daemon=True).start()

    def on_stop() -> None:
        if not messagebox.askyesno(
            "Stop GoldLive?",
            "This will stop the active GoldLive session.\n\nStop now?",
        ):
            return
        busy["working"] = True
        start_btn.config(state=tk.DISABLED)
        stop_btn.config(state=tk.DISABLED)
        status_var.set("● STOPPING")
        detail_var.set("asking GoldLive to shut down cleanly...")

        def work() -> None:
            ok, why = stop_goldlive()
            if not ok:
                log.error("stop incomplete: %s", why)
            busy["working"] = False

        threading.Thread(target=work, daemon=True).start()

    start_btn.config(command=on_start)
    stop_btn.config(command=on_stop)

    def on_close() -> None:
        # Closing the window is not a stop and not a start. If GoldLive is
        # running the user is told, because a broadcasting application that
        # keeps running after its window closes must say so out loud.
        if lifecycle.load().may_run:
            keep = messagebox.askyesnocancel(
                "GoldLive is running",
                "GoldLive is still RUNNING.\n\n"
                "Yes  - stop GoldLive and close\n"
                "No   - leave it running and close this window\n"
                "Cancel - go back",
            )
            if keep is None:
                return
            if keep:
                stop_goldlive()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    poll()
    root.mainloop()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        return run_panel()
    except ImportError as exc:
        print(f"\n  The control panel needs tkinter, which is missing: {exc}\n"
              "  Use the command line instead:  GoldLive.exe supervise\n")
        return 1
