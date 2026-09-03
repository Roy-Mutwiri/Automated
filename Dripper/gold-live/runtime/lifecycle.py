"""Whether the user has asked GoldLive to run.

This exists to make one rule enforceable rather than merely intended:

    INSTALL != RUN, and RUNNING requires an explicit START.

The supervisor is built to keep sessions alive, which is right while the user
wants them alive and wrong the moment they press STOP. Without a recorded
intent, "the child process is missing" and "the user stopped it" look identical
from inside the supervisor, and it would dutifully resurrect a session the user
just shut down. That is a consent bug, not a scheduling bug, so the fix is to
write the intent down.

    STOPPED ──START──> START_REQUESTED ──> RUNNING
       ^                                     │
       │                                    STOP
       │                                     │
       └──────── STOPPED <── STOP_REQUESTED ─┘

CRASHED is distinct from STOPPED on purpose. A session that died on its own is
not the same as one the user ended, and the control panel says so -- but this
milestone still does not restart it automatically. The user decides.

The file lives in the writable data directory, never beside the executable,
and is written atomically so a machine that loses power mid-write comes back
with a readable intent rather than a truncated one.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from shared.paths import data_path

log = logging.getLogger(__name__)

FILENAME = "lifecycle.json"
STATE_VERSION = 1


class RunState(str, Enum):
    STOPPED = "stopped"
    START_REQUESTED = "start_requested"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    CRASHED = "crashed"
    ERROR = "error"


#: States in which the supervisor is allowed to spawn or restart a session.
#: Everything else means the user does not want it running.
ACTIVE_STATES = frozenset({RunState.START_REQUESTED, RunState.RUNNING})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pid_alive(pid: int) -> bool:
    """Is this process id still running?

    NOT os.kill(pid, 0). On POSIX that is the idiomatic existence test, but on
    Windows Python maps os.kill to TerminateProcess -- so the "harmless" probe
    either fails outright or kills the process it was only supposed to ask
    about. Using it here reported a perfectly healthy supervisor as crashed,
    and could have taken it down.

    OpenProcess with QUERY_LIMITED_INFORMATION is the read-only question, and
    a process that has exited still has a handle until it is reaped, so the
    exit code has to be checked too.
    """
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32

    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


@dataclass
class Lifecycle:
    state: RunState = RunState.STOPPED
    state_version: int = STATE_VERSION
    #: PID of the supervisor that owns the run, so a stale RUNNING left by a
    #: hard kill can be told apart from one that is genuinely live.
    supervisor_pid: int | None = None
    requested_by: str = "user"
    started_at: str | None = None
    stopped_at: str | None = None
    last_error: str | None = None
    updated_at: str | None = None

    # -- queries ----------------------------------------------------------

    @property
    def may_run(self) -> bool:
        """Is the supervisor permitted to have sessions running right now?"""
        return self.state in ACTIVE_STATES

    @property
    def user_stopped(self) -> bool:
        return self.state in (RunState.STOPPED, RunState.STOP_REQUESTED)

    def supervisor_alive(self) -> bool:
        """Is the recorded supervisor process still there?

        A RUNNING state whose process is gone means a crash, not a running
        system, and the control panel must not show a green light for it.
        """
        pid = self.supervisor_pid
        if not pid:
            return False
        return _pid_alive(pid)

    def reconcile(self) -> Lifecycle:
        """Correct a state that the world has since contradicted.

        Called on load. If we think we are RUNNING but the supervisor process
        is gone, the truth is that it crashed -- and saying so is the whole
        point of distinguishing CRASHED from STOPPED.
        """
        if self.state in (RunState.RUNNING, RunState.START_REQUESTED):
            if not self.supervisor_alive():
                log.warning(
                    "lifecycle said %s but supervisor pid %s is gone; "
                    "recording it as a crash", self.state.value, self.supervisor_pid
                )
                self.state = RunState.CRASHED
                self.supervisor_pid = None
                self.stopped_at = self.stopped_at or _now()
        return self

    # -- transitions ------------------------------------------------------

    def request_start(self) -> None:
        self.state = RunState.START_REQUESTED
        self.started_at = _now()
        self.stopped_at = None
        self.last_error = None

    def mark_running(self, supervisor_pid: int) -> None:
        self.state = RunState.RUNNING
        self.supervisor_pid = supervisor_pid

    def request_stop(self) -> None:
        self.state = RunState.STOP_REQUESTED

    def mark_stopped(self) -> None:
        self.state = RunState.STOPPED
        self.supervisor_pid = None
        self.stopped_at = _now()

    def mark_error(self, reason: str) -> None:
        self.state = RunState.ERROR
        self.supervisor_pid = None
        self.last_error = reason
        self.stopped_at = _now()

    # -- serialisation ----------------------------------------------------

    def to_json(self) -> dict:
        return {
            "state_version": STATE_VERSION,
            "state": self.state.value,
            "supervisor_pid": self.supervisor_pid,
            "requested_by": self.requested_by,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "last_error": self.last_error,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, raw: dict) -> Lifecycle:
        try:
            state = RunState(raw.get("state", "stopped"))
        except ValueError:
            state = RunState.STOPPED
        return cls(
            state=state,
            state_version=raw.get("state_version", STATE_VERSION),
            supervisor_pid=raw.get("supervisor_pid"),
            requested_by=raw.get("requested_by", "user"),
            started_at=raw.get("started_at"),
            stopped_at=raw.get("stopped_at"),
            last_error=raw.get("last_error"),
            updated_at=raw.get("updated_at"),
        )


# -- persistence ----------------------------------------------------------


def state_path() -> Path:
    return data_path(FILENAME, create_parent=True)


def save(lc: Lifecycle, path: Path | None = None) -> Path:
    target = path or state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    lc.updated_at = _now()

    payload = json.dumps(lc.to_json(), indent=2)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".lifecycle-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return target


def load(path: Path | None = None, reconcile: bool = True) -> Lifecycle:
    """Read the recorded intent.

    An unreadable file means STOPPED. That default is deliberate: when in
    doubt about whether the user wanted this running, the safe answer is no.
    """
    target = path or state_path()
    if not target.exists():
        return Lifecycle()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("lifecycle state must be a JSON object")
    except (OSError, ValueError) as exc:
        log.warning("lifecycle state unreadable (%s); assuming STOPPED", exc)
        return Lifecycle()

    lc = Lifecycle.from_json(raw)
    return lc.reconcile() if reconcile else lc


# -- convenience used by the CLI and the control panel --------------------


def request_start() -> Lifecycle:
    lc = load()
    lc.request_start()
    save(lc)
    return lc


def request_stop() -> Lifecycle:
    lc = load()
    lc.request_stop()
    save(lc)
    return lc


def mark_stopped() -> Lifecycle:
    lc = load(reconcile=False)
    lc.mark_stopped()
    save(lc)
    return lc


def current() -> RunState:
    return load().state
