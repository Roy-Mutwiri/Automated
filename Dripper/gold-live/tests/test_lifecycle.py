"""User consent: GoldLive runs only when a person has asked it to.

This is the safety property of the whole product. GoldLive speaks through an
audio device and can be routed into a live broadcast, so "did the user ask for
this?" is not a UX detail -- it is the difference between a tool and something
that behaves like malware. Every test here asserts that INSTALL does not imply
RUN, and that STOP is not something the supervisor may undo.
"""

from __future__ import annotations

import json

import pytest

from runtime import lifecycle
from runtime.lifecycle import ACTIVE_STATES, Lifecycle, RunState, load, save


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Never touch the real machine's lifecycle file."""
    path = tmp_path / "lifecycle.json"
    monkeypatch.setattr(lifecycle, "state_path", lambda: path)
    return path


# -- the default is off ----------------------------------------------------


def test_a_fresh_machine_is_stopped():
    """No file means no consent means not running."""
    assert load().state is RunState.STOPPED
    assert not load().may_run


def test_an_unreadable_state_file_means_stopped(isolated_state):
    """When consent cannot be determined, the safe answer is no."""
    isolated_state.write_text('{"state": "runn', encoding="utf-8")
    assert load().state is RunState.STOPPED
    assert not load().may_run


def test_an_unknown_state_value_means_stopped(isolated_state):
    isolated_state.write_text(json.dumps({"state": "definitely-running"}), encoding="utf-8")
    assert load().state is RunState.STOPPED


def test_only_start_requested_and_running_permit_execution():
    """The whole gate rests on this set; widening it silently would let the
    supervisor run in states the user never asked for."""
    assert ACTIVE_STATES == {RunState.START_REQUESTED, RunState.RUNNING}
    for state in RunState:
        lc = Lifecycle(state=state)
        assert lc.may_run is (state in ACTIVE_STATES)


# -- transitions -----------------------------------------------------------


def test_start_then_running_then_stopped(isolated_state):
    lc = lifecycle.request_start()
    assert lc.state is RunState.START_REQUESTED and lc.may_run

    lc = load(reconcile=False)
    lc.mark_running(supervisor_pid=999999)
    save(lc)
    assert load(reconcile=False).state is RunState.RUNNING

    lifecycle.request_stop()
    assert load(reconcile=False).state is RunState.STOP_REQUESTED
    assert not load(reconcile=False).may_run

    lifecycle.mark_stopped()
    assert load().state is RunState.STOPPED
    assert load().supervisor_pid is None


def test_stop_twice_is_safe(isolated_state):
    lifecycle.request_start()
    lifecycle.mark_stopped()
    lifecycle.mark_stopped()
    assert load().state is RunState.STOPPED


def test_stop_requested_immediately_withdraws_permission():
    """The supervisor checks may_run on every tick, so STOP must take effect
    before the graceful shutdown has even finished."""
    lc = Lifecycle(state=RunState.RUNNING)
    assert lc.may_run
    lc.request_stop()
    assert not lc.may_run


# -- a crash is not a stop -------------------------------------------------


def test_a_dead_supervisor_is_reported_as_crashed(isolated_state):
    """RUNNING with no process is a crash. Saying STOPPED would hide it, and
    saying RUNNING would show a green light for a dead system."""
    lc = Lifecycle(state=RunState.RUNNING, supervisor_pid=2_000_000_000)
    save(lc)
    assert load().state is RunState.CRASHED


def test_a_crash_still_does_not_auto_restart(isolated_state):
    """CRASHED must not grant permission to run: this milestone shows the user
    a START button instead of restarting behind their back."""
    save(Lifecycle(state=RunState.RUNNING, supervisor_pid=2_000_000_000))
    assert not load().may_run


def test_a_user_stop_is_distinguishable_from_a_crash(isolated_state):
    lifecycle.mark_stopped()
    assert load().state is RunState.STOPPED

    save(Lifecycle(state=RunState.RUNNING, supervisor_pid=2_000_000_000))
    assert load().state is RunState.CRASHED


def test_a_live_supervisor_stays_running(isolated_state):
    import os

    save(Lifecycle(state=RunState.RUNNING, supervisor_pid=os.getpid()))
    assert load().state is RunState.RUNNING


# -- restart of the application ---------------------------------------------


def test_reopening_after_a_stop_stays_stopped(isolated_state):
    """Simulates closing and reopening the control panel."""
    lifecycle.request_start()
    lifecycle.mark_stopped()
    for _ in range(3):
        assert load().state is RunState.STOPPED
        assert not load().may_run


def test_a_simulated_reboot_stays_stopped(isolated_state):
    """After a reboot no supervisor exists, so a recorded STOPPED must survive
    and must not be reinterpreted as 'should be running'."""
    lifecycle.mark_stopped()
    reloaded = Lifecycle.from_json(json.loads(isolated_state.read_text(encoding="utf-8")))
    assert reloaded.reconcile().state is RunState.STOPPED
    assert not reloaded.may_run


# -- atomicity --------------------------------------------------------------


def test_no_temp_files_are_left_behind(isolated_state):
    for _ in range(3):
        save(Lifecycle())
    leftovers = [p.name for p in isolated_state.parent.iterdir()
                 if p.name.startswith(".lifecycle-")]
    assert leftovers == []


# -- the supervisor honours the recorded intent -----------------------------


def supervisor(**kw):
    from runtime.supervisor import ManagedSession, Supervisor

    return Supervisor([ManagedSession("SESSION_001", 9101, [])], **kw)


def test_the_supervisor_refuses_to_run_when_stopped(isolated_state):
    lifecycle.mark_stopped()
    assert supervisor().consent_to_run() is False


def test_the_supervisor_runs_once_start_is_requested(isolated_state):
    lifecycle.request_start()
    assert supervisor().consent_to_run() is True


def test_stop_withdraws_consent_from_the_supervisor(isolated_state):
    lifecycle.request_start()
    sup = supervisor()
    assert sup.consent_to_run() is True

    lifecycle.request_stop()
    assert sup.consent_to_run() is False, (
        "STOP must take effect on the supervisor's next tick, not whenever the "
        "current session happens to end"
    )


def test_spawn_does_nothing_after_a_user_stop(isolated_state, monkeypatch):
    """The consent bug this whole file exists to prevent:
    STOP -> supervisor notices the child is missing -> supervisor restarts it."""
    import runtime.supervisor as mod

    started: list[str] = []
    monkeypatch.setattr(mod.subprocess, "Popen",
                        lambda *a, **kw: started.append("spawned"))

    sup = supervisor()
    session = sup.sessions["SESSION_001"]

    lifecycle.request_start()
    sup.spawn(session)
    assert started == ["spawned"], "a requested start must actually start"

    lifecycle.request_stop()
    sup.spawn(session)
    assert started == ["spawned"], "STOP must not be undone by the supervisor"


def test_consent_can_be_bypassed_only_explicitly(isolated_state):
    """Tests and debugging need a bypass; it must never be the default."""
    lifecycle.mark_stopped()
    assert supervisor().consent_to_run() is False
    assert supervisor(ignore_consent=True).consent_to_run() is True


def test_an_unreadable_lifecycle_file_denies_the_supervisor(isolated_state):
    isolated_state.write_text("{ broken", encoding="utf-8")
    assert supervisor().consent_to_run() is False


# -- setup must never start anything ---------------------------------------


def test_setup_leaves_the_machine_stopped(isolated_state, monkeypatch):
    """Provisioning is allowed to download gigabytes. It is not allowed to
    start a broadcast."""
    import runtime.setup_wizard as wizard

    monkeypatch.setattr(wizard, "provision", lambda **kw: None, raising=False)
    lifecycle.request_start()  # pretend a previous run left this set

    # The wizard records STOPPED before reporting readiness.
    lifecycle.mark_stopped()
    assert load().state is RunState.STOPPED
    assert not load().may_run


# -- liveness must be a question, not an action -----------------------------


def test_pid_liveness_is_read_only_and_does_not_kill(tmp_path):
    """Regression from a real START test.

    supervisor_alive() used os.kill(pid, 0), which is the idiomatic existence
    test on POSIX. On Windows Python maps os.kill to TerminateProcess, so the
    "harmless" probe either fails outright -- which reported a healthy
    supervisor as crashed -- or kills the process it was only asked about.
    """
    import subprocess
    import sys
    import time

    from runtime.lifecycle import _pid_alive

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert _pid_alive(child.pid), "a running process must read as alive"
        # The probe is asked twice precisely because the old one was destructive.
        assert _pid_alive(child.pid)
        time.sleep(0.3)
        assert child.poll() is None, "checking liveness must not terminate anything"
    finally:
        child.kill()
        child.wait(timeout=10)

    assert not _pid_alive(child.pid), "a dead process must read as dead"


def test_our_own_process_reads_as_alive():
    import os

    from runtime.lifecycle import _pid_alive

    assert _pid_alive(os.getpid())


def test_an_impossible_pid_reads_as_dead():
    from runtime.lifecycle import _pid_alive

    assert not _pid_alive(2_000_000_000)
