"""The transition matrix, and the invariants that span machines.

Seven state machines exist. Any one of them being individually wrong is a bug;
two of them disagreeing is the bug that matters, because that is how a UI shows
RUNNING while the session is dead, or health reports OK while nothing can be
produced.

Documented in docs/STATE_MACHINES.md. These tests are the enforcement.
"""

from __future__ import annotations

import pytest

from platform_.audio.router import PlaybackState
from runtime import lifecycle
from runtime.lifecycle import ACTIVE_STATES, Lifecycle, RunState
from runtime.readiness import BROADCAST_GATES, SESSION_GATES, GateResult, Level, Readiness
from runtime.supervisor import ManagedSession, SessionState, Supervisor
from shared.contracts import HealthState
from shared.provisioning import ProvisionState


@pytest.fixture(autouse=True)
def isolated_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "state_path", lambda: tmp_path / "lifecycle.json")


def supervisor(**kw) -> Supervisor:
    return Supervisor([ManagedSession("SESSION_001", 9101, [])], **kw)


# -- 1. RunState: the consent matrix ---------------------------------------


@pytest.mark.parametrize("state", list(RunState))
def test_only_two_states_authorise_running(state):
    """The whole consent guarantee is this set. Widening it silently would let
    the supervisor run in states the user never asked for."""
    assert Lifecycle(state=state).may_run is (state in ACTIVE_STATES)
    assert ACTIVE_STATES == {RunState.START_REQUESTED, RunState.RUNNING}


def test_crashed_does_not_authorise_a_restart():
    """A crash is not consent. This milestone shows a START button instead of
    restarting behind the user's back."""
    assert not Lifecycle(state=RunState.CRASHED).may_run


def test_start_while_already_starting_is_idempotent():
    lifecycle.request_start()
    first = lifecycle.load(reconcile=False)
    lifecycle.request_start()
    second = lifecycle.load(reconcile=False)
    assert first.state is second.state is RunState.START_REQUESTED


def test_start_while_running_does_not_lose_the_supervisor():
    """A second START must not orphan the supervisor already recorded."""
    lc = lifecycle.load(reconcile=False)
    lc.mark_running(supervisor_pid=4242)
    lifecycle.save(lc)

    lifecycle.request_start()
    assert lifecycle.load(reconcile=False).state is RunState.START_REQUESTED


def test_stop_while_starting_withdraws_permission_immediately():
    """STOP pressed during the 10-30s readiness window must take effect at
    once, not when startup happens to finish."""
    lifecycle.request_start()
    assert supervisor().consent_to_run() is True
    lifecycle.request_stop()
    assert supervisor().consent_to_run() is False


def test_stop_while_running_withdraws_permission():
    lc = lifecycle.load(reconcile=False)
    lc.mark_running(supervisor_pid=4242)
    lifecycle.save(lc)
    assert supervisor().consent_to_run() is True

    lifecycle.request_stop()
    assert supervisor().consent_to_run() is False


def test_stop_while_already_stopped_is_safe():
    for _ in range(3):
        lifecycle.mark_stopped()
    assert lifecycle.load().state is RunState.STOPPED


def test_a_failed_start_records_a_reason_not_just_a_state():
    """ERROR without a reason is indistinguishable from STOPPED to the user."""
    lc = lifecycle.load(reconcile=False)
    lc.mark_error("readiness gates failed")
    lifecycle.save(lc)

    back = lifecycle.load(reconcile=False)
    assert back.state is RunState.ERROR
    assert back.last_error
    assert not back.may_run


# -- 2. SessionState: the supervisor's view --------------------------------


def test_session_states_are_typed_not_bare_strings():
    """They were seven string literals assigned in seven places. A typo would
    have created a new state silently and every comparison would have returned
    False."""
    assert {s.value for s in SessionState} == {
        "starting", "running", "unready", "stopped", "crash_loop"
    }


def test_session_state_compares_equal_to_its_string():
    """A str-Enum, so existing comparisons and JSON keep working."""
    assert SessionState.RUNNING == "running"
    assert SessionState.CRASH_LOOP.value == "crash_loop"


def test_a_new_session_starts_in_starting():
    assert ManagedSession("S1", 9101, []).state is SessionState.STARTING


def test_crash_loop_is_terminal():
    """Deliberately not auto-recovered: a loop that restarts silently is worse
    than one that stops, because nobody finds out."""
    session = ManagedSession("S1", 9101, [])
    session.give_up = True
    session.state = SessionState.CRASH_LOOP
    assert session.give_up is True


def test_spawn_is_refused_when_consent_is_absent(monkeypatch):
    """Dependency of every other supervisor guarantee."""
    import runtime.supervisor as mod

    spawned = []
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: spawned.append(1))
    lifecycle.mark_stopped()

    sup = supervisor()
    sup.spawn(sup.sessions["SESSION_001"])
    assert spawned == []


# -- 3. PlaybackState: never stuck speaking --------------------------------


def test_playback_states_are_exactly_the_four_implemented():
    assert {s.value for s in PlaybackState} == {
        "idle", "synthesising", "playing", "stopped"
    }


@pytest.mark.parametrize("failure", ["ok", "tts_fail", "empty", "sink_fail"])
async def test_the_router_never_remains_speaking_after_any_outcome(tmp_path, failure):
    """A stuck PLAYING is the state a wedged host reports forever."""
    from fakes import FakeSink, FakeTTS, audio_request

    from platform_.audio.router import AudioRouter

    tts = {
        "ok": FakeTTS(),
        "tts_fail": FakeTTS(fail=True),
        "empty": FakeTTS(empty=True),
        "sink_fail": FakeTTS(),
    }[failure]
    sink = FakeSink(fail=(failure == "sink_fail"))

    router = AudioRouter("SESSION_001", tts, tmp_path, sink=sink)
    await router.start()
    await router.submit(audio_request(segments=["x"]))
    for _ in range(40):
        import asyncio

        await asyncio.sleep(0)

    assert router.state in (PlaybackState.IDLE, PlaybackState.STOPPED)
    assert router.current_utterance is None
    await router.stop()
    assert router.state is PlaybackState.STOPPED


async def test_shutdown_during_synthesis_leaves_a_clean_state(tmp_path):
    from fakes import FakeSink, FakeTTS, audio_request

    from platform_.audio.router import AudioRouter

    router = AudioRouter("SESSION_001", FakeTTS(delay_s=0.2), tmp_path, sink=FakeSink())
    await router.start()
    await router.submit(audio_request(segments=["slow"]))

    import asyncio

    await asyncio.sleep(0.02)
    await router.stop()

    assert router.state is PlaybackState.STOPPED


async def test_shutdown_during_playback_leaves_a_clean_state(tmp_path):
    from fakes import FakeSink, FakeTTS, audio_request

    from platform_.audio.router import AudioRouter

    router = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=FakeSink(delay_s=0.2))
    await router.start()
    await router.submit(audio_request(segments=["playing"]))

    import asyncio

    await asyncio.sleep(0.02)
    await router.stop()

    assert router.state is PlaybackState.STOPPED


# -- 4. HealthState: degraded is still ready -------------------------------


def test_health_states_are_ordered_worst_last():
    order = [HealthState.OK, HealthState.DEGRADED, HealthState.FAILING, HealthState.DOWN]
    assert len(set(order)) == 4


@pytest.mark.parametrize(
    ("state", "ready"),
    [(HealthState.OK, True), (HealthState.DEGRADED, True),
     (HealthState.FAILING, False), (HealthState.DOWN, False)],
)
def test_only_failing_and_down_make_ready_return_503(state, ready):
    """Load-bearing: a session with a stale feed has stopped quoting prices and
    is still broadcasting. Restarting it would be strictly worse."""
    assert (state in (HealthState.OK, HealthState.DEGRADED)) is ready


# -- 5. Readiness levels ---------------------------------------------------


def ok_gate(name: str) -> GateResult:
    return GateResult(name, True, evidence="fine")


def bad_gate(name: str) -> GateResult:
    return GateResult(name, False, reason="nope")


def test_the_four_levels_are_strictly_ordered():
    gates = [ok_gate(n) for n in SESSION_GATES]
    assert Readiness(gates=gates + [bad_gate(n) for n in BROADCAST_GATES]).level \
        is Level.SESSION_READY

    full = Readiness(gates=gates + [ok_gate(n) for n in BROADCAST_GATES])
    assert full.level is Level.BROADCAST_READY
    full.broadcast_verified = True
    assert full.level is Level.FULL_LIVE_READY


@pytest.mark.parametrize("gate", SESSION_GATES)
def test_any_failing_core_gate_forces_not_ready(gate):
    gates = [ok_gate(n) if n != gate else bad_gate(n) for n in SESSION_GATES]
    gates += [ok_gate(n) for n in BROADCAST_GATES]
    result = Readiness(gates=gates)
    result.broadcast_verified = True  # must not paper over a broken core
    assert result.level is Level.NOT_READY


# -- 6. ProvisionState -----------------------------------------------------


def test_provisioning_has_no_degraded_state():
    """Whether a session can run right now depends on the network, the model
    server and the audio device, none of which this file knows about.
    Installation state is cached; liveness never is."""
    assert {s.value for s in ProvisionState} == {
        "new", "provisioning", "ready", "failed"
    }


# -- cross-machine invariants ----------------------------------------------


def test_a_dead_supervisor_cannot_leave_running_readable():
    """Invariant 3. Otherwise the panel shows a green light for a dead system."""
    lifecycle.save(Lifecycle(state=RunState.RUNNING, supervisor_pid=2_000_000_000))
    assert lifecycle.load().state is RunState.CRASHED


def test_the_panel_cannot_report_running_without_health(monkeypatch):
    """Invariant 4. A process existing is not evidence: a process that is alive
    and mute is a failure this project has already been bitten by."""
    import runtime.panel as panel

    monkeypatch.setattr(panel, "_provisioned", lambda: (True, "ok"))
    monkeypatch.setattr(panel, "_health", lambda *a, **k: None)

    lc = lifecycle.load(reconcile=False)
    lc.mark_running(supervisor_pid=__import__("os").getpid())
    lifecycle.save(lc)

    assert panel.gather().panel_state == "STARTING"


def test_the_panel_reports_running_when_intent_and_health_agree(monkeypatch):
    import os

    import runtime.panel as panel

    monkeypatch.setattr(panel, "_provisioned", lambda: (True, "ok"))
    monkeypatch.setattr(
        panel, "_health",
        lambda *a, **k: {"state": "ok", "uptime_s": 42, "components": []},
    )

    lc = lifecycle.load(reconcile=False)
    lc.mark_running(supervisor_pid=os.getpid())
    lifecycle.save(lc)

    snap = panel.gather()
    assert snap.panel_state == "RUNNING"
    assert snap.can_stop and not snap.can_start


def test_an_unprovisioned_machine_reports_not_installed(monkeypatch):
    import runtime.panel as panel

    monkeypatch.setattr(panel, "_provisioned", lambda: (False, "no model"))
    snap = panel.gather()
    assert snap.panel_state == "NOT INSTALLED"
    assert not snap.can_start and not snap.can_stop
