"""Supervisor behaviour.

The component the entire reliability story rests on, and it had no tests. Its
job is to be correct exactly when everything else is broken, which is the worst
time to discover its restart policy is wrong.

Processes are faked: spawning real ones would make these slow and flaky, and
the logic under test is the policy, not subprocess plumbing.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from runtime.supervisor import ManagedSession, Supervisor


class FakeProcess:
    """Stands in for subprocess.Popen. `alive` is flipped by the test."""

    def __init__(self, pid: int = 1234) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.ignores_sigterm = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if not self.ignores_sigterm:
            self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def session(session_id: str = "SESSION_001", port: int = 9101) -> ManagedSession:
    return ManagedSession(session_id=session_id, health_port=port, args=[])


def supervisor(sessions=None, **kw) -> Supervisor:
    # These tests exercise supervision mechanics -- restart, backoff, crash
    # loops -- not user consent, which has its own suite in test_lifecycle.py.
    # Without this they would depend on whatever the developer's real
    # lifecycle.json happens to say, which is both flaky and beside the point.
    kw.setdefault("ignore_consent", True)
    return Supervisor(sessions or [session()], **kw)


# -- restart accounting ----------------------------------------------------


def test_restart_window_is_rolling():
    """Failures from two hours ago must not count toward the crash-loop limit;
    a long-running session that failed twice last week is healthy."""
    s = session()
    now = time.time()
    s.restarts = [now - 7200, now - 4000, now - 60, now - 10]
    assert s.restarts_last_hour() == 2


def test_uptime_and_liveness_track_the_process():
    s = session()
    assert not s.alive
    s.process = FakeProcess()
    s.started_at = time.time() - 30
    assert s.alive
    assert 29 <= s.uptime_s <= 31

    s.process.returncode = 1
    assert not s.alive


# -- crash-loop policy -----------------------------------------------------


async def test_gives_up_after_repeated_failures(monkeypatch):
    """A crash loop that restarts silently is worse than one that stops,
    because nobody finds out."""
    s = session()
    sup = supervisor([s], crash_loop_threshold=3)

    spawns = []
    monkeypatch.setattr(sup, "spawn", lambda sess: spawns.append(time.time()))
    monkeypatch.setattr(sup, "check_ready", lambda sess: _true())

    s.restarts = [time.time()] * 3
    s.state = "running"
    task = asyncio.create_task(sup._watch(s))
    await asyncio.sleep(0.15)
    sup.stopping.set()
    task.cancel()

    assert s.give_up
    assert s.state == "crash_loop"
    assert spawns == [], "must stop respawning once it has given up"


async def test_healthy_session_is_not_given_up_on(monkeypatch):
    s = session()
    s.process = FakeProcess()
    s.started_at = time.time() - 300
    sup = supervisor([s])
    monkeypatch.setattr(sup, "check_ready", lambda sess: _true())

    task = asyncio.create_task(sup._watch(s))
    await asyncio.sleep(0.1)
    sup.stopping.set()
    task.cancel()

    assert not s.give_up
    assert s.state == "running"


async def _true():
    return True


async def _false():
    return False


# -- readiness and hangs ---------------------------------------------------


async def test_a_wedged_process_is_killed(monkeypatch):
    """A process that stays up but stops serving is the failure supervision
    usually misses -- it never exits, so nothing restarts it."""
    s = session()
    s.process = FakeProcess()
    s.started_at = time.time() - 300
    s.last_ready_at = time.time() - 600  # unready for ten minutes

    sup = supervisor([s], unready_grace_s=60)
    monkeypatch.setattr(sup, "check_ready", lambda sess: _false())

    task = asyncio.create_task(sup._watch(s))
    await asyncio.sleep(0.2)
    sup.stopping.set()
    task.cancel()

    assert s.process.terminated, "a hung process must be killed so it can restart"


async def test_briefly_unready_is_tolerated(monkeypatch):
    """Startup, a model reload, a slow health check -- none of these should
    trigger a kill."""
    s = session()
    s.process = FakeProcess()
    s.started_at = time.time() - 10
    s.last_ready_at = time.time() - 5

    sup = supervisor([s], unready_grace_s=120)
    monkeypatch.setattr(sup, "check_ready", lambda sess: _false())

    task = asyncio.create_task(sup._watch(s))
    await asyncio.sleep(0.15)
    sup.stopping.set()
    task.cancel()

    assert not s.process.terminated
    assert s.state == "unready"


async def test_unreachable_health_endpoint_counts_as_not_ready():
    sup = supervisor()
    # Port 1 is reserved and never listening.
    assert await sup.check_ready(session(port=1)) is False


# -- termination -----------------------------------------------------------


async def test_sigterm_before_sigkill():
    """SIGTERM lets the session checkpoint what it covered and flush traces.
    Killing it outright means the restarted session forgets and repeats."""
    s = session()
    s.process = FakeProcess()
    sup = supervisor([s], term_grace_s=1.0)

    await sup.terminate(s)
    assert s.process.terminated
    assert not s.process.killed
    assert s.state == "stopped"


async def test_sigkill_when_sigterm_is_ignored():
    s = session()
    s.process = FakeProcess()
    s.process.ignores_sigterm = True
    sup = supervisor([s], term_grace_s=0.3)

    await sup.terminate(s)
    assert s.process.terminated
    assert s.process.killed, "must escalate rather than hang forever"


async def test_terminating_a_dead_process_is_harmless():
    s = session()
    await supervisor([s]).terminate(s)  # no process at all
    s.process = FakeProcess()
    s.process.returncode = 0
    await supervisor([s]).terminate(s)


# -- backoff ---------------------------------------------------------------


def test_backoff_grows_and_is_capped():
    s = session()
    sup = supervisor([s], max_backoff_s=10.0)
    backoff = s.backoff_s
    for _ in range(10):
        backoff = min(backoff * 2, sup.max_backoff_s)
    assert backoff == 10.0


async def test_restart_delay_is_jittered(monkeypatch):
    """Seven sessions restarting in lockstep would hit the model server all at
    once, which is how a recoverable blip becomes an outage."""
    delays: list[float] = []

    s = session()
    sup = supervisor([s])
    monkeypatch.setattr(sup, "spawn", lambda sess: None)
    monkeypatch.setattr(sup, "check_ready", lambda sess: _true())

    async def capture(d):
        delays.append(d)
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", capture)
    with pytest.raises(asyncio.CancelledError):
        await sup._watch(s)

    assert delays and delays[0] != 1.0, "delay should be jittered, not exactly the backoff"
    assert 0.5 < delays[0] < 2.0


# -- reporting -------------------------------------------------------------


def test_snapshot_exposes_what_an_operator_needs():
    s = session()
    s.process = FakeProcess()
    s.started_at = time.time() - 120
    s.restarts = [time.time()]
    snap = supervisor([s]).snapshot()

    assert snap["sessions"][0]["session_id"] == "SESSION_001"
    assert snap["sessions"][0]["alive"] is True
    assert snap["sessions"][0]["restarts_1h"] == 1
    assert snap["sessions"][0]["given_up"] is False


def test_build_sessions_assigns_distinct_ports():
    from runtime.supervisor import build_sessions

    sessions = build_sessions(None, ["--mode", "offline"])
    ports = [s.health_port for s in sessions]
    assert len(set(ports)) == len(ports), "each session needs its own health port"
    assert all("--mode" in s.args for s in sessions)


def test_selecting_a_missing_session_fails_loudly():
    from runtime.supervisor import build_sessions

    with pytest.raises(SystemExit):
        build_sessions(["SESSION_DOES_NOT_EXIST"], [])


# -- session control endpoints ---------------------------------------------


async def test_control_endpoints_are_served(tmp_path):
    """Operator controls must be reachable while a session is running, since
    the alternative -- restarting to change something -- loses the session's
    memory of what it has covered."""
    import json as _json
    import urllib.request

    from runtime.health import HealthServer
    from shared.contracts import HealthState, ServiceHealth

    calls: list[tuple[str, dict]] = []

    def make(name):
        def action(payload):
            calls.append((name, payload))
            return {"ok": name}
        return action

    server = HealthServer(
        provider=lambda: [ServiceHealth(component="t", state=HealthState.OK)],
        port=9199,
        controls={n: make(n) for n in ("pause", "resume", "say")},
    )
    server.start()
    try:
        with urllib.request.urlopen("http://127.0.0.1:9199/controls", timeout=3) as r:
            assert set(_json.loads(r.read())["available"]) == {"pause", "resume", "say"}

        req = urllib.request.Request(
            "http://127.0.0.1:9199/control/say",
            data=b'{"text": "hello"}', method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            assert _json.loads(r.read()) == {"ok": "say"}
        assert calls == [("say", {"text": "hello"})]
    finally:
        server.stop()


async def test_unknown_control_lists_what_is_available():
    import json as _json
    import urllib.error
    import urllib.request

    from runtime.health import HealthServer
    from shared.contracts import HealthState, ServiceHealth

    server = HealthServer(
        provider=lambda: [ServiceHealth(component="t", state=HealthState.OK)],
        port=9198,
        controls={"pause": lambda p: {"ok": True}},
    )
    server.start()
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:9198/control/nope", data=b"{}", method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=3)
        body = _json.loads(exc.value.read())
        assert body["available"] == ["pause"]
    finally:
        server.stop()


async def test_a_failing_control_does_not_take_down_the_server():
    """A control is meant to manage a session, not be a way to crash it."""
    import urllib.error
    import urllib.request

    from runtime.health import HealthServer
    from shared.contracts import HealthState, ServiceHealth

    def boom(_payload):
        raise RuntimeError("control exploded")

    server = HealthServer(
        provider=lambda: [ServiceHealth(component="t", state=HealthState.OK)],
        port=9197, controls={"boom": boom},
    )
    server.start()
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:9197/control/boom", data=b"{}", method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=3)
        assert exc.value.code == 500

        # Still serving afterwards.
        with urllib.request.urlopen("http://127.0.0.1:9197/health", timeout=3) as r:
            assert r.status == 200
    finally:
        server.stop()
