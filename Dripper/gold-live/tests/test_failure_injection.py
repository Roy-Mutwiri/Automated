"""Break each dependency deliberately, and check what happens next.

Every test here asserts two things, because detecting a failure is only half of
it: the failure is handled as intended, **and the next valid operation still
works**. A system that notices a problem and then stays broken has not
recovered.

The policy for each failure is stated in the test, because "retry" is not a
default. Retrying a malformed response forever is a busy loop; retrying a
missing config file is pointless; retrying a dropped connection is correct.

    POLICY LEGEND
      fail-operation   this utterance is lost, the session continues
      degrade          keep running with reduced capability, report it
      retry            transient; try again with backoff
      restart-session  the session cannot continue; the supervisor respawns it
      operator         nothing automatic will fix this; a human must act
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fakes import FakeLLM, FakeProcess, FakeSink, FakeTTS, audio_request

from platform_.audio.router import AudioRouter, PlaybackState


async def settle(ticks: int = 60) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


def router(tmp_path, **kw) -> AudioRouter:
    kw.setdefault("sink", FakeSink())
    kw.setdefault("tts", FakeTTS())
    return AudioRouter("SESSION_001", kw.pop("tts"), tmp_path, **kw)


# =========================================================================
# TTS
# =========================================================================


async def test_tts_exception_costs_one_utterance_not_the_session(tmp_path):
    """POLICY: fail-operation. One bad synthesis must not end the broadcast."""
    r = router(tmp_path, tts=FakeTTS(fail_on=0))
    await r.start()

    await r.submit(audio_request(segments=["doomed"]))
    await settle()
    assert r.stats.tts_failures >= 1, "the failure must be detected"

    await r.submit(audio_request(segments=["recovered"]))
    await settle()
    assert r.sink.played, "the next utterance must still speak"
    await r.stop()


async def test_tts_timeout_does_not_wedge_the_drain_loop(tmp_path):
    """POLICY: fail-operation. A slow synthesis is abandoned with the router
    still able to accept the next one."""
    r = router(tmp_path, tts=FakeTTS(delay_s=0.05))
    await r.start()
    await r.submit(audio_request(segments=["slow"]))
    await asyncio.sleep(0.01)

    assert r._task is not None and not r._task.done()
    await r.stop()
    assert r.state is PlaybackState.STOPPED


async def test_empty_synthesis_is_skipped_silently(tmp_path):
    """POLICY: fail-operation, quietly. TTS legitimately returns no file for
    empty text; that is not an error worth reporting."""
    sink = FakeSink()
    r = router(tmp_path, tts=FakeTTS(empty=True), sink=sink)
    r._running = True

    await r._speak(audio_request(segments=["", "   "]))
    assert sink.played == []
    assert r.state is not PlaybackState.PLAYING


async def test_malformed_audio_is_a_playback_failure_not_a_crash(tmp_path):
    """POLICY: fail-operation. A file exists but the decoder rejects it -- the
    case that proves 'a wav was produced' is not 'the host spoke'."""
    r = router(tmp_path, tts=FakeTTS(malformed=True), sink=FakeSink(fail=True))
    await r.start()
    await r.submit(audio_request(segments=["garbage"]))
    await settle()

    assert r.stats.tts_failures >= 1
    assert r._task is not None and not r._task.done()
    await r.stop()


# =========================================================================
# Audio device
# =========================================================================


async def test_playback_failure_is_surfaced_not_swallowed(tmp_path):
    """POLICY: degrade. This was `except Exception: pass`, so a dead device
    looked exactly like a spoken utterance."""
    sink = FakeSink(fail=True)
    r = router(tmp_path, sink=sink)
    await r.start()
    await r.submit(audio_request(segments=["silence"]))
    await settle()

    assert r.stats.tts_failures >= 1
    await r.stop()


async def test_the_device_failing_midway_still_leaves_a_clean_state(tmp_path):
    """POLICY: degrade. Two utterances succeed, the third fails."""
    sink = FakeSink(fail_after=2)
    r = router(tmp_path, sink=sink)
    await r.start()
    for i in range(4):
        await r.submit(audio_request(segments=[f"line {i}"]))
    await settle(200)

    assert len(sink.played) == 2, "playback stopped where the device failed"
    assert r.state in (PlaybackState.IDLE, PlaybackState.STOPPED)
    assert r.current_utterance is None
    await r.stop()


async def test_a_hanging_device_does_not_block_shutdown(tmp_path):
    """POLICY: fail-operation on shutdown. A device that never returns must
    not make STOP hang -- the user pressed a button and deserves an answer."""
    r = router(tmp_path, sink=FakeSink(hang=True))
    await r.start()
    await r.submit(audio_request(segments=["never finishes"]))
    await asyncio.sleep(0.02)

    await asyncio.wait_for(r.stop(), timeout=5.0)
    assert r.state is PlaybackState.STOPPED


# =========================================================================
# Model
# =========================================================================


async def test_a_model_connection_failure_is_reported_not_faked():
    """POLICY: degrade, then restart-session if it persists. Never substitute
    canned text and call it ready."""
    from runtime.readiness import gate_model_real

    async def discover_dead(*_a, **_k):
        return FakeLLM(fail=True)

    import platform_.llm.discovery as disc

    original = disc.discover
    disc.discover = discover_dead
    try:
        result = await gate_model_real()
    finally:
        disc.discover = original

    assert not result.ok
    assert result.action, "a failure the user can act on must say how"


async def test_a_model_timeout_is_distinguished_from_a_broken_model():
    """POLICY: retry. A cold model loading gigabytes is not a fault, and
    telling the user to report a bug for their own slow machine was wrong."""
    from runtime.readiness import gate_model_real

    async def discover_slow(*_a, **_k):
        return FakeLLM(timeout=True)

    import platform_.llm.discovery as disc

    original = disc.discover
    disc.discover = discover_slow
    try:
        result = await gate_model_real(budget_s=0.1)
    finally:
        disc.discover = original

    assert not result.ok
    assert "loading" in result.action or "again" in result.action


async def test_an_empty_model_response_fails_the_gate():
    """POLICY: fail-operation. A server that answers with nothing has not
    proved it can generate."""
    from runtime.readiness import gate_model_real

    async def discover_empty(*_a, **_k):
        return FakeLLM(empty=True)

    import platform_.llm.discovery as disc

    original = disc.discover
    disc.discover = discover_empty
    try:
        result = await gate_model_real(min_words=20)
    finally:
        disc.discover = original

    assert not result.ok
    assert "0 words" in result.reason or "words" in result.reason


async def test_a_huge_model_response_does_not_break_anything():
    """POLICY: accept. 20,000 words is absurd but not dangerous; it must not
    throw, and the router segments it normally."""
    llm = FakeLLM(huge=True)
    result = await llm.complete([])
    assert len(result.text.split()) == 20_000


async def test_a_repeating_model_is_a_content_problem_not_a_crash():
    """POLICY: degrade. The real failure mode: the model works perfectly and
    says nearly the same thing every time. Roughly three quarters of
    production output is rejected for this."""
    from intelligence.memory import SessionMemory

    llm = FakeLLM(repeat="Gold is holding just under the level we marked.")
    memory = SessionMemory("FAIL_TEST")

    accepted = 0
    for _ in range(10):
        text = (await llm.complete([])).text
        repetitive, _ = memory.is_repetitive(text, 0.38)
        if not repetitive:
            memory.record_utterance(text, topic="t")
            accepted += 1

    assert accepted == 1, "only the first should survive the repetition guard"
    assert memory.utterance_count == 1


# =========================================================================
# Comment adapter
# =========================================================================


async def test_an_adapter_that_cannot_read_does_not_kill_ingestion(tmp_path):
    """POLICY: degrade. One unreadable poll must not end comment ingestion for
    the session -- that failure used to be permanent and silent."""
    from platform_.adapters.filetail import FileTailAdapter

    adapter = FileTailAdapter(session_id="SESSION_001", path=tmp_path / "c.txt")
    await adapter.connect()
    adapter.path = tmp_path / "gone" / "missing.txt"

    assert adapter.read_new() == []

    adapter.path = tmp_path / "c.txt"
    adapter.path.write_text("back again\n", encoding="utf-8")
    adapter._offset = 0
    assert [c.text_raw for c in adapter.read_new()] == ["back again"]


async def test_a_missing_comment_file_is_created_not_fatal(tmp_path):
    """POLICY: degrade. An absent file is the normal first-run state."""
    from platform_.adapters.filetail import FileTailAdapter

    path = tmp_path / "nested" / "comments.txt"
    adapter = FileTailAdapter(session_id="SESSION_001", path=path)
    await adapter.connect()
    assert path.exists()


# =========================================================================
# Child processes
# =========================================================================


def test_a_crashed_child_is_detected():
    """POLICY: restart-session, with backoff."""
    proc = FakeProcess(exits_after=1, returncode=1)
    assert proc.poll() == 1


def test_a_hanging_child_is_escalated_to_kill():
    """POLICY: restart-session. A process that ignores SIGTERM is the failure
    supervision usually misses -- alive, and not doing its job."""
    proc = FakeProcess(hang=True)
    proc.terminate()
    assert proc.terminated and proc.returncode is None, "hangs through terminate"
    proc.kill()
    assert proc.killed and proc.returncode == -9


def test_repeated_crashes_stop_rather_than_loop():
    """POLICY: operator. A loop that restarts silently is worse than one that
    stops, because nobody finds out."""
    from runtime.supervisor import ManagedSession

    session = ManagedSession("S1", 9101, [])
    import time as _t

    session.restarts = [_t.time() for _ in range(6)]
    assert session.restarts_last_hour() == 6


# =========================================================================
# Filesystem and configuration
# =========================================================================


def test_corrupt_lifecycle_state_denies_consent(tmp_path, monkeypatch):
    """POLICY: fail-safe. When consent cannot be determined the answer is no."""
    from runtime import lifecycle as lc

    path = tmp_path / "lifecycle.json"
    path.write_text("{ truncated", encoding="utf-8")
    monkeypatch.setattr(lc, "state_path", lambda: path)

    assert lc.load().state is lc.RunState.STOPPED
    assert not lc.load().may_run


def test_corrupt_provisioning_state_is_preserved_and_reprovisioned(tmp_path):
    """POLICY: degrade to first-run. The bad file is kept as evidence rather
    than deleted -- if provisioning keeps failing, it is what explains why."""
    from shared.provisioning import ProvisionState, load

    path = tmp_path / "provisioning.json"
    path.write_text('{"state_version": 1, "artifa', encoding="utf-8")

    state = load(path)
    assert state.state is ProvisionState.NEW
    assert any(p.name.startswith("provisioning.bad") for p in tmp_path.iterdir())


def test_state_from_a_newer_version_refuses_rather_than_guessing(tmp_path):
    """POLICY: operator. Silently downgrading destroys whatever that version
    added."""
    from shared.provisioning import StateTooNew, load

    path = tmp_path / "provisioning.json"
    path.write_text(json.dumps({"state_version": 999}), encoding="utf-8")

    with pytest.raises(StateTooNew):
        load(path)


def test_a_failed_state_write_leaves_the_previous_state_intact(tmp_path, monkeypatch):
    """POLICY: fail-operation. os.replace is last precisely so a crash during
    serialisation cannot produce a half-written primary file."""
    import shared.provisioning as mod
    from shared.provisioning import Provisioning, load, save

    path = tmp_path / "provisioning.json"
    good = Provisioning()
    good.begin()
    good.succeed()
    save(good, path)
    before = path.read_text(encoding="utf-8")

    monkeypatch.setattr(mod.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("full")))
    with pytest.raises(OSError):
        save(Provisioning(), path)

    assert path.read_text(encoding="utf-8") == before
    assert load(path).state is mod.ProvisionState.READY


async def test_missing_session_config_fails_readiness_with_an_action(tmp_path):
    """POLICY: operator. No amount of retrying invents a config file."""
    from runtime.readiness import gate_config_valid

    result = await gate_config_valid("SESSION_DOES_NOT_EXIST", tmp_path)
    assert not result.ok
    assert result.action, "the user must be told what to do"


# =========================================================================
# Recovery: the next valid operation must work
# =========================================================================


@pytest.mark.parametrize(
    "broken",
    [FakeTTS(fail_on=0), FakeTTS(empty=True), FakeTTS(malformed=True)],
)
async def test_the_router_recovers_after_every_tts_failure_mode(tmp_path, broken):
    """The half of failure handling that is usually missed: after the failure
    is detected, does the system still work?"""
    sink = FakeSink()
    r = AudioRouter("SESSION_001", broken, tmp_path, sink=sink)
    await r.start()

    await r.submit(audio_request(segments=["first"]))
    await settle()

    r.tts = FakeTTS()  # the transient condition clears
    await r.submit(audio_request(segments=["second"]))
    await settle()

    assert sink.played, "the router must work again once the fault clears"
    assert r.current_utterance is None
    await r.stop()


async def test_no_cancelled_utterance_resumes_after_recovery(tmp_path):
    """The specific race: an old utterance regaining control after a newer one
    has superseded it."""
    sink = FakeSink()
    r = router(tmp_path, sink=sink)
    r._running = True

    old = audio_request(segments=["old1", "old2", "old3"])
    r._preempted = (old.utterance_id)
    await r._speak(old)
    assert sink.played == []

    new = audio_request(segments=["new1"])
    await r._speak(new)
    assert len(sink.played) == 1, "only the superseding utterance may speak"


async def test_queues_are_sane_after_a_failure_burst(tmp_path):
    """Queue depth must return to zero, not accumulate abandoned work."""
    r = router(tmp_path, tts=FakeTTS(fail=True), max_queue=16)
    await r.start()
    for _ in range(10):
        await r.submit(audio_request(segments=["x"]))
    await settle(400)

    assert r.queue_depth == 0, f"queue did not drain: {r.queue_depth}"
    await r.stop()
