"""Provisioning state.

The whole point of this file on disk is that a launch can trust it. So the
tests are mostly about *not* trusting it: truncated writes, files from the
future, artifacts that vanished. A state file that reads back cleanly is the
easy case and the least interesting one.
"""

from __future__ import annotations

import json

import pytest

from shared.provisioning import (
    Artifact,
    ProvisionState,
    Provisioning,
    StateTooNew,
    load,
    save,
    sha256_file,
)


def state_file(tmp_path):
    return tmp_path / "provisioning.json"


# -- round trip ------------------------------------------------------------


def test_a_new_machine_reports_new(tmp_path):
    state = load(state_file(tmp_path))
    assert state.state is ProvisionState.NEW
    assert not state.is_provisioned()


def test_round_trip_preserves_everything(tmp_path):
    path = state_file(tmp_path)
    original = Provisioning()
    original.begin()
    original.hardware = {"cpu_model": "test cpu", "ram_total_gb": 16.0}
    original.profile_hash = "sha256:abc"
    original.author_salt = "deadbeef"
    original.selection = {"model_id": "llama3.2:3b", "tier": "baseline"}
    original.record_artifact(
        Artifact(kind="model", artifact_id="model:x", digest="sha256:1", bytes=99)
    )
    original.succeed()
    save(original, path)

    back = load(path)
    assert back.state is ProvisionState.READY
    assert back.hardware["cpu_model"] == "test cpu"
    assert back.profile_hash == "sha256:abc"
    assert back.author_salt == "deadbeef"
    assert back.selection["model_id"] == "llama3.2:3b"
    assert back.artifacts["model:x"].digest == "sha256:1"
    assert back.is_provisioned()


def test_failure_records_the_stage_and_reason(tmp_path):
    path = state_file(tmp_path)
    state = Provisioning()
    state.begin()
    state.fail("model", "Ollama is not running")
    save(state, path)

    back = load(path)
    assert back.state is ProvisionState.FAILED
    assert back.failed_stage == "model"
    assert "Ollama" in back.failed_reason
    assert not back.is_provisioned()


def test_degraded_is_never_persisted(tmp_path):
    """Whether a session can run *now* depends on the network and the model
    server, neither of which this file knows about. Only installation state
    belongs here."""
    assert {s.value for s in ProvisionState} == {
        "new", "provisioning", "ready", "failed"
    }


# -- corruption ------------------------------------------------------------


def test_truncated_json_is_moved_aside_and_treated_as_new(tmp_path):
    path = state_file(tmp_path)
    path.write_text('{"state_version": 1, "goldlive_ver', encoding="utf-8")

    state = load(path)
    assert state.state is ProvisionState.NEW
    assert any(p.name.startswith("provisioning.bad") for p in tmp_path.iterdir()), (
        "the corrupt file must be preserved as evidence, not deleted"
    )


def test_json_that_is_not_an_object_is_rejected(tmp_path):
    path = state_file(tmp_path)
    path.write_text('["not", "a", "state"]', encoding="utf-8")
    assert load(path).state is ProvisionState.NEW


def test_a_newer_state_version_refuses_rather_than_guessing(tmp_path):
    """Silently downgrading a file from a future version destroys whatever
    that version added. Refusing is the only safe answer."""
    path = state_file(tmp_path)
    path.write_text(json.dumps({"state_version": 99}), encoding="utf-8")

    with pytest.raises(StateTooNew):
        load(path)


# -- atomicity -------------------------------------------------------------


def test_no_temp_files_are_left_behind(tmp_path):
    path = state_file(tmp_path)
    for _ in range(3):
        save(Provisioning(), path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".provisioning-")]
    assert leftovers == []


def test_a_failed_write_leaves_the_previous_state_intact(tmp_path, monkeypatch):
    """os.replace is the last step precisely so a crash during serialisation
    cannot produce a half-written primary file."""
    path = state_file(tmp_path)
    good = Provisioning()
    good.begin()
    good.succeed()
    save(good, path)
    before = path.read_text(encoding="utf-8")

    import shared.provisioning as mod

    def boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(mod.os, "replace", boom)
    with pytest.raises(OSError):
        save(Provisioning(), path)

    assert path.read_text(encoding="utf-8") == before
    assert load(path).state is ProvisionState.READY


# -- artifacts -------------------------------------------------------------


def test_a_present_artifact_passes_the_cheap_check(tmp_path):
    blob = tmp_path / "voice.onnx"
    blob.write_bytes(b"x" * 128)
    art = Artifact(kind="voice", artifact_id="voice:a", path=str(blob), bytes=128)
    assert art.looks_present()


def test_a_deleted_artifact_is_detected(tmp_path):
    blob = tmp_path / "voice.onnx"
    blob.write_bytes(b"x" * 128)
    art = Artifact(kind="voice", artifact_id="voice:a", path=str(blob), bytes=128)
    blob.unlink()
    assert not art.looks_present()
    ok, reason = art.verify()
    assert not ok and "missing" in reason


def test_a_truncated_artifact_is_detected_by_size(tmp_path):
    blob = tmp_path / "voice.onnx"
    blob.write_bytes(b"x" * 128)
    art = Artifact(kind="voice", artifact_id="voice:a", path=str(blob), bytes=128)
    blob.write_bytes(b"x" * 64)
    assert not art.looks_present()


def test_a_corrupted_artifact_is_detected_by_checksum(tmp_path):
    """Same size, different content -- the case the cheap check cannot see."""
    blob = tmp_path / "voice.onnx"
    blob.write_bytes(b"a" * 128)
    art = Artifact(
        kind="voice", artifact_id="voice:a", path=str(blob),
        bytes=128, sha256=sha256_file(blob),
    )
    assert art.verify()[0]

    blob.write_bytes(b"b" * 128)
    assert art.looks_present(), "size is unchanged, so the cheap check passes"
    ok, reason = art.verify()
    assert not ok and "sha256" in reason


def test_server_managed_artifacts_have_no_path_to_check(tmp_path):
    art = Artifact(kind="model", artifact_id="model:x", path=None, digest="sha256:1")
    assert art.looks_present()
    assert art.verify()[0]


def test_missing_artifacts_are_listed(tmp_path):
    blob = tmp_path / "gone.onnx"
    blob.write_bytes(b"x")
    state = Provisioning()
    state.record_artifact(
        Artifact(kind="voice", artifact_id="voice:gone", path=str(blob), bytes=1)
    )
    assert state.missing_artifacts() == []
    blob.unlink()
    assert [a.artifact_id for a in state.missing_artifacts()] == ["voice:gone"]


# -- the fast-path decision ------------------------------------------------


def test_needs_provisioning_on_a_new_machine(tmp_path, monkeypatch):
    from runtime import provision as prov

    needed, why = prov.needs_provisioning(Provisioning())
    assert needed and "not provisioned" in why


def test_provisioned_machine_takes_the_fast_path(tmp_path):
    from runtime.provision import needs_provisioning
    from shared.version import app_version

    state = Provisioning()
    state.begin()
    state.goldlive_version = app_version()
    state.succeed()

    needed, why = needs_provisioning(state)
    assert not needed and why == "already provisioned"


def test_an_updated_application_needs_revalidation(tmp_path):
    from runtime.provision import needs_provisioning

    state = Provisioning()
    state.begin()
    state.succeed()
    state.goldlive_version = "0.0.1-old"

    needed, why = needs_provisioning(state)
    assert needed and "updated" in why


def test_a_deleted_artifact_forces_repair(tmp_path):
    from runtime.provision import needs_provisioning
    from shared.version import app_version

    blob = tmp_path / "voice.onnx"
    blob.write_bytes(b"x" * 10)
    state = Provisioning()
    state.begin()
    state.record_artifact(
        Artifact(kind="voice", artifact_id="voice:a", path=str(blob), bytes=10)
    )
    state.succeed()
    state.goldlive_version = app_version()
    assert not needs_provisioning(state)[0]

    blob.unlink()
    needed, why = needs_provisioning(state)
    assert needed and "voice:a" in why
