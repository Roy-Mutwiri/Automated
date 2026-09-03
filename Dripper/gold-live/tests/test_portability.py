"""Capability probing, session spawning, model selection and the bundle.

These cover the things that were wrong specifically *because* the frozen build
behaves differently from a source checkout -- which is why the spawn tests
assert on argv rather than starting a process. The bug was in the command, so
the command is what needs to be assertable.
"""

from __future__ import annotations

import sys

import pytest

from runtime.provision import ProvisionError, select_model
from runtime.selftest import BUNDLED_IMPORTS, missing_imports
from runtime.supervisor import session_command
from shared.capability import HardwareProfile, probe


# -- session_command -------------------------------------------------------


def test_source_mode_runs_the_module():
    cmd = session_command("SESSION_001", 9101, ["--market", "gold"])
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "runtime.live"]
    assert "--session" in cmd and "SESSION_001" in cmd
    assert cmd[-2:] == ["--market", "gold"]


def test_frozen_mode_uses_the_exe_subcommand(monkeypatch):
    """The exe is not a Python interpreter: `GoldLive.exe -m runtime.live`
    reached the CLI dispatcher, which answered "Unknown command: -m", so every
    supervised session died instantly and the restart logic turned that into a
    spawn loop."""
    import runtime.supervisor as sup

    monkeypatch.setattr(sup, "is_frozen", lambda: True)
    cmd = session_command("SESSION_001", 9101, [])
    assert "-m" not in cmd, "the frozen build has no -m"
    assert cmd[1] == "run"
    assert cmd[2:4] == ["--session", "SESSION_001"]


def test_health_port_is_passed_through_in_both_modes(monkeypatch):
    import runtime.supervisor as sup

    for frozen in (False, True):
        monkeypatch.setattr(sup, "is_frozen", lambda value=frozen: value)
        cmd = session_command("S1", 9207, [])
        assert cmd[cmd.index("--health-port") + 1] == "9207"


def test_the_spawned_command_is_a_pure_function():
    """No side effects, so the thing that broke can be asserted without
    starting a process -- which is exactly why it went unnoticed."""
    a = session_command("S1", 9101, ["--tts", "piper"])
    b = session_command("S1", 9101, ["--tts", "piper"])
    assert a == b


# -- capability profile ----------------------------------------------------


def test_probe_returns_a_usable_profile():
    p = probe()
    assert p.cpu_logical_cores > 0
    assert p.ram_total_gb > 0
    assert p.data_root
    assert p.disk_data_total_gb > 0


def test_probe_never_raises_even_when_every_source_fails(monkeypatch):
    """A machine that reports "unknown" still provisions; a probe that throws
    during startup does not."""
    import shared.capability as cap

    for name in ("_probe_os", "_probe_cpu", "_probe_ram", "_probe_disk",
                 "_probe_audio", "_probe_network", "_probe_model_server"):
        def boom(_p, _n=name):
            raise RuntimeError(f"{_n} exploded")

        monkeypatch.setattr(cap, name, boom)

    p = cap.probe()
    assert isinstance(p, HardwareProfile)


def test_profile_hash_is_stable_across_calls():
    assert probe().profile_hash() == probe().profile_hash()


def test_profile_hash_ignores_volatile_fields():
    """Free disk and available RAM change between any two launches. Including
    them would mean the hash never matches and the fast path never fires."""
    a = HardwareProfile(cpu_model="x", ram_total_gb=16.0, ram_available_gb=8.0,
                        disk_data_free_gb=100.0, market_feed_latency_ms=50)
    b = HardwareProfile(cpu_model="x", ram_total_gb=16.0, ram_available_gb=1.2,
                        disk_data_free_gb=3.0, market_feed_latency_ms=900)
    assert a.profile_hash() == b.profile_hash()


def test_profile_hash_ignores_whether_the_gpu_probe_ran():
    """doctor probes the GPU and the launch path does not. If the hash covered
    it, every doctor run would look like a hardware change."""
    a = HardwareProfile(cpu_model="x", ram_total_gb=16.0, gpu_model="unknown")
    b = HardwareProfile(cpu_model="x", ram_total_gb=16.0, gpu_model="Radeon RX 580")
    assert a.profile_hash() == b.profile_hash()


def test_profile_hash_changes_when_real_hardware_changes():
    a = HardwareProfile(cpu_model="ryzen 5", ram_total_gb=16.0, cpu_logical_cores=12)
    b = HardwareProfile(cpu_model="ryzen 9", ram_total_gb=16.0, cpu_logical_cores=12)
    c = HardwareProfile(cpu_model="ryzen 5", ram_total_gb=64.0, cpu_logical_cores=12)
    assert len({a.profile_hash(), b.profile_hash(), c.profile_hash()}) == 3


def test_ram_is_rounded_so_a_few_megabytes_do_not_change_the_hash():
    a = HardwareProfile(cpu_model="x", ram_total_gb=17.12)
    b = HardwareProfile(cpu_model="x", ram_total_gb=17.09)
    assert a.profile_hash() == b.profile_hash()


def test_a_missing_cable_is_part_of_the_profile():
    a = HardwareProfile(cpu_model="x", virtual_cable=None)
    b = HardwareProfile(cpu_model="x", virtual_cable="CABLE Input")
    assert a.profile_hash() != b.profile_hash()


def test_describe_is_plain_english():
    text = probe().describe()
    assert "cores" in text and "RAM" in text
    assert text.isascii(), "the Windows console is cp1252; non-ASCII renders as ?"


# -- model selection -------------------------------------------------------


CATALOGUE = {
    "defaults": {"ram_fraction": 0.6, "disk_headroom": 1.3},
    "tiers": [
        {"id": "tiny", "tier": "minimum", "bytes": 1_300_000_000, "ram_gb": 3},
        {"id": "small", "tier": "baseline", "bytes": 2_020_000_000, "ram_gb": 6},
        {"id": "medium", "tier": "good", "bytes": 4_700_000_000, "ram_gb": 12},
        {"id": "large", "tier": "best", "bytes": 9_000_000_000, "ram_gb": 24},
    ],
}


def profile_with(ram_total, ram_free, disk_free):
    return HardwareProfile(
        cpu_model="test", ram_total_gb=ram_total,
        ram_available_gb=ram_free, disk_data_free_gb=disk_free,
    )


def test_a_big_machine_gets_a_bigger_model():
    choice = select_model(profile_with(64, 48, 500), CATALOGUE)
    assert choice.model_id == "large"


def test_a_modest_machine_gets_the_baseline():
    choice = select_model(profile_with(16, 12, 100), CATALOGUE)
    assert choice.model_id in ("small", "medium")


def test_a_small_machine_gets_the_minimum():
    choice = select_model(profile_with(8, 6, 50), CATALOGUE)
    assert choice.model_id in ("tiny", "small")


def test_no_model_is_hardcoded_for_every_machine():
    """The requirement was explicit: selection must depend on the machine."""
    big = select_model(profile_with(64, 48, 500), CATALOGUE).model_id
    small = select_model(profile_with(8, 5, 40), CATALOGUE).model_id
    assert big != small


def test_a_full_disk_blocks_a_model_that_would_otherwise_fit():
    """Plenty of RAM, no room to put the file.

    2 GB free admits tiny (1.3 GB x 1.3 headroom = 1.7 GB) and excludes small
    (2.6 GB), even though 48 GB of RAM would run either comfortably.
    """
    assert select_model(profile_with(64, 48, 2), CATALOGUE).model_id == "tiny"
    # The same machine with room to spare is not held back.
    assert select_model(profile_with(64, 48, 500), CATALOGUE).model_id == "large"


def test_a_machine_too_small_for_anything_fails_with_an_action():
    with pytest.raises(ProvisionError) as exc:
        select_model(profile_with(2, 0.5, 1), CATALOGUE)
    assert "cannot run even the smallest" in exc.value.reason
    assert exc.value.action


def test_a_busy_machine_is_not_permanently_downgraded():
    """Available RAM right now is honest but transient; a machine that happens
    to be busy during provisioning should not be stuck on the tiny model."""
    busy = select_model(profile_with(64, 2, 500), CATALOGUE)
    assert busy.model_id != "tiny"


def test_the_shipped_catalogue_parses_and_is_ordered():
    from runtime.provision import load_catalogue

    cat = load_catalogue()
    tiers = cat["tiers"]
    assert [t["tier"] for t in tiers] == ["minimum", "baseline", "good", "best"]
    sizes = [t["ram_gb"] for t in tiers]
    assert sizes == sorted(sizes), "selection takes the last affordable entry"


# -- the bundle ------------------------------------------------------------


def test_every_lazily_imported_package_is_in_the_manifest():
    """websockets and piper are imported inside functions, so PyInstaller's
    static analysis never saw them and the shipped exe could neither fetch a
    price nor speak. They must be declared explicitly or the bug returns."""
    names = {name for name, _why in BUNDLED_IMPORTS}
    assert {"websockets", "piper", "httpx", "sounddevice", "soundfile"} <= names


def test_the_manifest_matches_the_pyinstaller_spec():
    """If these drift, the build ships something the selftest does not check."""
    from pathlib import Path

    spec = Path(__file__).resolve().parent.parent / "build_tools" / "GoldLive.spec"
    text = spec.read_text(encoding="utf-8")
    block = text.split("BUNDLED = [", 1)[1].split("]", 1)[0]
    in_spec = {line.split('"')[1] for line in block.splitlines() if '"' in line}
    in_selftest = {name for name, _ in BUNDLED_IMPORTS}
    assert in_spec <= in_selftest, f"in the spec but unchecked: {in_spec - in_selftest}"


def test_this_environment_has_everything_the_build_claims():
    assert missing_imports() == []


def test_an_already_installed_model_is_preferred_over_a_download():
    """Pulling a second model when an affordable one is already present is the
    wasted work this whole system exists to prevent."""
    profile = profile_with(16, 12, 100)
    fresh = select_model(profile, CATALOGUE).model_id
    reused = select_model(profile, CATALOGUE, installed=["tiny"]).model_id
    assert reused == "tiny" and reused != fresh
    assert "already installed" in select_model(
        profile, CATALOGUE, installed=["tiny"]
    ).reason


def test_an_installed_model_that_does_not_fit_is_not_reused():
    """Reuse is a preference among affordable options, not an override."""
    choice = select_model(profile_with(8, 6, 50), CATALOGUE, installed=["large"])
    assert choice.model_id != "large"
