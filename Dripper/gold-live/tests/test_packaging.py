"""The shape of the downloadable distribution.

The ZIP is the only thing most users ever see, and it is assembled by a script
that runs on one machine, by hand, at release time. That is the least-observed
code in the project, so the invariants it has to hold are asserted here rather
than discovered by whoever extracts the archive.

The bug that motivated this file: `stage()` copied each executable to the top
of the folder and left the PyInstaller runtime in `app/`. Both executables were
present, so the packaging check passed. Neither of the ones at the top could
start -- PyInstaller resolves its Python DLL relative to the executable, so an
exe without `_internal/` beside it dies before reaching main(). The README told
people to double-click precisely those copies.
"""

from __future__ import annotations

import pytest

from build_tools import make_zip


@pytest.fixture
def built(tmp_path, monkeypatch):
    """A stand-in for `dist/GoldLive` as PyInstaller leaves it."""
    root = tmp_path / "root"
    built = tmp_path / "dist" / "GoldLive"
    stage = tmp_path / "dist" / "_zipstage" / "GoldLive"

    (built / "_internal").mkdir(parents=True)
    (built / "_internal" / "python314.dll").write_bytes(b"runtime")
    (built / "voices" / "en_US").mkdir(parents=True)
    (built / "voices" / "en_US" / "voice.onnx").write_bytes(b"voice")
    (built / "GoldLive.exe").write_bytes(b"panel")
    (built / "GoldLive Setup.exe").write_bytes(b"setup")
    (built / "THIRD_PARTY_NOTICES.md").write_text("notices", encoding="utf-8")
    # Auto-start machinery that must never be shipped.
    (built / "install_windows.ps1").write_text("Register-ScheduledTask", encoding="utf-8")

    (root / "docs").mkdir(parents=True)
    (root / "docs" / "INSTALL_WINDOWS.md").write_text("install", encoding="utf-8")
    (root / "docs" / "USER_GUIDE.md").write_text("guide", encoding="utf-8")

    monkeypatch.setattr(make_zip, "ROOT", root)
    monkeypatch.setattr(make_zip, "DIST", tmp_path / "dist")
    monkeypatch.setattr(make_zip, "BUILT", built)
    monkeypatch.setattr(make_zip, "STAGE", stage)
    return stage


def test_every_executable_has_the_runtime_beside_it(built):
    """The invariant a presence check cannot express."""
    staged = make_zip.stage()

    exes = list(staged.rglob("*.exe"))
    assert exes, "the distribution must contain executables"
    for exe in exes:
        assert (exe.parent / "_internal").is_dir(), (
            f"{exe.relative_to(staged)} has no _internal/ beside it; "
            "PyInstaller cannot load its Python DLL and it will not start"
        )


def test_the_entry_points_are_at_the_top_where_the_readme_says(built):
    staged = make_zip.stage()

    assert (staged / "GoldLive.exe").is_file()
    assert (staged / "GoldLive Setup.exe").is_file()
    assert (staged / "README.txt").is_file()


def test_the_runtime_is_not_duplicated(built):
    """Two copies of a 100 MB runtime is a slow download and a stale-copy bug
    waiting to happen -- the user updates one and runs the other."""
    staged = make_zip.stage()

    assert len(list(staged.rglob("_internal"))) == 1
    assert len(list(staged.rglob("GoldLive.exe"))) == 1


def test_auto_start_machinery_is_refused(built):
    """GoldLive runs only when a person asks. Shipping the service installer
    would make that a promise the archive itself breaks."""
    staged = make_zip.stage()

    shipped = [p.name for p in staged.rglob("*")]
    for forbidden in make_zip.FORBIDDEN:
        assert forbidden not in shipped


def test_the_voices_and_docs_travel_with_the_build(built):
    staged = make_zip.stage()

    assert (staged / "voices" / "en_US" / "voice.onnx").is_file()
    assert (staged / "docs" / "INSTALL_WINDOWS.md").is_file()
    assert (staged / "THIRD_PARTY_NOTICES.md").is_file()
