"""Assemble the downloadable Windows distribution.

    python -m build_tools.make_zip

Produces `dist/GoldLive-Windows-x64.zip` laid out so that extracting it gives
a folder with one obvious thing to run:

    GoldLive/
        GoldLive Setup.exe      <- run this first
        GoldLive.exe            <- control panel: START / STOP
        README.txt
        app/                    <- the packaged runtime

Deliberately absent: the language model and the Piper voices. They are
gigabytes and their licences differ per artifact, so they are provisioned at
setup time by the existing download machinery rather than committed to a git
repository or shipped in a ZIP.

Also deliberately absent: anything that could make GoldLive start by itself.
No service installer, no scheduled task, no Startup shortcut. `ops/` is
excluded from the distribution for exactly that reason.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUILT = DIST / "GoldLive"
STAGE = DIST / "_zipstage" / "GoldLive"

#: Files that must never reach the distribution because they arrange for
#: GoldLive to start without a person asking.
FORBIDDEN = ("install_windows.ps1", "goldlive-session@.service",
             "goldlive-dashboard.service")

README = """GOLDLIVE
========

WHAT THIS IS
    An AI host that talks about the live gold market. It speaks out loud
    through your PC's audio, so you decide when it runs.

INSTALL
    1. Run  "GoldLive Setup.exe"
       It checks your PC, downloads the AI model and the voice, and tests
       that everything works. This takes about 15 minutes, mostly
       downloading, and needs roughly 3 GB of free disk space.

    2. When it says READY TO START, close it.

START
    Open  GoldLive.exe  and press START.

STOP
    Press STOP.

IMPORTANT
    GoldLive does NOT start by itself.
    It does NOT start when Windows starts.
    It only runs while you have started it, and STOP always stops it.

WHAT SETUP CANNOT INSTALL FOR YOU
    Ollama          Runs the AI model. Setup will tell you if it is missing.
                    Get it from https://ollama.com
    VB-CABLE        Only needed to send GoldLive's audio into streaming
                    software. Get it from https://vb-audio.com/Cable
                    Without it GoldLive still works, but plays to your
                    speakers instead of to a broadcast.

WHERE YOUR FILES GO
    Settings, logs, the model and the voices live in your user folder,
    not in this one, so you can replace this folder when updating.
    Run "GoldLive.exe paths" to see exactly where.

NOT YET PROVEN
    Broadcasting to TikTok LIVE has not been verified end to end.
    See docs/USER_GUIDE.md for what is and is not tested.
"""


def stage() -> Path:
    if not BUILT.exists():
        sys.exit(f"nothing to package: {BUILT} does not exist.\n"
                 "Build first:  python -m PyInstaller build_tools/GoldLive.spec --noconfirm")

    if STAGE.parent.exists():
        shutil.rmtree(STAGE.parent)
    (STAGE / "app").mkdir(parents=True)

    for item in BUILT.iterdir():
        if item.name in FORBIDDEN:
            print(f"  refusing to ship {item.name}: it can start GoldLive without consent")
            continue
        target = STAGE / item.name if item.suffix == ".exe" else STAGE / "app" / item.name
        if item.is_dir():
            shutil.copytree(item, STAGE / "app" / item.name)
        else:
            shutil.copy2(item, target)

    # The executables sit at the top so the folder has an obvious entry point,
    # but PyInstaller needs _internal beside them, so a copy stays in app/ too.
    for exe in ("GoldLive.exe", "GoldLive Setup.exe"):
        source = BUILT / exe
        if source.exists():
            shutil.copy2(source, STAGE / "app" / exe)

    (STAGE / "README.txt").write_text(README, encoding="utf-8")
    for doc in ("docs/INSTALL_WINDOWS.md", "docs/USER_GUIDE.md"):
        source = ROOT / doc
        if source.exists():
            (STAGE / "docs").mkdir(exist_ok=True)
            shutil.copy2(source, STAGE / "docs" / source.name)
    return STAGE


def make_zip() -> Path:
    from shared.version import app_version

    staged = stage()
    out = DIST / f"GoldLive-Windows-x64-v{app_version()}.zip"
    if out.exists():
        out.unlink()

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(staged.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staged.parent))

    size_mb = out.stat().st_size / 1e6
    print(f"\n  {out.name}  ({size_mb:.0f} MB)")
    print(f"  {out}\n")
    return out


def main() -> int:
    sys.path.insert(0, str(ROOT))
    out = make_zip()

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    top = sorted({n.split("/")[1] for n in names if n.count("/") >= 1 and n.split("/")[1]})
    print("  extracted folder contains:")
    for name in top:
        print(f"    {name}")

    missing = [n for n in ("GoldLive/GoldLive.exe", "GoldLive/GoldLive Setup.exe",
                           "GoldLive/README.txt") if n not in names]
    if missing:
        sys.exit(f"\n  ZIP is incomplete, missing: {missing}")

    leaked = [n for n in names if any(bad in n for bad in FORBIDDEN)]
    if leaked:
        sys.exit(f"\n  ZIP contains auto-start machinery: {leaked}")

    print("\n  ok: entry points present, no auto-start machinery included\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
