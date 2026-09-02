"""Build the distributable.

    python -m build_tools.build                 # full build
    python -m build_tools.build --skip-tests    # iterate on packaging only
    python -m build_tools.build --zip           # also produce a release archive

Runs the test suite first by default. Shipping a build whose tests were not run
is how a regression reaches a customer, and this is the last gate before that.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "GoldLive"
VERSION = "0.4.0"


def run(cmd: list[str], **kw) -> int:
    print(f"\n$ {' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=ROOT, **kw)  # noqa: S603


def purge(path: Path, attempts: int = 5) -> None:
    """Remove a directory, tolerating transient Windows file locks.

    A sync client or scanner holding a handle for a moment is normal and
    self-resolving; failing the whole build over it is not.
    """
    import time

    for attempt in range(attempts):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            return
        except (PermissionError, OSError):
            if attempt == attempts - 1:
                print(
                    f"warning: could not remove {path} (a file is locked).\n"
                    "         Building on top of it; run --clean later if the "
                    "output looks stale."
                )
                return
            time.sleep(0.6 * (attempt + 1))


def check_toolchain() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        sys.exit("PyInstaller is not installed.  pip install pyinstaller")


def write_version_info() -> None:
    """Windows file properties. Customers and AV heuristics both look here; an
    unsigned binary with no version metadata is treated as more suspicious."""
    if sys.platform != "win32":
        return
    major, minor, patch = (int(p) for p in VERSION.split("."))
    (ROOT / "build_tools" / "version_info.txt").write_text(
        f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({major}, {minor}, {patch}, 0),
    prodvers=({major}, {minor}, {patch}, 0),
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('FileDescription', 'Gold Live broadcasting system'),
      StringStruct('FileVersion', '{VERSION}'),
      StringStruct('InternalName', 'GoldLive'),
      StringStruct('OriginalFilename', 'GoldLive.exe'),
      StringStruct('ProductName', 'Gold Live'),
      StringStruct('ProductVersion', '{VERSION}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""",
        encoding="utf-8",
    )


def stage_extras() -> None:
    """Files that sit beside the exe rather than inside it."""
    for name in ("THIRD_PARTY_NOTICES.md", "SETUP.md"):
        source = ROOT / "build_tools" / name
        if not source.exists():
            source = ROOT / name
        if source.exists():
            shutil.copy2(source, DIST / name)

    (DIST / "voices").mkdir(exist_ok=True)
    (DIST / "voices" / "PUT_VOICE_MODELS_HERE.txt").write_text(
        "Piper voice models (.onnx plus .onnx.json) go in this directory.\n"
        "Download from https://github.com/rhasspy/piper/blob/master/VOICES.md\n"
        "Each voice carries its own licence -- check it before distributing.\n",
        encoding="utf-8",
    )

    (DIST / "START HERE.txt").write_text(
        f"""Gold Live {VERSION}

Read SETUP.md - it walks through the whole thing.

The short version:

  1.  GoldLive.exe setup
  2.  Install Ollama from ollama.com, then:  ollama serve
      (Gold Live finds it automatically - no configuration needed)
  3.  GoldLive.exe doctor      tells you what is missing
  4.  GoldLive.exe dryrun      hear it before setting up anything else

Then, when you want it on a real stream, SETUP.md covers audio,
comment capture, and running several sessions at once.

Your settings live outside this folder so upgrades do not overwrite
them. "GoldLive.exe paths" shows where.
""",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Gold Live distributable")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--zip", action="store_true", help="produce a release archive")
    ap.add_argument("--clean", action="store_true", help="remove build/ and dist/ first")
    args = ap.parse_args()

    check_toolchain()

    if args.clean:
        for path in (ROOT / "build", ROOT / "dist"):
            purge(path)
        print("cleaned build/ and dist/")

    if not args.skip_tests:
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        if run([sys.executable, "-m", "pytest", "-q"], env=env) != 0:
            sys.exit("tests failed - not building. Fix them or pass --skip-tests.")

    write_version_info()

    # PyInstaller's own --clean uses a plain rmtree, which fails on Windows
    # when a sync client (OneDrive, Dropbox) or an antivirus scanner still has
    # a handle on the previous build. Do it ourselves with retries instead, and
    # do not pass --clean.
    purge(ROOT / "build")
    purge(DIST.parent / "GoldLive")

    code = run([
        sys.executable, "-m", "PyInstaller",
        str(ROOT / "build_tools" / "GoldLive.spec"),
        "--noconfirm",
    ])
    if code != 0:
        sys.exit(f"PyInstaller failed ({code})")

    if not DIST.exists():
        sys.exit(f"expected output at {DIST}")

    stage_extras()

    size_mb = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file()) / 1e6
    print(f"\n  Built {DIST}  ({size_mb:.0f} MB)")

    if args.zip:
        archive = ROOT / "dist" / f"GoldLive-{VERSION}-win64.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in DIST.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(DIST.parent))
        print(f"  Archive {archive}  ({archive.stat().st_size / 1e6:.0f} MB)")

    print(
        "\n  Before distributing:\n"
        "    - code-sign GoldLive.exe, or customers get SmartScreen warnings\n"
        "    - confirm every model and voice licence permits commercial use\n"
        "    - test on a clean machine with no Python installed\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
