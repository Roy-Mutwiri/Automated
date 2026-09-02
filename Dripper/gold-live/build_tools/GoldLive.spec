# PyInstaller spec for the full application.
#
#   pyinstaller build_tools/GoldLive.spec --noconfirm
#
# Produces dist/GoldLive/GoldLive.exe as a ONE-DIRECTORY build, deliberately.
# A one-file build unpacks tens of megabytes to a temp directory on every
# launch, which adds seconds to startup and breaks under systems that clear
# temp aggressively. For a service that runs 24/7 and is restarted by a
# supervisor, one-dir is the right shape.
#
# What is NOT bundled, and cannot be:
#   the language model      tens of gigabytes; downloaded separately and served
#                           by vLLM or Ollama. `GoldLive.exe doctor` checks it.
#   piper voices            ~60MB each, licensed separately per voice
#   paddleocr models        downloaded on first use into the data directory
#
# See packaging/README.md for the distribution checklist.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent

datas = [
    (str(ROOT / "configs"), "configs"),
    (str(ROOT / ".env.example"), "."),
    (str(ROOT / "build_tools" / "THIRD_PARTY_NOTICES.md"), "."),
]

# Pydantic and yaml pull things in dynamically that static analysis misses.
hiddenimports = [
    "pydantic.deprecated.decorator",
    "yaml",
    "sqlite3",
    *collect_submodules("pydantic"),
]

# Optional heavy extras. Present only if installed at build time -- the base
# build stays small and the capture-enabled build is a superset.
optional = {
    "httpx": "httpx",
    "numpy": "numpy",
    "sounddevice": "sounddevice",
    "soundfile": "soundfile",
    "mss": "mss",
}
for module, package in optional.items():
    try:
        __import__(module)
        hiddenimports.append(package)
        datas += collect_data_files(package)
    except ImportError:
        print(f"[spec] optional dependency not installed, skipping: {package}")

excludes = [
    # Never ship a test framework or notebook stack to a customer.
    "pytest", "_pytest", "IPython", "jupyter", "notebook",
    "matplotlib", "tkinter", "PIL.ImageTk",
    # Torch arrives via paddleocr on some systems and is enormous. The model
    # runs in a separate server, so it is not needed here.
    "torch", "torchvision",
]

a = Analysis(
    [str(ROOT / "runtime" / "cli.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GoldLive",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX-packed exes are a common false positive for AV
    console=True,       # this is a service; it must log to a console
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "build_tools" / "icon.ico")
    if (ROOT / "build_tools" / "icon.ico").exists()
    else None,
    version=str(ROOT / "build_tools" / "version_info.txt")
    if sys.platform == "win32" and (ROOT / "build_tools" / "version_info.txt").exists()
    else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="GoldLive",
)
