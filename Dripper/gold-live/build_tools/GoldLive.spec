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

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH).parent

datas = [
    (str(ROOT / "configs"), "configs"),
    (str(ROOT / ".env.example"), "."),
    (str(ROOT / "pyproject.toml"), "."),
    (str(ROOT / "build_tools" / "THIRD_PARTY_NOTICES.md"), "."),
]

# Imports that static analysis cannot see. Two categories, both real:
#
#   dynamic   pydantic and yaml build things at import time
#   lazy      modules imported INSIDE a function, which PyInstaller's
#             bytecode scan never reaches
#
# The lazy ones are why a packaged build shipped without websockets and piper
# and therefore could not fetch a price or speak. They are named explicitly
# here and verified after the build by `GoldLive.exe selftest --imports`.
hiddenimports = [
    "pydantic.deprecated.decorator",
    "yaml",
    "sqlite3",
    *collect_submodules("pydantic"),
]

# Every third-party package the runtime imports, declared statically.
#
# This used to be a `try: __import__ / except ImportError: skip` loop, which
# made the contents of the shipped artifact a function of whatever happened to
# be installed in the build machine's virtualenv -- the same build script
# produced a different product on different machines. It is now a manifest:
# if something here is missing, the build FAILS rather than quietly shipping a
# reduced product.
#
# Keep this in sync with [project].dependencies in pyproject.toml.
BUNDLED = [
    "websockets",     # gold market feed        (lazy: exchange_feed.py, feeds.py)
    "piper",          # text to speech          (lazy: tts/piper.py)
    "onnxruntime",    # piper's inference engine
    "httpx",          # LLM client, REST feeds  (lazy: supervisor.check_ready)
    "numpy",          # audio buffers, similarity index
    "sounddevice",    # audio output            (lazy: audio/router.py)
    "soundfile",      # WAV read/write          (lazy: audio/router.py)
    "mss",            # screen capture
]

# sounddevice, soundfile and onnxruntime wrap native libraries (PortAudio,
# libsndfile, onnxruntime.dll) that are NOT Python modules -- collect_data_files
# alone misses them and the built exe fails at import with an opaque OSError on
# a machine with no system copy. collect_dynamic_libs is what ships the DLLs.
binaries = []
missing = []
for package in BUNDLED:
    try:
        __import__(package)
    except ImportError as exc:
        missing.append(f"{package} ({exc})")
        continue
    hiddenimports.append(package)
    hiddenimports += collect_submodules(package)
    datas += collect_data_files(package)
    binaries += collect_dynamic_libs(package)

if missing:
    lines = [
        "",
        "[spec] BUILD FAILED -- required packages are not installed:",
        *[f"    {m}" for m in missing],
        "",
        "Install them and rebuild:  pip install -e .",
        "",
        "Shipping without them produces an executable that cannot fetch a price",
        "or speak -- silently, because both subsystems degrade rather than crash.",
        "That is exactly the failure this check exists to prevent, so the build",
        "does not continue.",
        "",
    ]
    raise SystemExit("\n".join(lines))

print(f"[spec] bundling {len(BUNDLED)} runtime packages: {', '.join(BUNDLED)}")

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
    binaries=binaries,
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
