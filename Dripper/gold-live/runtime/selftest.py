"""Does this build actually contain what it claims to contain?

Kept deliberately separate from `doctor` and from readiness:

    doctor      what is installed, configured, or missing on this machine
    selftest    can the real components perform their real operations
    readiness   is this installation ready to start a production session

`selftest --imports` is the cheapest and most important of the three. Two
packages -- websockets and piper -- are imported lazily inside functions, so
PyInstaller's static analysis never saw them and never bundled them. The
packaged executable therefore could not fetch a price or speak, and said
nothing about it, because both subsystems are designed to degrade quietly.

Running this INSIDE the frozen process is the whole point. Importing these
names from a source checkout proves nothing about the exe.
"""

from __future__ import annotations

import asyncio
import importlib
import sys

#: Must match BUNDLED in build_tools/GoldLive.spec. Anything imported lazily
#: belongs here even more than the top-level imports do, because those are the
#: ones the build cannot discover on its own.
BUNDLED_IMPORTS: list[tuple[str, str]] = [
    ("websockets", "gold market feed (lazy: exchange_feed.py, feeds.py)"),
    ("piper", "text to speech (lazy: tts/piper.py)"),
    ("onnxruntime", "piper's inference engine"),
    ("httpx", "LLM client (lazy: supervisor.check_ready)"),
    ("numpy", "audio buffers, similarity index"),
    ("sounddevice", "audio output (lazy: audio/router.py)"),
    ("soundfile", "WAV read/write (lazy: audio/router.py)"),
    ("mss", "screen capture"),
    ("pydantic", "every contract"),
    ("yaml", "session and persona config"),
]


def missing_imports() -> list[str]:
    missing = []
    for name, _why in BUNDLED_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:  # ImportError, but also broken native loads
            missing.append(f"{name} ({exc})")
    return missing


def check_imports(verbose: bool = True) -> int:
    from shared.paths import is_frozen

    if verbose:
        mode = "frozen build" if is_frozen() else "source checkout"
        print(f"\n  Gold Live selftest -- imports ({mode})\n")

    failed = 0
    for name, why in BUNDLED_IMPORTS:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "")
            if verbose:
                print(f"  [  ok  ] {name:<14} {version:<10} {why}")
        except Exception as exc:
            failed += 1
            if verbose:
                print(f"  [ FAIL ] {name:<14} {'':<10} {exc}")

    if verbose:
        print()
        if failed:
            print(f"  {failed} required package(s) missing. This build is incomplete;\n"
                  "  it cannot fetch a price or speak. Download Gold Live again.\n")
        else:
            print(f"  All {len(BUNDLED_IMPORTS)} bundled packages import correctly.\n")
    return 1 if failed else 0


def main() -> int:
    argv = sys.argv[1:]
    imports_only = "--imports" in argv

    code = check_imports()
    if imports_only or code != 0:
        return code

    # Without --imports, also exercise the real components. This overlaps
    # readiness on purpose: selftest answers "do the parts work", readiness
    # answers "is this installation ready to broadcast", and the first is
    # useful on a machine where the second is not yet achievable.
    from runtime.readiness import check

    result = asyncio.run(check(include_broadcast=True))
    print(result.render())
    print()
    return 0 if result.by_name("model_real") and all(
        g.ok for g in result.gates if g.name in ("voice_real", "audio_out")
    ) else 1
