"""First-run setup and preflight for the packaged build.

A packaged product lands on a machine that has none of the assumptions the
development checkout makes. This seeds writable config, then checks the things
that will otherwise fail confusingly at 3am, and says plainly which are fatal
and which merely degrade.

    GoldLive.exe doctor      check everything, change nothing
    GoldLive.exe setup       seed config into the data directory
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from shared.paths import config_path, data_path, data_root, describe, resource_root

SEED_FILES = ["sessions.yaml", "content.yaml"]
SEED_DIRS = ["personas"]


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = False

    def render(self) -> str:
        if self.ok:
            mark = "  ok  "
        else:
            mark = " FAIL " if self.fatal else " warn "
        return f"[{mark}] {self.name:<22} {self.detail}"


def seed_config(force: bool = False) -> list[str]:
    """Copy bundled defaults into the writable config directory.

    Only ever copies what is missing. An operator who has edited personas on a
    deployed machine must not lose that on the next launch.
    """
    written: list[str] = []
    target = data_root() / "configs"
    target.mkdir(parents=True, exist_ok=True)

    for name in SEED_FILES:
        dest = target / name
        if dest.exists() and not force:
            continue
        source = resource_root() / "configs" / name
        if source.exists():
            shutil.copy2(source, dest)
            written.append(str(dest))

    for name in SEED_DIRS:
        dest = target / name
        source = resource_root() / "configs" / name
        if not source.is_dir():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for item in source.glob("*.yaml"):
            out = dest / item.name
            if out.exists() and not force:
                continue
            shutil.copy2(item, out)
            written.append(str(out))

    env = data_root() / ".env"
    if not env.exists():
        sample = resource_root() / ".env.example"
        if sample.exists():
            shutil.copy2(sample, env)
            written.append(str(env))
    return written


def check_llm() -> Check:
    """Probe every well-known endpoint, not just one.

    Someone running this for the first time should not have to know that vLLM
    listens on 8000 and Ollama on 11434.
    """
    import os
    import urllib.error
    import urllib.request

    from platform_.llm.discovery import KNOWN_ENDPOINTS

    configured = os.environ.get("LLM_BASE_URL")
    candidates = (
        [("configured", configured)] if configured else list(KNOWN_ENDPOINTS)
    )

    for label, url in candidates:
        try:
            with urllib.request.urlopen(f"{url}/models", timeout=2) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            continue
        models = [m.get("id") for m in body.get("data", [])] or ["a model"]
        return Check("language model", True, f"{label} at {url}: {models[0]}")

    tried = ", ".join(url.split("//")[-1].split("/")[0] for _l, url in candidates)
    return Check(
        "language model", False,
        f"no server on {tried}. Start Ollama (`ollama serve`) or vLLM. "
        "Without one the host falls back to canned text.",
        fatal=True,
    )


def check_tts(voices_dir: Path) -> Check:
    if shutil.which("piper") is None:
        return Check("text to speech", False,
                     "piper not on PATH - audio will be a placeholder tone")
    voices = list(voices_dir.glob("*.onnx")) if voices_dir.is_dir() else []
    if not voices:
        return Check("text to speech", False,
                     f"piper found but no voices in {voices_dir}")
    return Check("text to speech", True, f"piper with {len(voices)} voice(s)")


def check_audio() -> Check:
    try:
        from platform_.audio.devices import find_virtual_cable, list_output_devices
    except Exception as exc:
        return Check("audio output", False, f"could not enumerate devices: {exc}")

    devices = list_output_devices()
    if not devices:
        return Check("audio output", False,
                     "no output devices (install sounddevice)")
    cable = find_virtual_cable()
    if cable is None:
        return Check(
            "audio output", False,
            f"{len(devices)} devices but no virtual cable. Install VB-CABLE and "
            "point LIVE Studio's MICROPHONE at it - never desktop audio.",
        )
    return Check("audio output", True, f"virtual cable: {cable.name}")


def check_ocr() -> Check:
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        return Check("comment capture", False,
                     "paddleocr not available - screen capture disabled")
    return Check("comment capture", True, "paddleocr available")


def check_calibration() -> Check:
    path = data_path("configs", "devices.json", create_parent=False)
    if not path.exists():
        return Check("capture calibration", False,
                     f"not calibrated - run: {_launcher()} calibrate --session SESSION_001")
    try:
        devices = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return Check("capture calibration", False, f"devices.json is invalid: {exc}")
    bound = [d for d in devices.values() if d.get("capture", {}).get("crop_w")]
    if not bound:
        return Check("capture calibration", False, "devices.json has no crop region")
    return Check("capture calibration", True, f"{len(bound)} device(s) calibrated")


def check_config() -> Check:
    try:
        import yaml

        cfg = yaml.safe_load(config_path("sessions.yaml").read_text(encoding="utf-8"))
        count = len(cfg.get("sessions", []))
        return Check("configuration", count > 0, f"{count} session(s) configured",
                     fatal=count == 0)
    except Exception as exc:
        return Check("configuration", False, f"unreadable: {exc}", fatal=True)


def check_disk() -> Check:
    try:
        usage = shutil.disk_usage(data_root())
    except OSError as exc:
        return Check("disk space", False, str(exc))
    free_gb = usage.free / 1e9
    # Seven sessions writing wav segments around the clock adds up quickly.
    return Check("disk space", free_gb > 10,
                 f"{free_gb:.1f} GB free at {data_root()}")


def doctor(voices_dir: Path | None = None) -> int:
    checks = [
        check_config(),
        check_llm(),
        check_tts(voices_dir or data_root() / "voices"),
        check_audio(),
        check_ocr(),
        check_calibration(),
        check_disk(),
    ]

    print(f"\n  Gold Live preflight\n\n{describe()}\n")
    for check in checks:
        print("  " + check.render())

    fatal = [c for c in checks if c.fatal and not c.ok]
    warnings = [c for c in checks if not c.fatal and not c.ok]

    print()
    if fatal:
        print(f"  {len(fatal)} blocking problem(s). The system will not run correctly.")
    elif warnings:
        print(f"  Ready, with {len(warnings)} degraded capability(ies).")
        print("  It will run, but not at full function until those are fixed.")
    else:
        print("  All checks passed.")
    print()
    return 1 if fatal else 0


def setup(force: bool = False) -> int:
    written = seed_config(force=force)
    print(f"\n  Data directory: {data_root()}")
    if written:
        print(f"  Seeded {len(written)} file(s):")
        for path in written:
            print(f"    {path}")
    else:
        print("  Already set up; nothing copied.")
    print(f"\n  Edit configs there, then run:  {_launcher()} doctor\n")
    return 0


def _launcher() -> str:
    """How the user actually invokes this build.

    sys.argv[0] is rewritten to 'goldlive-<command>' by the CLI dispatcher, so
    it cannot be used here -- it would tell a customer to run a command that
    does not exist.
    """
    from shared.paths import is_frozen

    return "GoldLive.exe" if is_frozen() else "python -m runtime.cli"
