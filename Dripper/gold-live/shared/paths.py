"""Path resolution that works both from source and frozen into an exe.

PyInstaller unpacks bundled data to a temporary directory exposed as
`sys._MEIPASS`, which is different every run and read-only in practice. That
splits paths into two kinds, and conflating them is the classic packaging bug:

    resource_path()  read-only things shipped INSIDE the exe -- personas,
                     content seed, default config
    data_path()      writable things that must OUTLIVE the process -- the trace
                     database, captured audio, per-machine calibration, .env

Writing to a resource path appears to work in development and silently
evaporates in the packaged build, because the temp directory is deleted on
exit. Every write in this codebase must go through data_path().
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "GoldLive"


def is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def resource_root() -> Path:
    """Read-only files bundled with the application."""
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """Writable per-machine state. Survives upgrades and reinstalls.

    Honours GOLDLIVE_DATA_DIR so an operator can put it on a specific drive --
    seven sessions writing audio adds up, and the system disk is rarely where
    you want it.
    """
    override = os.environ.get("GOLDLIVE_DATA_DIR")
    if override:
        return Path(override).expanduser()

    if not is_frozen():
        # From source, keep everything in the checkout so nothing leaks into
        # the user profile during development.
        return Path(__file__).resolve().parent.parent

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )
    return base / APP_NAME


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def data_path(*parts: str, create_parent: bool = True) -> Path:
    path = data_root().joinpath(*parts)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def config_path(name: str) -> Path:
    """Config resolves writable-first, then falls back to the bundled default.

    That is what lets an operator edit personas or sessions.yaml on a machine
    running the packaged build without repackaging anything.
    """
    override = data_path("configs", name, create_parent=False)
    if override.exists():
        return override
    return resource_path("configs", name)


def config_dir(name: str) -> Path:
    override = data_root() / "configs" / name
    if override.is_dir() and any(override.iterdir()):
        return override
    return resource_root() / "configs" / name


def describe() -> str:
    return (
        f"frozen={is_frozen()}\n"
        f"resources={resource_root()}\n"
        f"data={data_root()}\n"
        f"database={data_path('data', 'gold-live.db', create_parent=False)}"
    )
