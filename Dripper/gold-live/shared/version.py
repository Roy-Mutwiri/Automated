"""One application version, resolved from pyproject.toml.

There were three hardcoded copies of the version string -- pyproject said
0.1.0, the CLI said 0.4.0 and the build script said 0.4.0 again. A version
that disagrees with itself is worse than no version at all: the provisioning
state keys off it to decide whether an installation needs migrating, so a
wrong answer here silently corrupts the upgrade path.

pyproject.toml is the single source of truth. Resolution order:

  1. installed package metadata   (pip install / pip install -e)
  2. pyproject.toml next to the code (source checkout, and frozen builds,
     which bundle it precisely so this keeps working)
  3. a sentinel, never a plausible-looking number

Step 3 deliberately does not invent "0.0.0". An installation that cannot
determine its own version must look obviously broken, because provisioning
will refuse to migrate it.
"""

from __future__ import annotations

import re
from functools import lru_cache

from shared.paths import resource_root

DIST_NAME = "gold-live"
UNKNOWN = "0.0.0+unknown"

_VERSION_RE = re.compile(r'^\s*version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


def _from_metadata() -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(DIST_NAME)
    except (ImportError, PackageNotFoundError):
        return None
    except Exception:  # pragma: no cover - metadata backends vary
        return None


def _from_pyproject() -> str | None:
    path = resource_root() / "pyproject.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    # Only the [project] table's version, not a version pin inside a
    # dependency list further down the file.
    project = text.split("[project]", 1)
    if len(project) != 2:
        return None
    body = project[1].split("\n[", 1)[0]
    match = _VERSION_RE.search(body)
    return match.group(1) if match else None


@lru_cache(maxsize=1)
def app_version() -> str:
    return _from_metadata() or _from_pyproject() or UNKNOWN


def is_known(version: str | None = None) -> bool:
    """False when the version could not be resolved at all."""
    return (version or app_version()) != UNKNOWN
