"""Durable record of what this machine has been provisioned with.

The point of this file is to make provisioning *idempotent*. Without it every
launch has to re-derive what is installed, which either means re-downloading
gigabytes or guessing -- and guessing is how you end up telling someone their
session is ready when the model was deleted last week.

    NEW ──► PROVISIONING ──► READY
             │       │         │
             ▼       │         │
           FAILED ◄──┘         │
             │                 │
             └──── repair ─────┘

DEGRADED is deliberately not a persisted state. Whether the machine can run a
session *right now* depends on the market feed, the model server and the audio
device, none of which this file knows anything about. Readiness is recomputed
every launch; this file only records what was installed and verified.

Atomicity matters more than it looks: a half-written provisioning.json on a
machine that lost power mid-download is indistinguishable from a corrupt one,
and both must degrade to "provision me again" rather than to a crash.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from shared.paths import data_path
from shared.version import app_version

log = logging.getLogger(__name__)

#: Bumped when the *shape* of this file changes incompatibly. Distinct from
#: the application version, which changes far more often and for other reasons.
STATE_VERSION = 1

FILENAME = "provisioning.json"


class ProvisionState(str, Enum):
    NEW = "new"
    PROVISIONING = "provisioning"
    READY = "ready"
    FAILED = "failed"


class StateTooNew(RuntimeError):
    """The file was written by a newer GoldLive than the one reading it.

    Guessing at a schema from the future is how state gets silently destroyed,
    so this is raised rather than handled: the correct fix is to update the
    application, and anything else loses data.
    """


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Artifact:
    """One provisioned file or model, and the evidence that it is intact."""

    kind: str                      # "voice" | "model"
    artifact_id: str
    path: str | None = None        # None for server-managed models (Ollama)
    bytes: int = 0
    sha256: str | None = None      # voices: content hash we computed
    digest: str | None = None      # models: digest the server reports
    source: str | None = None
    licence: str | None = None
    verified_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "kind": self.kind, "path": self.path, "bytes": self.bytes,
            "sha256": self.sha256, "digest": self.digest, "source": self.source,
            "licence": self.licence, "verified_at": self.verified_at,
            **({"extra": self.extra} if self.extra else {}),
        }

    @classmethod
    def from_json(cls, artifact_id: str, raw: dict) -> Artifact:
        return cls(
            kind=raw.get("kind", "unknown"), artifact_id=artifact_id,
            path=raw.get("path"), bytes=raw.get("bytes", 0),
            sha256=raw.get("sha256"), digest=raw.get("digest"),
            source=raw.get("source"), licence=raw.get("licence"),
            verified_at=raw.get("verified_at"), extra=raw.get("extra", {}),
        )

    def looks_present(self) -> bool:
        """Cheap check: does the file exist at the recorded size?

        Deliberately does not hash. Hashing a 2 GB model on every launch would
        cost ten seconds to learn something that has not changed since the last
        launch. `verify()` is for when this is not enough.
        """
        if self.path is None:
            return True  # server-managed; presence is the server's to report
        p = Path(self.path)
        try:
            return p.is_file() and (self.bytes == 0 or p.stat().st_size == self.bytes)
        except OSError:
            return False

    def verify(self) -> tuple[bool, str]:
        """Full content check. Returns (ok, reason)."""
        if self.path is None:
            return True, "server-managed"
        p = Path(self.path)
        if not p.is_file():
            return False, f"missing: {p}"
        size = p.stat().st_size
        if self.bytes and size != self.bytes:
            return False, f"size {size} != recorded {self.bytes}"
        if self.sha256:
            actual = sha256_file(p)
            if actual != self.sha256:
                return False, f"sha256 {actual[:12]}… != recorded {self.sha256[:12]}…"
        return True, "verified"


@dataclass
class Provisioning:
    state: ProvisionState = ProvisionState.NEW
    state_version: int = STATE_VERSION
    goldlive_version: str = ""
    contracts_schema_version: str = ""
    bundle_id: str | None = None
    author_salt: str | None = None

    first_run_started_at: str | None = None
    first_run_completed_at: str | None = None

    hardware: dict[str, Any] = field(default_factory=dict)
    profile_hash: str | None = None

    dependencies: dict[str, dict] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)

    failed_stage: str | None = None
    failed_reason: str | None = None
    attempts: int = 0
    updated_at: str | None = None

    # -- serialisation ----------------------------------------------------

    def to_json(self) -> dict:
        return {
            "state_version": self.state_version,
            "state": self.state.value,
            "goldlive_version": self.goldlive_version,
            "contracts_schema_version": self.contracts_schema_version,
            "bundle_id": self.bundle_id,
            "author_salt": self.author_salt,
            "first_run": {
                "started_at": self.first_run_started_at,
                "completed_at": self.first_run_completed_at,
                "completed": self.first_run_completed_at is not None,
            },
            "hardware": self.hardware,
            "profile_hash": self.profile_hash,
            "dependencies": self.dependencies,
            "artifacts": {k: a.to_json() for k, a in self.artifacts.items()},
            "selection": self.selection,
            "last_provision": {
                "attempts": self.attempts,
                "failed_stage": self.failed_stage,
                "failed_reason": self.failed_reason,
            },
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, raw: dict) -> Provisioning:
        version = raw.get("state_version", 0)
        if version > STATE_VERSION:
            raise StateTooNew(
                f"provisioning.json is state_version {version}, but this build "
                f"understands {STATE_VERSION}. Update Gold Live rather than "
                f"letting an older build rewrite it."
            )

        first = raw.get("first_run", {}) or {}
        last = raw.get("last_provision", {}) or {}
        try:
            state = ProvisionState(raw.get("state", "new"))
        except ValueError:
            state = ProvisionState.NEW

        return cls(
            state=state,
            state_version=version or STATE_VERSION,
            goldlive_version=raw.get("goldlive_version", ""),
            contracts_schema_version=raw.get("contracts_schema_version", ""),
            bundle_id=raw.get("bundle_id"),
            author_salt=raw.get("author_salt"),
            first_run_started_at=first.get("started_at"),
            first_run_completed_at=first.get("completed_at"),
            hardware=raw.get("hardware", {}) or {},
            profile_hash=raw.get("profile_hash"),
            dependencies=raw.get("dependencies", {}) or {},
            artifacts={
                k: Artifact.from_json(k, v)
                for k, v in (raw.get("artifacts", {}) or {}).items()
            },
            selection=raw.get("selection", {}) or {},
            attempts=last.get("attempts", 0),
            failed_stage=last.get("failed_stage"),
            failed_reason=last.get("failed_reason"),
            updated_at=raw.get("updated_at"),
        )

    # -- queries ----------------------------------------------------------

    def is_provisioned(self) -> bool:
        return self.state is ProvisionState.READY and self.first_run_completed_at

    def version_changed(self) -> bool:
        return bool(self.goldlive_version) and self.goldlive_version != app_version()

    def bundle_changed(self, current: str | None) -> bool:
        return bool(self.bundle_id) and current is not None and self.bundle_id != current

    def profile_changed(self, current: str | None) -> bool:
        return bool(self.profile_hash) and current is not None and self.profile_hash != current

    def missing_artifacts(self) -> list[Artifact]:
        return [a for a in self.artifacts.values() if not a.looks_present()]

    # -- mutation ---------------------------------------------------------

    def record_artifact(self, artifact: Artifact) -> None:
        artifact.verified_at = artifact.verified_at or _now()
        self.artifacts[artifact.artifact_id] = artifact

    def begin(self) -> None:
        self.state = ProvisionState.PROVISIONING
        self.attempts += 1
        self.failed_stage = None
        self.failed_reason = None
        self.first_run_started_at = self.first_run_started_at or _now()

    def succeed(self) -> None:
        self.state = ProvisionState.READY
        self.goldlive_version = app_version()
        self.first_run_completed_at = self.first_run_completed_at or _now()
        self.failed_stage = None
        self.failed_reason = None

    def fail(self, stage: str, reason: str) -> None:
        self.state = ProvisionState.FAILED
        self.failed_stage = stage
        self.failed_reason = reason


# -- persistence ----------------------------------------------------------


def state_path() -> Path:
    return data_path(FILENAME, create_parent=True)


def save(state: Provisioning, path: Path | None = None) -> Path:
    """Write atomically. A partially-written primary file must never exist.

    Temp file in the same directory (so os.replace stays on one volume and is
    therefore atomic on Windows), flushed and fsynced before the rename, so a
    power loss leaves either the old file or the new one.
    """
    target = path or state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = _now()
    state.state_version = STATE_VERSION
    state.goldlive_version = state.goldlive_version or app_version()

    payload = json.dumps(state.to_json(), indent=2, sort_keys=False)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".provisioning-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return target



def load(path: Path | None = None) -> Provisioning:
    """Read the state, or return a NEW one.

    A corrupt file is moved aside rather than deleted -- if provisioning goes
    wrong repeatedly, the previous states are the evidence -- and the machine
    is treated as needing provisioning, which is the safe interpretation.

    StateTooNew is deliberately allowed to propagate.
    """
    target = path or state_path()
    if not target.exists():
        return Provisioning()

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("provisioning state must be a JSON object")
    except (OSError, ValueError) as exc:
        bad = target.with_suffix(f".bad-{datetime.now(timezone.utc):%Y%m%d%H%M%S}")
        try:
            target.replace(bad)
            log.warning("provisioning state unreadable (%s); moved to %s", exc, bad.name)
        except OSError:
            log.warning("provisioning state unreadable (%s) and could not be moved", exc)
        return Provisioning()

    return Provisioning.from_json(raw)


def author_salt() -> str:
    """The salt used to hash viewer handles, resolved once per process.

    Order: an explicit AUTHOR_SALT override, then the per-install value written
    at provisioning time, then the placeholder.

    The placeholder matters: without a per-install salt every copy of the
    application hashes the same handle to the same value, so hashes are
    correlatable across installs and the pseudonymisation is largely
    decorative. Provisioning generates 16 random bytes precisely to stop that,
    but the value was being written and never read.
    """
    override = os.environ.get("AUTHOR_SALT")
    if override:
        return override
    try:
        stored = load().author_salt
    except Exception:
        stored = None
    if stored:
        return stored
    log.warning(
        "no per-install author salt found; viewer handles will hash "
        "identically to every other install. Run: GoldLive.exe provision"
    )
    return "change-me"
