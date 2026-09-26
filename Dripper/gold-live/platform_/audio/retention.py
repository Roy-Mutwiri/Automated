"""Retention for spoken audio, so a session that runs for months does not
fill the disk.

Every synthesised utterance is written to a wav before it is played. Nothing
used to delete them: a measured 5,446 files and 1.75 GB accumulated in three
days, which at the production cadence is roughly 28 GB a month. That is a
deterministic disk leak, not allocator noise -- the files are only ever added.

The policy deliberately keeps a window rather than deleting on playback.
Recent audio is worth having: it is the only way to hear what the host
actually said when something sounds wrong, and it costs nothing as long as
the window is bounded.

Two bounds, whichever binds first:

  * age    -- nothing older than `max_age_s` survives
  * size   -- the directory never exceeds `max_bytes`, oldest evicted first

Both are needed. Age alone is unbounded if the host talks fast; size alone
keeps ancient audio on a quiet session.

Sweeps are driven by counted events (start, every N utterances, stop), never
by a background task or a wall-clock sleep: a cleaner that outlives the
session it cleans is its own kind of leak.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

#: One file per segment, and the transcript keeps 500 utterances
#: (runtime.session.RETAINED_UTTERANCES). Matching the audio window to the
#: transcript window keeps the two consistent: if a line is still in the
#: transcript, the audio for it is usually still on disk.
#:
#: At the measured mean of 329 KB per file that is roughly 190 MB, so the
#: size bound below is the one that actually protects the disk when
#: utterances are long.
AUDIO_MAX_AGE_S: float = 6 * 60 * 60
AUDIO_MAX_BYTES: int = 256 * 1024 * 1024

#: How often to enforce, in completed utterances. Counted, not timed, so the
#: behaviour is identical on a fast machine and a slow one.
#:
#: One, not 25. At 25 the frozen build reached 9.21 MB against a 4 MB ceiling:
#: the host speaks roughly twice a minute, so a 25-utterance interval is about
#: twelve minutes during which the directory grows unchecked. A ceiling that
#: is only checked occasionally is not a ceiling. Sweeping every utterance
#: costs one glob over a directory the policy itself keeps small, against an
#: utterance that took seconds to synthesise.
#:
#: It also keeps `last_sweep` fresh, which is what health reports.
SWEEP_EVERY_UTTERANCES: int = 1

#: The policy can only evict a file after it exists, so the directory may
#: exceed `max_bytes` by at most the audio of the one utterance in flight.
#: Stated here rather than hidden: the bound is "cap plus one utterance".


@dataclass(frozen=True)
class RetentionPolicy:
    max_age_s: float = AUDIO_MAX_AGE_S
    max_bytes: int = AUDIO_MAX_BYTES

    def describe(self) -> str:
        return (f"{self.max_age_s / 3600:.1f}h / "
                f"{self.max_bytes / (1024 * 1024):.0f} MB")


@dataclass
class SweepResult:
    deleted: int = 0
    freed_bytes: int = 0
    kept: int = 0
    kept_bytes: int = 0
    protected: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"deleted {self.deleted} ({self.freed_bytes / 1e6:.1f} MB), "
                f"kept {self.kept} ({self.kept_bytes / 1e6:.1f} MB), "
                f"protected {self.protected}, errors {len(self.errors)}")


class AudioRetention:
    """Enforces the policy over exactly one session's audio directory.

    Scoped to one directory on purpose: sessions each own their own, so one
    session can never evict another's audio, and no cross-session locking is
    needed.
    """

    def __init__(self, directory: Path, policy: RetentionPolicy | None = None) -> None:
        self.directory = directory
        self.policy = policy or RetentionPolicy()

    def _entries(self) -> list[tuple[Path, float, int]]:
        found: list[tuple[Path, float, int]] = []
        if not self.directory.is_dir():
            return found
        for path in self.directory.glob("*.wav"):
            try:
                stat = path.stat()
            except OSError:
                # Vanished between listing and stat: someone else's problem,
                # and nothing to reclaim.
                continue
            found.append((path, stat.st_mtime, stat.st_size))
        found.sort(key=lambda e: e[1])  # oldest first
        return found

    def sweep(self, protect: set[Path] | None = None,
              now: float | None = None) -> SweepResult:
        """Apply the policy. Files in `protect` are never touched.

        `protect` is the set the router is currently synthesising into or
        playing from. A file being read into memory by soundfile must not be
        unlinked underneath it, and a file that has not finished being written
        is not a candidate for eviction either.
        """
        protect = {Path(p) for p in (protect or set())}
        now = now if now is not None else time.time()
        result = SweepResult()

        entries = self._entries()
        survivors: list[tuple[Path, float, int]] = []

        # 1. Age bound.
        for path, mtime, size in entries:
            if path in protect:
                result.protected += 1
                survivors.append((path, mtime, size))
                continue
            if now - mtime > self.policy.max_age_s:
                self._remove(path, size, result)
            else:
                survivors.append((path, mtime, size))

        # 2. Size bound, oldest first, protected files excluded from eviction
        #    but still counted against the total -- they are really there.
        total = sum(size for _, _, size in survivors)
        for path, mtime, size in list(survivors):
            if total <= self.policy.max_bytes:
                break
            if path in protect:
                continue
            if self._remove(path, size, result):
                total -= size
                survivors.remove((path, mtime, size))

        result.kept = len(survivors)
        result.kept_bytes = sum(size for _, _, size in survivors)
        return result

    def _remove(self, path: Path, size: int, result: SweepResult) -> bool:
        try:
            path.unlink()
        except FileNotFoundError:
            return True
        except OSError as exc:
            # A locked file is expected occasionally on Windows. Recorded
            # rather than swallowed, and never fatal: failing to reclaim disk
            # must not take the session down.
            result.errors.append(f"{path.name}: {exc}")
            log.warning("could not remove spoken audio %s: %s", path.name, exc)
            return False
        result.deleted += 1
        result.freed_bytes += size
        return True

    def usage(self) -> tuple[int, int]:
        """(file count, total bytes) currently on disk."""
        entries = self._entries()
        return len(entries), sum(size for _, _, size in entries)
