"""Comment adapter that tails a text file. A development tool, deliberately.

The point is to prove the comment half of the pipeline without depending on a
platform. You type a line into a file, and the host hears it as a viewer
comment, classifies it, decides whether to answer, generates a reply, speaks it.

    GoldLive.exe run --session SESSION_001 --adapter file --tts piper
    # then, in another terminal:
    echo Where is resistance on Gold? >> %LOCALAPPDATA%\\GoldLive\\comments.txt

That isolates failures cleanly. If the host answers a typed comment but not a
real one, the fault is in OCR or the platform adapter, not in classification,
the Director, generation, TTS or audio -- all of which this exercises exactly
as the real path does.

It is also the only comment source that works inside the live process without
paddleocr, an API key, or a live broadcast, which makes it the natural thing to
develop against.

NOT for production: there is no authentication and no rate limiting, so anyone
who can write to the file can make the host say something. It is listed as a
development adapter for that reason.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

from platform_.adapters.base import PlatformAdapter
from shared.contracts import (
    CommentEvent,
    HealthCheck,
    HealthState,
    ServiceHealth,
)

log = logging.getLogger(__name__)

# "author: message" if a colon is present, otherwise the whole line with a
# default author -- typing a bare question should just work.
DEFAULT_AUTHOR = "tester"


class FileTailAdapter(PlatformAdapter):
    def __init__(
        self,
        session_id: str,
        path: str | Path,
        author_salt: str = "dev",
        poll_interval_s: float = 0.5,
        from_start: bool = False,
    ) -> None:
        self.session_id = session_id
        self.path = Path(path)
        self.author_salt = author_salt
        self.poll_interval_s = poll_interval_s
        #: Default is to start at the end, so an old file does not replay its
        #: whole history into a live session on startup.
        self.from_start = from_start

        self._running = False
        self._offset = 0
        self.lines_read = 0
        self.emitted = 0
        self.started_at: datetime | None = None
        self.last_emit_at: datetime | None = None

    # -- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

        self._offset = 0 if self.from_start else self.path.stat().st_size
        self._running = True
        self.started_at = datetime.now(timezone.utc)
        log.info(
            "file comment adapter watching %s (type lines to talk to the host)",
            self.path,
        )

    async def disconnect(self) -> None:
        self._running = False

    # -- parsing ----------------------------------------------------------

    def parse(self, line: str) -> CommentEvent | None:
        text = line.strip()
        if not text or text.startswith("#"):
            return None

        author = DEFAULT_AUTHOR
        # Only split on a colon that looks like a short handle, so a message
        # containing a colon mid-sentence is not mangled into a silly author.
        if ":" in text:
            head, _, tail = text.partition(":")
            if tail.strip() and 0 < len(head.strip()) <= 32 and " " not in head.strip():
                author, text = head.strip(), tail.strip()

        return CommentEvent(
            session_id=self.session_id,
            platform="mock",
            platform_msg_id=CommentEvent.synth_msg_id(author, text),
            author_hash=CommentEvent.hash_author(author, self.author_salt),
            text_raw=text,
            text_norm=" ".join(text.lower().split()),
        )

    def read_new(self) -> list[CommentEvent]:
        """Read whatever has been appended since the last check."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return []

        if size < self._offset:
            # The file was truncated or replaced; start again from the top
            # rather than seeking past the end and going permanently silent.
            log.info("comment file truncated; re-reading from the start")
            self._offset = 0
        if size == self._offset:
            return []

        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(self._offset)
            chunk = fh.read()
            self._offset = fh.tell()

        out: list[CommentEvent] = []
        for line in chunk.splitlines():
            self.lines_read += 1
            comment = self.parse(line)
            if comment is None:
                continue
            self.emitted += 1
            self.last_emit_at = datetime.now(timezone.utc)
            out.append(comment)
        return out

    async def comments(self) -> AsyncIterator[CommentEvent]:  # type: ignore[override]
        while self._running:
            try:
                for comment in self.read_new():
                    log.info("[%s] comment: %s", self.session_id, comment.text_raw)
                    yield comment
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("comment file read failed: %s", exc)
            await asyncio.sleep(self.poll_interval_s)

    # -- health -----------------------------------------------------------

    async def health(self) -> ServiceHealth:
        exists = self.path.exists()
        return ServiceHealth(
            component="file_comment_adapter",
            session_id=self.session_id,
            state=HealthState.OK if (self._running and exists) else HealthState.DOWN,
            checks=[
                HealthCheck(name="running", ok=self._running),
                HealthCheck(name="file", ok=exists, detail=str(self.path)),
                HealthCheck(
                    name="comments", ok=True,
                    detail=f"{self.emitted} emitted of {self.lines_read} lines",
                ),
            ],
            degraded_reason=None if self._running else "adapter not started",
        )
