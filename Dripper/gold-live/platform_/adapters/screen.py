"""Screen-capture comment adapter.

Reads the LIVE Studio comment panel off our own screen and emits normalised
CommentEvents. Nothing above this class knows the comments came from pixels --
it satisfies the same PlatformAdapter interface as an API-backed adapter, which
is what keeps an account-level problem on one platform survivable.

Pipeline per frame:

    grab crop -> hash -> (unchanged? stop) -> OCR rows -> parse author/text
              -> filter overlays -> dedupe -> CommentEvent

The hash gate matters more than it looks. Comments render for seconds, so 5fps
misses nothing, and on a quiet stream the perceptual hash short-circuits almost
every frame before OCR runs. Without it you are running seven OCR pipelines
continuously for no reason.

Known limits, stated rather than hidden:
  - Fast chat drops comments. Inherent at this frame rate. The Director already
    samples comments, so impact is low, but you cannot promise a viewer their
    question will be seen.
  - A LIVE Studio update that moves the panel silently produces zero comments.
    That is what the stall watchdog below is for.
  - Emoji and non-Latin scripts OCR poorly; low-confidence rows are dropped
    rather than guessed at.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

from platform_.adapters.base import PlatformAdapter
from platform_.adapters.ocr import OCREngine, OCRLine
from shared.contracts import (
    CaptureCalibration,
    CommentEvent,
    HealthCheck,
    HealthState,
    ServiceHealth,
)

log = logging.getLogger(__name__)

# "author: message" as rendered in most chat panels. The colon is the usual
# separator; some skins use a bullet or an em dash.
AUTHOR_SPLIT = re.compile(r"^\s*([^:•—]{1,32})\s*[:•—]\s*(.+)$")

# Rows that are not comments: joins, gifts, follows, system notices.
OVERLAY_PATTERNS = [
    re.compile(r"\b(joined|is watching|shared the|followed|liked|sent)\b", re.I),
    re.compile(r"^\s*(welcome|gift|combo|x\d+)\s*$", re.I),
]


@dataclass
class CaptureStats:
    frames: int = 0
    ocr_runs: int = 0
    rows_read: int = 0
    rows_dropped_confidence: int = 0
    rows_dropped_overlay: int = 0
    duplicates: int = 0
    emitted: int = 0
    last_emit_at: datetime | None = None
    ocr_ms: list[float] = field(default_factory=list)

    @property
    def skip_rate(self) -> float:
        """Fraction of frames the hash gate short-circuited before OCR."""
        return 0.0 if not self.frames else 1 - (self.ocr_runs / self.frames)


class FrameSource:
    """Grabs a fixed crop of the screen. Replaceable for tests."""

    def __init__(self, calibration: CaptureCalibration) -> None:
        self.cal = calibration
        self._sct = None

    def grab(self):
        if self._sct is None:
            try:
                import mss
            except ImportError as exc:  # pragma: no cover - depends on host
                raise RuntimeError("mss not installed. pip install mss") from exc
            self._sct = mss.mss()
        region = {
            "left": self.cal.crop_x, "top": self.cal.crop_y,
            "width": self.cal.crop_w, "height": self.cal.crop_h,
            "mon": self.cal.monitor,
        }
        import numpy as np

        return np.array(self._sct.grab(region))[:, :, :3]

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None


def frame_hash(image) -> int:
    """Cheap perceptual hash. Identical frames must hash identically; a single
    new comment row must not."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return hash(bytes(image))
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    # Downsample to 32x32 and threshold on the mean -- robust to antialiasing
    # jitter, sensitive to a row of text appearing.
    h, w = arr.shape[:2]
    ys = np.linspace(0, h - 1, min(32, h)).astype(int)
    xs = np.linspace(0, w - 1, min(32, w)).astype(int)
    small = arr[np.ix_(ys, xs)]
    bits = (small > small.mean()).astype(np.uint8).flatten()
    return hash(bits.tobytes())


def parse_row(line: OCRLine) -> tuple[str, str] | None:
    """Split one OCR row into (author, text). None if it is not a comment."""
    text = line.text.strip()
    if not text:
        return None
    for pattern in OVERLAY_PATTERNS:
        if pattern.search(text):
            return None
    match = AUTHOR_SPLIT.match(text)
    if not match:
        return None
    author, body = match.group(1).strip(), match.group(2).strip()
    if not author or not body:
        return None
    return author, body


class ScreenCaptureAdapter(PlatformAdapter):
    def __init__(
        self,
        session_id: str,
        calibration: CaptureCalibration,
        ocr: OCREngine,
        author_salt: str,
        platform: str = "tiktok",
        frame_source: FrameSource | None = None,
        dedupe_window: int = 512,
        stall_alert_after_s: float = 300.0,
    ) -> None:
        self.session_id = session_id
        self.cal = calibration
        self.ocr = ocr
        self.author_salt = author_salt
        self.platform = platform
        self.frames = frame_source or FrameSource(calibration)
        self.stall_alert_after_s = stall_alert_after_s

        self._seen: OrderedDict[str, None] = OrderedDict()
        self._dedupe_window = dedupe_window
        self._last_hash: int | None = None
        self._running = False
        self.stats = CaptureStats()
        self.started_at: datetime | None = None

    # -- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        if self.cal.crop_w <= 0 or self.cal.crop_h <= 0:
            raise ValueError(
                f"calibration for {self.session_id} has no crop region; "
                "run scripts/calibrate_capture.py on this device first"
            )
        self.ocr.warmup()
        self._running = True
        self.started_at = datetime.now(timezone.utc)
        log.info(
            "screen capture started for %s at %dx%d, %d fps",
            self.session_id, self.cal.crop_w, self.cal.crop_h, self.cal.fps,
        )

    async def disconnect(self) -> None:
        self._running = False
        self.frames.close()

    # -- ingest -----------------------------------------------------------

    def _is_duplicate(self, msg_id: str) -> bool:
        if msg_id in self._seen:
            self._seen.move_to_end(msg_id)
            return True
        self._seen[msg_id] = None
        if len(self._seen) > self._dedupe_window:
            self._seen.popitem(last=False)
        return False

    def process_frame(self, image) -> list[CommentEvent]:
        """One frame in, zero or more comments out. Synchronous and pure enough
        to test without a screen."""
        self.stats.frames += 1

        digest = frame_hash(image)
        if digest == self._last_hash:
            return []
        self._last_hash = digest

        t0 = time.perf_counter()
        lines = self.ocr.read(image)
        self.stats.ocr_runs += 1
        self.stats.ocr_ms.append((time.perf_counter() - t0) * 1000)
        if len(self.stats.ocr_ms) > 500:
            del self.stats.ocr_ms[:-500]

        out: list[CommentEvent] = []
        for line in lines:
            self.stats.rows_read += 1
            if line.confidence < self.cal.min_ocr_confidence:
                # Garbled text reaching the generator produces confused replies
                # about comments nobody made. Dropping always beats guessing.
                self.stats.rows_dropped_confidence += 1
                continue
            parsed = parse_row(line)
            if parsed is None:
                self.stats.rows_dropped_overlay += 1
                continue
            author, text = parsed

            # No message ids on screen, and the same comment appears in many
            # frames at different scroll positions. Content hashing collapses
            # both problems.
            msg_id = CommentEvent.synth_msg_id(author, text)
            if self._is_duplicate(msg_id):
                self.stats.duplicates += 1
                continue

            self.stats.emitted += 1
            self.stats.last_emit_at = datetime.now(timezone.utc)
            out.append(
                CommentEvent(
                    session_id=self.session_id,
                    platform=self.platform,  # type: ignore[arg-type]
                    platform_msg_id=msg_id,
                    author_hash=CommentEvent.hash_author(author, self.author_salt),
                    text_raw=text,
                    text_norm=" ".join(text.lower().split()),
                )
            )
        return out

    async def comments(self) -> AsyncIterator[CommentEvent]:  # type: ignore[override]
        interval = 1.0 / max(1, self.cal.fps)
        while self._running:
            started = time.perf_counter()
            try:
                for event in self.process_frame(self.frames.grab()):
                    yield event
            except Exception as exc:  # noqa: BLE001 - capture must not die
                log.warning("[%s] capture frame failed: %s", self.session_id, exc)
                await asyncio.sleep(1.0)
                continue
            elapsed = time.perf_counter() - started
            await asyncio.sleep(max(0.0, interval - elapsed))

    # -- health -----------------------------------------------------------

    async def health(self) -> ServiceHealth:
        """A LIVE Studio update that moves the panel produces zero comments and
        no error. Elapsed time is the only signal, so it is the check."""
        now = datetime.now(timezone.utc)
        checks = [
            HealthCheck(
                name="capture_running", ok=self._running,
                detail=None if self._running else "adapter not started",
            ),
            HealthCheck(
                name="calibration",
                ok=self.cal.crop_w > 0 and self.cal.crop_h > 0,
                detail=f"{self.cal.crop_w}x{self.cal.crop_h} @ {self.cal.crop_x},{self.cal.crop_y}",
            ),
        ]

        reference = self.stats.last_emit_at or self.started_at
        silent_s = (now - reference).total_seconds() if reference else 0.0
        stalled = silent_s > self.stall_alert_after_s
        checks.append(
            HealthCheck(
                name="comments_flowing", ok=not stalled,
                detail=f"no comments for {silent_s:.0f}s"
                if stalled
                else f"{self.stats.emitted} emitted",
            )
        )

        state = HealthState.OK
        reason = None
        if not self._running:
            state, reason = HealthState.DOWN, "adapter stopped"
        elif stalled:
            state = HealthState.DEGRADED
            reason = (
                f"no comments read for {silent_s:.0f}s - the panel may have moved; "
                "recalibrate with scripts/calibrate_capture.py"
            )

        return ServiceHealth(
            component="screen_capture", session_id=self.session_id,
            state=state, checks=checks, degraded_reason=reason,
        )
