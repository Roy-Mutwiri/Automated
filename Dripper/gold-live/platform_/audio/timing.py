"""Owning the silence between chunks.

A synthesiser pads every render with a little near-silence at each end, and
the router used to synthesise the next chunk only after the previous one had
finished playing. Measured on en_US-john-medium, that made the gap inside a
single utterance:

    trailing silence in the first wav      136-186ms
    synthesis of the next chunk            336-385ms
    leading silence in the next wav         53- 63ms
    playback poll granularity                   20ms
                                          ----------
                                          ~526-617ms

Half a second of dead air in the middle of one thought. Nobody chose it; it
fell out of doing things in sequence, and it is the single most unnatural
thing the audio path did.

Two fixes, and the second is only possible because of the first:

  1. Prefetch the next chunk while the current one plays. Piper's real-time
     factor is ~0.05, so rendering eight seconds of audio costs about 380ms --
     there is enormous headroom to do it during playback. That removes the
     synthesis term entirely. (In runtime/../router.py, not here.)

  2. Trim the padded silence to a known small amount, then insert the pause
     the DeliveryPlan actually asked for.

Together they turn an accidental ~562ms into a deliberate 40-320ms, which is
what makes `pause_before_ms` and `gap_ms` mean anything at all. You cannot
express timing until you control it.

Trimming is deliberately conservative. Cutting into a plosive or a final
fricative is far more noticeable than a few extra milliseconds of silence, so
every guard here fails toward leaving audio alone.
"""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Fraction of peak amplitude below which a sample counts as silence. Piper
#: pads with very low-level noise rather than true zeros, so an absolute
#: threshold misses it completely.
SILENCE_FRACTION = 0.02

#: Silence deliberately left at each end after trimming. Not zero: speech that
#: starts on the very first sample sounds clipped, and a hard cut at the end
#: swallows the release of a final consonant.
KEEP_MS = 20.0

#: Refuse to trim more than this share of a file. If a render is mostly quiet
#: something is wrong with it, and aggressively trimming a bad render turns a
#: subtle problem into a loud one.
MAX_TRIM_FRACTION = 0.5

#: Files shorter than this are left alone entirely -- a reaction word is short
#: enough that its padding is most of it.
MIN_DURATION_MS = 120.0


@dataclass(frozen=True, slots=True)
class TrimResult:
    trimmed: bool
    removed_head_ms: float
    removed_tail_ms: float
    duration_ms: float
    reason: str = ""

    @property
    def removed_ms(self) -> float:
        return self.removed_head_ms + self.removed_tail_ms


def _samples(frames: bytes):
    """16-bit little-endian PCM as a numpy array.

    numpy is a hard dependency of this project (see pyproject), and these run
    on every rendered chunk -- a pure-Python loop over 176k samples costs tens
    of milliseconds for no reason.
    """
    import numpy as np

    return np.frombuffer(frames, dtype="<i2")


def _voiced_bounds(frames: bytes, silence_fraction: float) -> tuple[int, int] | None:
    """First and last sample index louder than the silence floor.

    The floor is relative to this render's own peak: Piper pads with low-level
    noise rather than true zeros, and an absolute threshold either misses the
    padding entirely or eats quiet speech, depending on the voice.
    """
    import numpy as np

    a = _samples(frames)
    if not a.size:
        return None
    magnitude = np.abs(a.astype("int32"))
    peak = int(magnitude.max())
    if peak == 0:
        return None
    loud = np.flatnonzero(magnitude > peak * silence_fraction)
    if not loud.size:
        return None
    return int(loud[0]), int(loud[-1])


def trim_edge_silence(
    path: Path,
    *,
    keep_ms: float = KEEP_MS,
    silence_fraction: float = SILENCE_FRACTION,
) -> TrimResult:
    """Trim padded silence from both ends of a mono 16-bit wav, in place.

    Returns what it did. Never raises for an unreadable or unexpected file --
    a failure to trim must degrade to "leave it alone", because the
    alternative is losing an utterance over a cosmetic optimisation.
    """
    try:
        with wave.open(str(path), "rb") as w:
            channels, width, rate, n = (
                w.getnchannels(),
                w.getsampwidth(),
                w.getframerate(),
                w.getnframes(),
            )
            frames = w.readframes(n)
    except Exception as exc:  # unreadable, truncated, or not a wav at all
        return TrimResult(False, 0.0, 0.0, 0.0, f"unreadable: {exc}")

    duration_ms = (n / rate * 1000) if rate else 0.0
    if channels != 1 or width != 2:
        return TrimResult(False, 0.0, 0.0, duration_ms, "not mono 16-bit")
    if duration_ms < MIN_DURATION_MS:
        return TrimResult(False, 0.0, 0.0, duration_ms, "too short to trim")

    bounds = _voiced_bounds(frames, silence_fraction)
    if bounds is None:
        return TrimResult(False, 0.0, 0.0, duration_ms, "silent render")

    first, last = bounds
    keep = int(keep_ms / 1000 * rate)
    start = max(0, first - keep)
    end = min(n, last + keep + 1)

    if end <= start:
        return TrimResult(False, 0.0, 0.0, duration_ms, "degenerate bounds")
    if (end - start) < n * (1 - MAX_TRIM_FRACTION):
        # Would remove more than half the file. Something is wrong with this
        # render; leave it intact so the problem stays audible and diagnosable.
        return TrimResult(False, 0.0, 0.0, duration_ms, "would trim too much")
    if start == 0 and end == n:
        return TrimResult(False, 0.0, 0.0, duration_ms, "nothing to trim")

    try:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(width)
            w.setframerate(rate)
            w.writeframes(frames[start * 2 : end * 2])
    except Exception as exc:
        log.warning("could not rewrite %s after trimming: %s", path.name, exc)
        return TrimResult(False, 0.0, 0.0, duration_ms, f"rewrite failed: {exc}")

    head_ms = start / rate * 1000
    tail_ms = (n - end) / rate * 1000
    return TrimResult(
        trimmed=True,
        removed_head_ms=round(head_ms, 1),
        removed_tail_ms=round(tail_ms, 1),
        duration_ms=round((end - start) / rate * 1000, 1),
    )
