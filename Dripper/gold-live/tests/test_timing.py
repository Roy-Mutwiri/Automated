"""Trimming the padding off a render, and refusing to when it would hurt.

Every guard here fails toward leaving audio alone. Clipping a plosive or the
release of a final consonant is far more audible than a few extra milliseconds
of silence, so a trimmer that is unsure must do nothing.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from platform_.audio.timing import (
    KEEP_MS,
    MIN_DURATION_MS,
    trim_edge_silence,
)

RATE = 22050


def write_wav(
    path: Path,
    *,
    lead_silence_ms: float = 0.0,
    tone_ms: float = 500.0,
    trail_silence_ms: float = 0.0,
    amplitude: int = 12000,
    channels: int = 1,
    width: int = 2,
) -> Path:
    frames = bytearray()

    def silence(ms: float) -> None:
        # Low-level noise, not true zeros: that is what a synthesiser
        # actually pads with, and an absolute threshold would miss it.
        for i in range(int(ms / 1000 * RATE)):
            frames.extend(struct.pack("<h", 40 if i % 2 else -40))

    silence(lead_silence_ms)
    for i in range(int(tone_ms / 1000 * RATE)):
        frames.extend(
            struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 220 * i / RATE)))
        )
    silence(trail_silence_ms)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(RATE)
        w.writeframes(bytes(frames))
    return path


def duration_ms(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate() * 1000


# -- the job ----------------------------------------------------------------


def test_padding_is_removed_from_both_ends(tmp_path):
    path = write_wav(
        tmp_path / "a.wav", lead_silence_ms=120, tone_ms=500, trail_silence_ms=240
    )
    before = duration_ms(path)
    result = trim_edge_silence(path)

    assert result.trimmed
    assert result.removed_ms > 250, f"only removed {result.removed_ms:.0f}ms"
    assert duration_ms(path) < before


def test_a_little_silence_is_deliberately_left(tmp_path):
    """Speech starting on the very first sample sounds clipped, and a hard cut
    at the end swallows the release of a final consonant."""
    path = write_wav(tmp_path / "a.wav", lead_silence_ms=200, tone_ms=400)
    trim_edge_silence(path)
    # The tone survives, plus roughly KEEP_MS at the front.
    assert duration_ms(path) >= 400 + KEEP_MS - 5


def test_the_audio_itself_is_never_touched(tmp_path):
    path = write_wav(tmp_path / "a.wav", lead_silence_ms=150, tone_ms=300,
                     trail_silence_ms=150)
    trim_edge_silence(path)
    remaining = duration_ms(path)
    assert remaining >= 300, f"cut into the speech: {remaining:.0f}ms left"


def test_trimming_is_idempotent(tmp_path):
    path = write_wav(tmp_path / "a.wav", lead_silence_ms=200, tone_ms=400,
                     trail_silence_ms=200)
    trim_edge_silence(path)
    after_once = duration_ms(path)
    second = trim_edge_silence(path)
    assert not second.trimmed or second.removed_ms < 1
    assert duration_ms(path) == pytest.approx(after_once, abs=1.0)


# -- the guards -------------------------------------------------------------


def test_a_file_with_no_padding_is_left_alone(tmp_path):
    path = write_wav(tmp_path / "a.wav", tone_ms=400)
    before = duration_ms(path)
    result = trim_edge_silence(path)
    assert duration_ms(path) == pytest.approx(before, abs=1.0)
    assert not result.trimmed or result.removed_ms < KEEP_MS + 5


def test_a_very_short_render_is_left_alone(tmp_path):
    """A reaction word is short enough that its padding is most of it."""
    path = write_wav(tmp_path / "a.wav", tone_ms=MIN_DURATION_MS / 2)
    before = duration_ms(path)
    result = trim_edge_silence(path)
    assert not result.trimmed
    assert result.reason == "too short to trim"
    assert duration_ms(path) == pytest.approx(before, abs=1.0)


def test_a_mostly_silent_render_is_left_intact(tmp_path):
    """Something is wrong with that render. Trimming it aggressively turns a
    subtle problem into a loud one and hides the evidence."""
    path = write_wav(
        tmp_path / "a.wav", lead_silence_ms=2000, tone_ms=100, trail_silence_ms=2000
    )
    before = duration_ms(path)
    result = trim_edge_silence(path)
    assert not result.trimmed
    assert result.reason == "would trim too much"
    assert duration_ms(path) == pytest.approx(before, abs=1.0)


def test_a_fully_silent_render_is_left_intact(tmp_path):
    path = write_wav(tmp_path / "a.wav", tone_ms=400, amplitude=0)
    result = trim_edge_silence(path)
    assert not result.trimmed
    assert result.reason == "silent render"


def test_a_stereo_or_wide_file_is_refused_rather_than_mangled(tmp_path):
    path = write_wav(tmp_path / "a.wav", tone_ms=400, channels=2)
    result = trim_edge_silence(path)
    assert not result.trimmed
    assert result.reason == "not mono 16-bit"


def test_an_unreadable_file_never_raises(tmp_path):
    """Losing an utterance over a cosmetic optimisation would be a bad trade."""
    path = tmp_path / "not-a-wav.wav"
    path.write_bytes(b"this is not a wav file")
    result = trim_edge_silence(path)
    assert not result.trimmed
    assert "unreadable" in result.reason


def test_a_missing_file_never_raises(tmp_path):
    result = trim_edge_silence(tmp_path / "absent.wav")
    assert not result.trimmed


# -- what it buys -----------------------------------------------------------


def test_trimming_removes_most_of_a_realistic_gap(tmp_path):
    """Measured Piper padding is ~50ms leading and ~140-240ms trailing. With
    prefetching removing the synthesis term, that padding is the whole
    remaining accidental gap."""
    lead = write_wav(tmp_path / "lead.wav", tone_ms=2800, trail_silence_ms=142)
    tail = write_wav(tmp_path / "tail.wav", lead_silence_ms=51, tone_ms=7600)

    before = 142 + 51
    a = trim_edge_silence(lead)
    b = trim_edge_silence(tail)
    after = (142 - a.removed_tail_ms) + (51 - b.removed_head_ms) + 2 * KEEP_MS

    assert after < before
    assert after < 100, f"{after:.0f}ms of padding still unaccounted for"
