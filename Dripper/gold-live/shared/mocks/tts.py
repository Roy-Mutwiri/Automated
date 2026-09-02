"""TTS mocks. FileTTS writes real audio so M1 output can actually be listened to."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from platform_.tts.base import TTSProvider, TTSResult


class MockTTS(TTSProvider):
    """Records what it was asked to say. No audio. Used by tests."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str, voice_id: str, out_path: Path) -> TTSResult:
        self.spoken.append(text)
        return TTSResult(path=None, duration_ms=len(text) * 55, first_audio_ms=5)


class FileTTS(TTSProvider):
    """Writes a placeholder WAV of the right duration, plus the text as a sidecar.

    This is not speech -- it is a stand-in that proves the audio path, timing and
    file plumbing all work before a real vendor is wired in at M4. Duration is
    estimated from a realistic speaking rate so the Director's timing logic gets
    exercised honestly.
    """

    WORDS_PER_MINUTE = 165
    SAMPLE_RATE = 24_000

    async def synthesize(self, text: str, voice_id: str, out_path: Path) -> TTSResult:
        words = max(1, len(text.split()))
        duration_s = (words / self.WORDS_PER_MINUTE) * 60
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n = int(duration_s * self.SAMPLE_RATE)
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.SAMPLE_RATE)
            # Quiet tone rather than silence, so a muted output device is
            # distinguishable from a working one during manual checks.
            frames = b"".join(
                struct.pack("<h", int(1200 * math.sin(2 * math.pi * 220 * i / self.SAMPLE_RATE)))
                for i in range(0, n, 1)
            )
            w.writeframes(frames)

        out_path.with_suffix(".txt").write_text(text, encoding="utf-8")
        return TTSResult(
            path=out_path, duration_ms=int(duration_s * 1000), first_audio_ms=120
        )
