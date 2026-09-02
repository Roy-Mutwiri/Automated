"""TTS provider interface.

The conversation engine outputs text. This turns text into audio. It knows
nothing about Gold, markets, or why the words were chosen -- keeping that
boundary is what makes the vendor swappable, and you will swap it.

Choose a vendor on measured time-to-first-audio, not on MOS scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TTSResult:
    path: Path | None
    duration_ms: int
    #: Time to first audible sample. This is the number that matters for the
    #: latency budget -- total synthesis time does not, because we stream.
    first_audio_ms: int


class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, text: str, voice_id: str, out_path: Path) -> TTSResult:
        """Render one segment. Called per sentence, not per utterance."""
