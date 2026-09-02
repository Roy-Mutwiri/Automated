"""Piper TTS -- local, fast, no network.

Fits the local-first architecture: runs on the same machine as the model, has
no per-character cost, and no third party sees the stream's content. Quality is
below the best hosted voices but it is genuinely usable, and for a 24/7 stream
across seven sessions the economics are not close.

Chosen on time-to-first-audio, which is the number that matters. Total
synthesis time does not, because segments are rendered per sentence while the
model is still writing the next one.

Voices: https://github.com/rhasspy/piper/blob/master/VOICES.md
Each voice is a .onnx plus a .onnx.json config. Give each session a different
voice -- seven identical voices across seven accounts is its own problem.

    piper --model en_GB-alba-medium.onnx --output_file out.wav

If Piper is not good enough once you hear it, XTTS or StyleTTS2 are the next
step up and slot in behind the same TTSProvider interface.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import wave
from pathlib import Path

from platform_.tts.base import TTSProvider, TTSResult

log = logging.getLogger(__name__)


class PiperTTS(TTSProvider):
    def __init__(
        self,
        voices_dir: str | Path = "voices",
        binary: str = "piper",
        default_voice: str = "en_GB-alba-medium",
        length_scale: float = 1.0,
        sentence_silence: float = 0.15,
    ) -> None:
        self.voices_dir = Path(voices_dir)
        self.binary = binary
        self.default_voice = default_voice
        #: >1 slows speech down. Live commentary usually wants slightly faster
        #: than the default, but not so fast it sounds clipped.
        self.length_scale = length_scale
        self.sentence_silence = sentence_silence

    def _model_path(self, voice_id: str) -> Path:
        name = voice_id if voice_id.endswith(".onnx") else f"{voice_id}.onnx"
        path = self.voices_dir / name
        if path.exists():
            return path
        fallback = self.voices_dir / f"{self.default_voice}.onnx"
        if fallback.exists():
            log.warning("voice %r not found, using %s", voice_id, self.default_voice)
            return fallback
        raise FileNotFoundError(
            f"No voice model for {voice_id!r} in {self.voices_dir}. "
            "Download one from https://github.com/rhasspy/piper/blob/master/VOICES.md"
        )

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    async def synthesize(self, text: str, voice_id: str, out_path: Path) -> TTSResult:
        if not text.strip():
            return TTSResult(path=None, duration_ms=0, first_audio_ms=0)

        model = self._model_path(voice_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        proc = await asyncio.create_subprocess_exec(
            self.binary,
            "--model", str(model),
            "--output_file", str(out_path),
            "--length_scale", str(self.length_scale),
            "--sentence_silence", str(self.sentence_silence),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate(text.encode("utf-8"))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if proc.returncode != 0:
            raise RuntimeError(
                f"piper failed ({proc.returncode}): {stderr.decode(errors='replace')[:300]}"
            )

        duration_ms = elapsed_ms
        try:
            with wave.open(str(out_path), "rb") as w:
                duration_ms = int(w.getnframes() / w.getframerate() * 1000)
        except (wave.Error, OSError):
            pass

        return TTSResult(
            path=out_path,
            duration_ms=duration_ms,
            # Piper renders the whole segment before returning, so first audio
            # is available when the file lands. Segments are per sentence, so
            # this is still ~200-400ms rather than a whole utterance.
            first_audio_ms=elapsed_ms,
        )
