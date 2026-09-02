"""Piper TTS -- local, no network, no per-character cost.

Fits the local-first architecture: runs on the same machine as the model, has
no per-use bill, and no third party sees the stream's content.

IN-PROCESS, NOT A SUBPROCESS PER SEGMENT. This started as a subprocess call and
measured 4.5 seconds of wall time for 2 seconds of audio -- 2.2x realtime, with
no improvement across repeated calls. The cost was reloading a 60-114MB ONNX
model on every single utterance. Loading each voice once and holding it removes
that entirely.

The synthesis call is CPU-bound and blocking, so it runs in a thread rather than
on the event loop. A session's whole job is reacting quickly; a blocked loop
means comments stop being read while the host is mid-sentence.

Voices: install with `python -m scripts.get_voices`, which also audits each
MODEL_CARD. Piper's repository is MIT, but that covers the code -- each voice
inherits the terms of the corpus it was trained on, and several English voices
turn out to be non-commercial or research-derived. That matters when the build
is shared, so the shipped personas use public-domain voices only.

Give each session a different voice; seven identical voices across seven
accounts is its own problem.
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
        default_voice: str = "en_US-ljspeech-high",
        length_scale: float | None = None,
        binary: str = "piper",
        use_cuda: bool = False,
    ) -> None:
        self.voices_dir = Path(voices_dir)
        self.default_voice = default_voice
        #: >1 slows speech. Live commentary wants slightly faster than default,
        #: but not so fast it clips.
        self.length_scale = length_scale
        self.binary = binary
        self.use_cuda = use_cuda
        self._voices: dict[str, object] = {}

    # -- voice files ------------------------------------------------------

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
            "Install with: python -m scripts.get_voices"
        )

    def available(self) -> bool:
        """Can this actually speak? The library or the binary will do."""
        try:
            import piper  # noqa: F401

            return True
        except ImportError:
            return shutil.which(self.binary) is not None

    # -- loading ----------------------------------------------------------

    def _load(self, voice_id: str):
        """Load a voice once and keep it. This is the whole latency fix."""
        path = self._model_path(voice_id)
        key = str(path)
        cached = self._voices.get(key)
        if cached is not None:
            return cached

        from piper import PiperVoice

        started = time.perf_counter()
        voice = PiperVoice.load(path, use_cuda=self.use_cuda)
        self._voices[key] = voice
        log.info(
            "loaded voice %s in %.0fms (held for the life of the process)",
            path.stem, (time.perf_counter() - started) * 1000,
        )
        return voice

    def warmup(self, voice_ids: list[str]) -> None:
        """Load voices before going live.

        Otherwise the first utterance of a stream pays the model-load cost --
        seconds -- and that is the utterance an audience is most likely to be
        waiting on.
        """
        for voice_id in voice_ids:
            try:
                self._load(voice_id)
            except (FileNotFoundError, ImportError) as exc:
                log.warning("could not preload %s: %s", voice_id, exc)

    # -- synthesis --------------------------------------------------------

    def _render(self, text: str, voice_id: str, out_path: Path) -> int:
        """Blocking; called in a thread. Returns audio duration in ms."""
        from piper import SynthesisConfig

        voice = self._load(voice_id)
        config = (
            SynthesisConfig(length_scale=self.length_scale)
            if self.length_scale is not None
            else None
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as wav:
            voice.synthesize_wav(text, wav, syn_config=config)

        with wave.open(str(out_path), "rb") as wav:
            return int(wav.getnframes() / wav.getframerate() * 1000)

    async def synthesize(self, text: str, voice_id: str, out_path: Path) -> TTSResult:
        if not text.strip():
            return TTSResult(path=None, duration_ms=0, first_audio_ms=0)

        started = time.perf_counter()
        try:
            # to_thread keeps the event loop free: a blocked loop means comments
            # stop being read while the host is mid-sentence.
            duration_ms = await asyncio.to_thread(self._render, text, voice_id, out_path)
        except ImportError:
            duration_ms = await self._render_subprocess(text, voice_id, out_path)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if duration_ms and elapsed_ms > duration_ms:
            # Slower than realtime means the host cannot keep up with itself.
            log.warning(
                "TTS slower than realtime: %dms to render %dms of audio (%s)",
                elapsed_ms, duration_ms, voice_id,
            )
        return TTSResult(path=out_path, duration_ms=duration_ms, first_audio_ms=elapsed_ms)

    async def _render_subprocess(self, text: str, voice_id: str, out_path: Path) -> int:
        """Fallback for installs with the binary but not the library.

        Reloads the model per call, so it is markedly slower. Kept only so a
        partial install degrades rather than fails outright.
        """
        model = self._model_path(voice_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        args = [self.binary, "--model", str(model), "--output_file", str(out_path)]
        if self.length_scale is not None:
            args += ["--length_scale", str(self.length_scale)]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _out, err = await proc.communicate(text.encode("utf-8"))
        if proc.returncode != 0:
            raise RuntimeError(
                f"piper failed ({proc.returncode}): {err.decode(errors='replace')[:300]}"
            )
        try:
            with wave.open(str(out_path), "rb") as wav:
                return int(wav.getnframes() / wav.getframerate() * 1000)
        except (wave.Error, OSError):
            return 0
