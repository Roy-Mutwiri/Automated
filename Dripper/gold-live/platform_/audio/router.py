"""Audio router: queue, playback, and barge-in.

Owns everything between "here is an AudioRequest" and "sound came out". Knows
nothing about Gold, conversation, or why the words were chosen -- that boundary
is what keeps the TTS vendor swappable.

Three behaviours carry the design:

  Segment streaming   an utterance arrives as sentences. Segment 1 is
                      synthesised and played while segment 3 is still being
                      written, which is most of the perceived latency win.

  Barge-in            a CRITICAL utterance can cut off one already playing.
                      Everything below CRITICAL waits. Without that rule the
                      host talks over itself constantly, which is the most
                      bot-like failure there is.

  Deadlines           a queued utterance past its deadline is dropped, not
                      spoken. A reaction to a sweep that arrives forty seconds
                      late is worse than silence -- it makes the host sound
                      disconnected from the chart the audience is watching.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import UUID

from platform_.tts.base import TTSProvider
from shared.contracts import AudioRequest, Priority

log = logging.getLogger(__name__)


class PlaybackState(str, Enum):
    IDLE = "idle"
    SYNTHESISING = "synthesising"
    PLAYING = "playing"
    STOPPED = "stopped"


@dataclass
class RouterStats:
    queued: int = 0
    played: int = 0
    preempted: int = 0
    dropped_expired: int = 0
    tts_failures: int = 0
    underruns: int = 0
    first_audio_ms: list[float] = field(default_factory=list)

    @property
    def median_first_audio_ms(self) -> float:
        if not self.first_audio_ms:
            return 0.0
        s = sorted(self.first_audio_ms)
        return s[len(s) // 2]


class AudioSink:
    """Plays a wav to a specific output device. Replaceable for tests."""

    def __init__(self, device_index: int | None = None) -> None:
        self.device_index = device_index
        self._stop = asyncio.Event()

    async def play(self, path: Path) -> None:
        self._stop.clear()
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            # No audio backend: behave as if played, so the rest of the
            # pipeline can be exercised on a machine with no sound card.
            log.debug("no audio backend; simulating playback of %s", path.name)
            await asyncio.sleep(0)
            return

        data, rate = sf.read(str(path), dtype="float32")
        sd.play(data, rate, device=self.device_index)
        try:
            while sd.get_stream().active:
                if self._stop.is_set():
                    sd.stop()
                    return
                await asyncio.sleep(0.02)
        except Exception:
            pass

    def stop(self) -> None:
        self._stop.set()


class AudioRouter:
    """One per session."""

    def __init__(
        self,
        session_id: str,
        tts: TTSProvider,
        out_dir: Path,
        sink: AudioSink | None = None,
        max_queue: int = 8,
    ) -> None:
        self.session_id = session_id
        self.tts = tts
        self.out_dir = out_dir / session_id
        self.sink = sink or AudioSink()
        self.state = PlaybackState.IDLE
        self.stats = RouterStats()

        self._queue: asyncio.Queue[AudioRequest] = asyncio.Queue(maxsize=max_queue)
        self._current: AudioRequest | None = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._seq = 0

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._drain())

    async def stop(self) -> None:
        self._running = False
        self.sink.stop()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self.state = PlaybackState.STOPPED

    # -- submission -------------------------------------------------------

    async def submit(self, request: AudioRequest) -> bool:
        if request.session_id != self.session_id:
            raise ValueError(
                f"audio for {request.session_id} submitted to router for {self.session_id}"
            )

        if request.priority is Priority.CRITICAL and self._current is not None:
            log.info(
                "[%s] barge-in: %s preempts %s",
                self.session_id, request.utterance_id, self._current.utterance_id,
            )
            self.stats.preempted += 1
            self.sink.stop()

        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            log.warning("[%s] audio queue full, dropping utterance", self.session_id)
            return False
        self.stats.queued += 1
        return True

    # -- playback ---------------------------------------------------------

    async def _drain(self) -> None:
        while self._running:
            try:
                request = await self._queue.get()
            except asyncio.CancelledError:
                return

            if request.expired():
                # Better silence than commentary about a moment that has passed.
                self.stats.dropped_expired += 1
                log.info(
                    "[%s] dropped expired utterance %s (deadline %dms)",
                    self.session_id, request.utterance_id, request.deadline_ms,
                )
                continue

            self._current = request
            try:
                await self._speak(request)
                self.stats.played += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self.stats.tts_failures += 1
                log.exception("[%s] playback failed", self.session_id)
            finally:
                self._current = None
                self.state = PlaybackState.IDLE

    async def _speak(self, request: AudioRequest) -> None:
        self._seq += 1
        for i, segment in enumerate(request.segments):
            if not self._running:
                return
            self.state = PlaybackState.SYNTHESISING
            path = (
                self.out_dir
                / f"{self._seq:05d}_{request.utterance_id.hex[:8]}_{i:02d}.wav"
            )
            result = await self.tts.synthesize(segment, request.voice_id, path)
            if i == 0:
                self.stats.first_audio_ms.append(result.first_audio_ms)

            if result.path is None:
                continue
            self.state = PlaybackState.PLAYING
            await self.sink.play(result.path)

    # -- introspection ----------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        return self._current is not None

    @property
    def current_utterance(self) -> UUID | None:
        return self._current.utterance_id if self._current else None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "queue_depth": self.queue_depth,
            "speaking": self.is_speaking,
            "played": self.stats.played,
            "preempted": self.stats.preempted,
            "dropped_expired": self.stats.dropped_expired,
            "tts_failures": self.stats.tts_failures,
            "median_first_audio_ms": self.stats.median_first_audio_ms,
            "at": datetime.now(timezone.utc).isoformat(),
        }
