"""Audio router: queue, playback, and barge-in.

Owns everything between "here is an AudioRequest" and "sound came out". Knows
nothing about Gold, conversation, or why the words were chosen -- that boundary
is what keeps the TTS vendor swappable.

Three behaviours carry the design:

  Segment streaming   an utterance arrives as sentences. The first is
                      synthesised and played while the rest are still being
                      written, which is most of the perceived latency win.
                      Only the first, though -- see group_for_prosody.

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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import UUID

from platform_.audio.retention import (
    SWEEP_EVERY_UTTERANCES,
    AudioRetention,
    RetentionPolicy,
    SweepResult,
)
from platform_.audio.timing import trim_edge_silence
from platform_.tts.base import Prosody, TTSProvider
from shared.contracts import AudioRequest, Priority
from shared.delivery import GAP_MAX_MS, NULL_CAPABILITIES, DeliveryPlan

log = logging.getLogger(__name__)


class PlaybackState(str, Enum):
    IDLE = "idle"
    SYNTHESISING = "synthesising"
    PLAYING = "playing"
    STOPPED = "stopped"


#: Latency samples kept for the health statistic. Bounded because this used
#: to be a plain list: one float per utterance, never trimmed, and sorted in
#: full on every health snapshot. Small in bytes, but unbounded is unbounded,
#: and a median over the recent window is the more useful statistic anyway --
#: what matters is whether the host is keeping up *now*.
RETAINED_LATENCIES = 500

#: Sentences synthesised on their own before the rest are joined into one call.
#:
#: One, because the whole latency argument rests on getting *some* audio out
#: quickly, and that only needs the first sentence.
LEAD_SEGMENTS = 1


#: How long a deliberate silence is slept for at a time. Barge-in is checked
#: between slices, so this is the worst-case delay a planned pause can add to
#: an interruption -- half the 20ms the sink already takes to notice a stop.
PAUSE_SLICE_S = 0.01

#: The beat after a reaction word. "Wait." followed immediately by the
#: sentence is not a reaction, it is a first word; the silence is what makes
#: it read as the host noticing something before formulating the thought.
REACTION_GAP_MS = 200


@dataclass(frozen=True, slots=True)
class SpeechUnit:
    """One render, and the silence that follows it."""

    text: str
    gap_after_ms: int


def plan_units(plan: DeliveryPlan, segments: list[str]) -> list[SpeechUnit]:
    """What actually gets rendered, in order, with the silence between.

    A reaction is a separate unit rather than the first words of the sentence,
    because the pause after it is the entire point -- a human reacts before
    they have formulated what to say, and the gap is what conveys that.

    It is only ever prepended to real speech. A reaction with nothing behind
    it would be the host saying "Whoa." and falling silent, which is both
    strange and the shape a runaway reaction loop would take.
    """
    units = [
        SpeechUnit(text, plan.gap_ms)
        for text in group_for_prosody(segments)
        if text.strip()
    ]
    if plan.reaction and units:
        beat = min(GAP_MAX_MS, max(plan.gap_ms, REACTION_GAP_MS))
        units.insert(0, SpeechUnit(plan.reaction, beat))
    return units


def group_for_prosody(segments: list[str], lead: int = LEAD_SEGMENTS) -> list[str]:
    """Regroup sentences so the tail is synthesised as one continuous span.

    Splitting an utterance into sentences is right for *streaming* and wrong
    for *prosody*. A TTS call knows nothing of the text around it, so every
    sentence rendered alone gets its own complete intonation contour, ending
    in the pitch fall that means "end of statement". Measured on
    en_US-john-medium, a three-sentence utterance rendered per sentence fell
    ~40Hz at each of the first two boundaries; rendered as one call the
    contour carried across, ending within 5Hz of where it started. Three full
    stops in a row is what makes a host sound like three announcements rather
    than one thought.

    So: the first sentence alone, to start audio quickly, and everything after
    it as a single span. Barge-in is unaffected -- it stops the sink mid-file,
    not between segments -- so the longer tail costs nothing in interrupt
    latency.
    """
    if len(segments) <= lead:
        return list(segments)
    tail = " ".join(s.strip() for s in segments[lead:] if s.strip())
    head = list(segments[:lead])
    return [*head, tail] if tail else head


@dataclass
class RouterStats:
    queued: int = 0
    played: int = 0
    preempted: int = 0
    dropped_expired: int = 0
    tts_failures: int = 0
    underruns: int = 0
    first_audio_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=RETAINED_LATENCIES)
    )

    @property
    def median_first_audio_ms(self) -> float:
        if not self.first_audio_ms:
            return 0.0
        s = sorted(self.first_audio_ms)
        return s[len(s) // 2]


class AudioPlaybackError(RuntimeError):
    """Playback could not start. Raised so the router can mark itself
    unhealthy instead of reporting a silent utterance as spoken."""


class AudioSink:
    """Plays a wav to a specific output device. Replaceable for tests."""

    def __init__(self, device_index: int | None = None) -> None:
        self.device_index = device_index
        self.last_error: str | None = None
        self._stop = asyncio.Event()

    async def play(self, path: Path) -> None:
        self._stop.clear()
        try:
            import sounddevice as sd
            import soundfile as sf
        except (ImportError, OSError):
            # No audio backend: behave as if played, so the rest of the
            # pipeline can be exercised on a machine with no sound card.
            log.debug("no audio backend; simulating playback of %s", path.name)
            await asyncio.sleep(0)
            return

        try:
            data, rate = sf.read(str(path), dtype="float32")
            sd.play(data, rate, device=self.device_index)
        except Exception as exc:
            # A device that has gone away, an exclusive-mode lock, a wav the
            # decoder rejects. This used to be swallowed together with the
            # polling loop below, so audio could fail on every utterance and
            # nothing above ever heard about it.
            self.last_error = f"playback failed: {exc}"
            log.error("audio playback failed for %s: %s", path.name, exc)
            raise AudioPlaybackError(str(exc)) from exc

        try:
            while sd.get_stream().active:
                if self._stop.is_set():
                    sd.stop()
                    return
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                sd.stop()
            raise
        except Exception as exc:
            # get_stream() raises once playback has finished, which is the
            # normal way this loop ends -- so it is not an error on its own.
            # It is recorded rather than discarded so a device failing midway
            # is still visible.
            log.debug("playback poll ended for %s: %s", path.name, exc)

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
        retention: RetentionPolicy | None = None,
    ) -> None:
        self.session_id = session_id
        self.tts = tts
        self.out_dir = out_dir / session_id
        self.sink = sink or AudioSink()
        self.state = PlaybackState.IDLE
        self.stats = RouterStats()

        #: Spoken audio is kept for a bounded window and then reclaimed.
        #: Without this the directory grew forever: every utterance the
        #: system had ever spoken was still on disk.
        self.retention = AudioRetention(self.out_dir, retention)
        #: Files currently being written or read. Never evicted -- soundfile
        #: reads the whole wav into memory before playback starts, but it must
        #: still survive until that read has happened.
        self._in_use: set[Path] = set()
        self._completed = 0
        self.last_sweep: SweepResult | None = None

        self._queue: asyncio.Queue[AudioRequest] = asyncio.Queue(maxsize=max_queue)
        self._current: AudioRequest | None = None
        #: The utterance that has been preempted, if any. Checked between
        #: segments, because stopping the sink only aborts the segment that is
        #: sounding -- without this the loop moved on to the next one and the
        #: preempted utterance carried on talking over whatever interrupted it.
        #:
        #: One id, not a set. Only the utterance currently in flight can be
        #: preempted, and a set grew without bound: a mark added for an
        #: utterance that had already finished was never discarded, so under
        #: sustained barge-in it accumulated one entry per interruption.
        self._preempted: UUID | None = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._seq = 0

    # -- lifecycle --------------------------------------------------------

    def sweep_audio(self) -> SweepResult:
        """Enforce the retention policy now, protecting in-flight files.

        Synchronous and cheap: the directory is bounded, so this is a scandir
        over a few hundred entries. Deliberately not a background task -- a
        cleaner that can outlive the session it cleans is its own leak.
        """
        result = self.retention.sweep(protect=self._in_use)
        self.last_sweep = result
        if result.deleted or result.errors:
            log.info("[%s] spoken audio: %s", self.session_id, result)
        return result

    async def start(self) -> None:
        self._running = True
        # Reclaim anything left by a previous session: a hard-killed
        # supervisor cannot clean up after itself, so the next start does it.
        self.sweep_audio()
        self._task = asyncio.create_task(self._drain())

    async def stop(self) -> None:
        self._running = False
        self.sink.stop()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self.state = PlaybackState.STOPPED
        # Nothing is in flight once the drain task is cancelled, so a final
        # sweep can reclaim whatever the last utterances left behind.
        self._in_use.clear()
        self.sweep_audio()

    # -- submission -------------------------------------------------------

    async def submit(
        self, request: AudioRequest, delivery: DeliveryPlan | None = None
    ) -> bool:
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
            self._preempted = self._current.utterance_id
            self.sink.stop()

        try:
            self._queue.put_nowait((request, delivery))
        except asyncio.QueueFull:
            log.warning("[%s] audio queue full, dropping utterance", self.session_id)
            return False
        self.stats.queued += 1
        return True

    # -- playback ---------------------------------------------------------

    async def _drain(self) -> None:
        while self._running:
            try:
                request, delivery = await self._queue.get()
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
            if self._preempted == request.utterance_id:
                self._preempted = None
            try:
                await self._speak(request, delivery)
                self.stats.played += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self.stats.tts_failures += 1
                log.exception("[%s] playback failed", self.session_id)
            finally:
                if self._preempted == request.utterance_id:
                    self._preempted = None
                self._current = None
                self.state = PlaybackState.IDLE
                # Counted, not timed: every completed utterance moves the
                # sweep clock, whether it played, failed or was interrupted,
                # so no path can accumulate audio indefinitely.
                self._completed += 1
                if self._completed % SWEEP_EVERY_UTTERANCES == 0:
                    self.sweep_audio()

    async def _quiet(self, ms: int, request: AudioRequest) -> None:
        """Deliberate silence.

        Slept in slices so that a planned pause cannot delay a barge-in. The
        alternative -- one long sleep -- would mean a 320ms gap could hold an
        interruption for 320ms, which is exactly the latency this router
        exists to protect.
        """
        if ms <= 0:
            return
        remaining = ms / 1000
        while remaining > 0:
            if not self._running or self._preempted == request.utterance_id:
                return
            slice_s = min(PAUSE_SLICE_S, remaining)
            await asyncio.sleep(slice_s)
            remaining -= slice_s

    async def _speak(
        self, request: AudioRequest, delivery: DeliveryPlan | None = None
    ) -> None:
        """Render and play one utterance, with the next chunk rendered while
        the current one is still sounding.

        The prefetch is the whole reason a planned pause means anything.
        Rendering strictly in sequence put the synthesis time of the next
        chunk -- measured 336-385ms -- into the middle of every utterance, on
        top of the silence Piper pads onto each render. The listener heard
        ~562ms of dead air inside a single thought, and no pause the delivery
        plan asked for could be distinguished from it.

        Piper's real-time factor is ~0.05, so rendering the next chunk during
        playback of the current one has enormous headroom. What is left is
        trimmed padding plus exactly the silence that was asked for.
        """
        self._seq += 1
        plan = (delivery or DeliveryPlan.neutral()).renderable_with(
            getattr(self.tts, "capabilities", NULL_CAPABILITIES)
        )
        units = plan_units(plan, request.segments)
        if not units:
            return
        prosody = Prosody.for_utterance(request.utterance_id, plan.pace)

        #: Every path this utterance claimed, so a cancelled prefetch cannot
        #: leave a file pinned against the retention sweep forever.
        claimed: set[Path] = set()

        def aborted() -> bool:
            return not self._running or self._preempted == request.utterance_id

        async def render(index: int, text: str) -> tuple[Path, object]:
            path = (
                self.out_dir
                / f"{self._seq:05d}_{request.utterance_id.hex[:8]}_{index:02d}.wav"
            )
            # Claimed before synthesis and released only after playback, so a
            # sweep can never unlink a file that is still being written or
            # read -- including when synthesis or playback raises.
            claimed.add(path)
            self._in_use.add(path)
            result = await self.tts.synthesize(
                text, request.voice_id, path, prosody=prosody
            )
            if result.path is not None:
                # Removes the padding Piper adds at both ends, so the only
                # silence between chunks is the silence that was planned.
                trim_edge_silence(result.path)
            return path, result

        pending: asyncio.Task | None = None
        try:
            await self._quiet(plan.pause_before_ms, request)

            for i, unit in enumerate(units):
                if aborted():
                    # Barge-in. Abandon the rest of this utterance rather than
                    # finishing it underneath whatever preempted it.
                    log.info(
                        "[%s] abandoning preempted utterance %s at unit %d/%d",
                        self.session_id, request.utterance_id, i, len(units),
                    )
                    return

                if pending is None:
                    self.state = PlaybackState.SYNTHESISING
                    path, result = await render(i, unit.text)
                else:
                    path, result = await pending
                    pending = None

                if i == 0:
                    self.stats.first_audio_ms.append(result.first_audio_ms)

                # Start the next render now, so it is ready the moment this
                # one stops sounding.
                if i + 1 < len(units) and not aborted():
                    pending = asyncio.create_task(render(i + 1, units[i + 1].text))

                try:
                    if result.path is None:
                        continue
                    if aborted():
                        return
                    self.state = PlaybackState.PLAYING
                    await self.sink.play(result.path)
                finally:
                    self._in_use.discard(path)
                    # Per unit, not per utterance. A file cannot be evicted
                    # before it exists, so the directory always exceeds the cap
                    # by whatever is currently being written -- sweeping here
                    # makes that overshoot one unit instead of one utterance.
                    self.sweep_audio()

                if i + 1 < len(units):
                    await self._quiet(unit.gap_after_ms, request)

            await self._quiet(plan.pause_after_ms, request)
        finally:
            if pending is not None:
                # An in-flight prefetch for audio that will never be played.
                pending.cancel()
                with contextlib.suppress(BaseException):
                    await pending
            for path in claimed:
                self._in_use.discard(path)
            self.sweep_audio()

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
        sweep = self.last_sweep
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
            # Reported so unbounded disk shows up in health rather than only
            # in a postmortem of a full drive.
            "audio_files": sweep.kept if sweep else 0,
            "audio_bytes": sweep.kept_bytes if sweep else 0,
            "at": datetime.now(timezone.utc).isoformat(),
        }
