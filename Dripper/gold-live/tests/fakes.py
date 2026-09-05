"""Deterministic stand-ins for the hardware and services this system needs.

The project already had mocks -- MockTTS, FileTTS, MockCommentSource -- but
every one of them only models success. Nothing could express "the device
disappeared", "synthesis took nine seconds", "the model returned an empty
string", so none of the failure paths were reachable from a test. That is why
the barge-in bug and the swallowed playback exception survived 490 tests.

Failure is a parameter here, not a subclass. One FakeTTS configured different
ways covers delay, exception, silence and malformed output, which keeps the
call sites readable and makes it obvious what is being simulated.

Everything is synchronous-deterministic unless a delay is explicitly asked
for: no real sleeps beyond what a test requests, no wall-clock dependence, no
hardware.
"""

from __future__ import annotations

import asyncio
import wave
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

from platform_.tts.base import TTSProvider, TTSResult
from shared.contracts import AudioRequest, Priority


# -- audio -----------------------------------------------------------------


class FakeSink:
    """An audio device that records what it was asked to play.

    Models the behaviours that actually bite: a device that fails, one that
    takes real time (so an interruption has a window to land in), and one that
    hangs. `stop()` interrupts a play in progress, as the real sink does.
    """

    def __init__(
        self,
        fail: bool = False,
        fail_after: int | None = None,
        delay_s: float = 0.0,
        hang: bool = False,
    ) -> None:
        self.played: list[str] = []
        self.fail = fail
        self.fail_after = fail_after
        self.delay_s = delay_s
        self.hang = hang
        self.stop_calls = 0
        self.last_error: str | None = None
        self._stop = asyncio.Event()

    async def play(self, path: Path) -> None:
        self._stop.clear()
        if self.fail or (self.fail_after is not None
                         and len(self.played) >= self.fail_after):
            self.last_error = "fake device failure"
            raise OSError("fake device failure")

        if self.hang:
            await asyncio.Event().wait()  # never returns

        if self.delay_s:
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.delay_s)
                return  # interrupted mid-play
            except asyncio.TimeoutError:
                pass
        self.played.append(path.name)

    def stop(self) -> None:
        self.stop_calls += 1
        self._stop.set()


class FakeTTS(TTSProvider):
    """Synthesis with configurable timing and failure.

    `silent` and `malformed` matter because both produce a file: the readiness
    gate learned the hard way that "a wav exists" is not the same as "the host
    said something".
    """

    def __init__(
        self,
        delay_s: float = 0.0,
        fail: bool = False,
        fail_on: int | None = None,
        silent: bool = False,
        malformed: bool = False,
        empty: bool = False,
    ) -> None:
        self.delay_s = delay_s
        self.fail = fail
        self.fail_on = fail_on
        self.silent = silent
        self.malformed = malformed
        self.empty = empty
        self.calls: list[str] = []

    async def synthesize(self, text: str, voice_id: str, out_path: Path) -> TTSResult:
        n = len(self.calls)
        self.calls.append(text)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail or (self.fail_on is not None and n == self.fail_on):
            raise RuntimeError("fake TTS failure")
        if self.empty:
            return TTSResult(path=None, duration_ms=0, first_audio_ms=0)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.malformed:
            out_path.write_bytes(b"this is not a wav file")
            return TTSResult(path=out_path, duration_ms=1000, first_audio_ms=10)

        amplitude = 0 if self.silent else 12000
        _write_wav(out_path, seconds=max(0.2, len(text.split()) / 165 * 60),
                   amplitude=amplitude)
        return TTSResult(path=out_path, duration_ms=1000, first_audio_ms=10)


def _write_wav(path: Path, seconds: float, amplitude: int, rate: int = 22050) -> None:
    import struct

    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", amplitude) for _ in range(frames)))


# -- language model --------------------------------------------------------


@dataclass
class FakeLLMResult:
    text: str
    model: str = "fake:test"
    total_ms: int = 10
    prompt_tokens: int | None = 100
    completion_tokens: int | None = 40
    prefix_cached: bool = False


class FakeLLM:
    """A model server with the failure modes that actually occur.

    `repeat` exists because the real failure is not that the model errors --
    it is that it says nearly the same thing over and over, and roughly three
    quarters of production output is rejected for exactly that.
    """

    def __init__(
        self,
        replies: list[str] | None = None,
        repeat: str | None = None,
        delay_s: float = 0.0,
        fail: bool = False,
        timeout: bool = False,
        empty: bool = False,
        huge: bool = False,
        model: str = "fake:test",
    ) -> None:
        self.replies = list(replies or [])
        self.repeat = repeat
        self.delay_s = delay_s
        self.fail = fail
        self.timeout = timeout
        self.empty = empty
        self.huge = huge
        self.model = model
        self.name = f"local:{model}"
        self.read_timeout_s = 15.0
        self.total_timeout_s = 60.0
        self.calls = 0
        self.closed = False

    async def complete(self, messages, **kw) -> FakeLLMResult:
        self.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.timeout:
            raise asyncio.TimeoutError
        if self.fail:
            raise ConnectionError("fake model server is unreachable")
        if self.empty:
            return FakeLLMResult(text="")
        if self.huge:
            return FakeLLMResult(text=" ".join(["word"] * 20_000))
        if self.repeat is not None:
            return FakeLLMResult(text=self.repeat)
        if self.replies:
            return FakeLLMResult(text=self.replies[(self.calls - 1) % len(self.replies)])
        return FakeLLMResult(text="A steady, unremarkable observation about the market.")

    async def stream(self, messages, **kw):
        result = await self.complete(messages, **kw)
        for word in result.text.split():
            yield word + " "

    async def health(self) -> bool:
        return not self.fail

    async def first_model(self) -> str | None:
        return None if self.fail else self.model

    async def close(self) -> None:
        self.closed = True


# -- time ------------------------------------------------------------------


@dataclass
class FakeClock:
    """Controllable time, so a soak does not need to take a soak's worth of
    wall clock. Only useful where the code under test accepts an injected
    `now`, which the Director and market engine deliberately do."""

    now: float = 0.0
    _sleeps: list[float] = field(default_factory=list)

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now

    async def sleep(self, seconds: float) -> None:
        self._sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)

    @property
    def total_slept(self) -> float:
        return sum(self._sleeps)


# -- process ---------------------------------------------------------------


class FakeProcess:
    """A child process the supervisor can manage without spawning one.

    `hang` models the case supervision usually misses: a process that is very
    much alive and no longer doing its job.
    """

    def __init__(self, pid: int = 4242, exits_after: int | None = None,
                 returncode: int = 0, hang: bool = False) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self._exits_after = exits_after
        self._polls = 0
        self._final_returncode = returncode
        self.hang = hang
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        self._polls += 1
        if self._exits_after is not None and self._polls >= self._exits_after:
            self.returncode = self._final_returncode
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if not self.hang:
            self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0


# -- convenience -----------------------------------------------------------


def audio_request(
    segments: list[str] | None = None,
    session_id: str = "SESSION_001",
    priority: Priority = Priority.LOW,
    utterance_id: UUID | None = None,
    **kw,
) -> AudioRequest:
    return AudioRequest(
        session_id=session_id,
        utterance_id=utterance_id or uuid4(),
        trace_id=kw.pop("trace_id", "trace-test"),
        segments=segments or ["one", "two", "three"],
        voice_id=kw.pop("voice_id", "fake-voice"),
        priority=priority,
        **kw,
    )
