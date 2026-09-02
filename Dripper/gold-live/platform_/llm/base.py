"""LLM backend interface.

The conversation engine talks to this, never to a specific server or vendor.
The primary implementation is a locally hosted model on the central machine;
a hosted API backend exists for A/B comparison and as an emergency fallback,
not as the default.

Streaming is not optional. Utterances are consumed sentence by sentence so the
first segment reaches TTS while the model is still writing the third -- that is
worth roughly two seconds of perceived latency and it is the difference between
a host that responds and one that pauses awkwardly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMResult:
    text: str
    model: str
    first_token_ms: int | None = None
    total_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: True when the backend reported a cached prefix hit (vLLM prefix caching,
    #: llama.cpp slot reuse). Worth tracking -- it is most of the latency win.
    prefix_cached: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class LLMBackend(ABC):
    """Any chat-completion server. Implementations must stream."""

    name: str

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 300,
        temperature: float = 0.85,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive."""

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 300,
        temperature: float = 0.85,
        stop: list[str] | None = None,
    ) -> LLMResult:
        """Collect a full response. Used for non-latency-critical calls such as
        topic proposal and comment classification."""

    @abstractmethod
    async def health(self) -> bool:
        """Is the server up and holding a loaded model?"""

    async def close(self) -> None:  # pragma: no cover - trivial
        return None
