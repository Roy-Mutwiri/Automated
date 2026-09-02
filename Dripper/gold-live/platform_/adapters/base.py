"""Platform adapter interface.

Implementations planned:
  ScreenCaptureAdapter -- OCR of the LIVE Studio comment panel (runs on device)
  YouTubeAdapter       -- official liveChatMessages API (the hedge)
  MockAdapter          -- scripted, for tests and the dry run

Nothing above this interface knows where comments came from. That is the whole
point: an adapter that reads pixels and one that reads an API produce identical
CommentEvents, so an account-level problem on one platform is survivable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from shared.contracts import CommentEvent, ServiceHealth


class PlatformAdapter(ABC):
    #: Set at construction. Every CommentEvent this adapter emits carries it,
    #: stamped before any processing. Isolation layer 3.
    session_id: str

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def comments(self) -> AsyncIterator[CommentEvent]:
        """Yield normalised comments until disconnected."""

    @abstractmethod
    async def health(self) -> ServiceHealth: ...

    async def disconnect(self) -> None:  # pragma: no cover - trivial
        return None
