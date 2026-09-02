"""Event bus.

Two implementations behind one interface:

  InMemoryBus  -- single process, no dependencies. Used by the M1 dry run
                  and by every test.
  RedisBus     -- Redis Streams. Consumer groups and replay, which is what
                  lets us answer "why did Session 4 say that?" after the fact.
                  Plain pub/sub would lose messages on restart.

Session isolation note: subscribe() filters on session_id. A session runtime
is constructed with its own id and can only ever receive its own events plus
shared-plane events (session_id=None).
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from typing import Any

from shared.contracts import Envelope, utcnow

log = logging.getLogger(__name__)

Handler = Callable[[Envelope], Any]


class EventBus(ABC):
    @abstractmethod
    async def publish(self, env: Envelope) -> None: ...

    @abstractmethod
    def subscribe(
        self, event_types: list[str], session_id: str | None = None
    ) -> AsyncIterator[Envelope]: ...

    async def close(self) -> None:  # pragma: no cover - trivial
        return None


class InMemoryBus(EventBus):
    """Dependency-free bus. Fan-out to every matching subscriber queue."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._subs: list[tuple[set[str], str | None, asyncio.Queue[Envelope]]] = []
        self._maxsize = maxsize
        self.published: list[Envelope] = []  # inspected by tests

    async def publish(self, env: Envelope) -> None:
        self.published.append(env)
        for types, sess, q in self._subs:
            if env.event_type not in types:
                continue
            # Shared-plane events (session_id=None) reach everyone; session
            # events reach only that session. This is isolation layer 2.
            if env.session_id is not None and sess is not None and env.session_id != sess:
                continue
            try:
                q.put_nowait(env)
            except asyncio.QueueFull:
                log.warning("bus queue full, dropping %s", env.event_type)

    async def subscribe(  # type: ignore[override]
        self, event_types: list[str], session_id: str | None = None
    ) -> AsyncIterator[Envelope]:
        q: asyncio.Queue[Envelope] = asyncio.Queue(maxsize=self._maxsize)
        self._subs.append((set(event_types), session_id, q))
        while True:
            yield await q.get()


class RedisBus(EventBus):
    """Redis Streams. One stream per event type, consumer group per session."""

    def __init__(self, url: str, group: str, consumer: str) -> None:
        self._url = url
        self._group = group
        self._consumer = consumer
        self._redis: Any = None

    async def _conn(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self._url, decode_responses=True)
        return self._redis

    @staticmethod
    def _stream(event_type: str) -> str:
        return f"stream:{event_type}"

    async def publish(self, env: Envelope) -> None:
        r = await self._conn()
        await r.xadd(
            self._stream(env.event_type),
            {"data": env.model_dump_json()},
            maxlen=10_000,
            approximate=True,
        )

    async def subscribe(  # type: ignore[override]
        self, event_types: list[str], session_id: str | None = None
    ) -> AsyncIterator[Envelope]:
        r = await self._conn()
        streams = {self._stream(t): ">" for t in event_types}
        for s in streams:
            try:
                await r.xgroup_create(s, self._group, id="0", mkstream=True)
            except Exception:  # group already exists
                pass
        while True:
            resp = await r.xreadgroup(
                self._group, self._consumer, streams, count=16, block=1000
            )
            for _stream_name, entries in resp or []:
                for entry_id, fields in entries:
                    env = Envelope.model_validate_json(fields["data"])
                    await r.xack(_stream_name, self._group, entry_id)
                    if (
                        env.session_id is not None
                        and session_id is not None
                        and env.session_id != session_id
                    ):
                        continue
                    yield env

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()


# --- event type constants -------------------------------------------------

MARKET_STATE_UPDATED = "market.state_updated"
MARKET_EVENT_DETECTED = "market.event_detected"
COMMENT_RECEIVED = "viewer.comment_received"
RESPONSE_GENERATED = "ai.response_generated"
AUDIO_REQUESTED = "audio.requested"
AUDIO_STARTED = "audio.playback_started"
AUDIO_COMPLETED = "audio.playback_completed"
SERVICE_HEALTH = "service.health"


def wrap(
    event_type: str,
    payload: dict[str, Any],
    trace_id: str,
    session_id: str | None = None,
    occurred_at: Any = None,
) -> Envelope:
    return Envelope(
        event_type=event_type,
        payload=payload,
        trace_id=trace_id,
        session_id=session_id,
        occurred_at=occurred_at or utcnow(),
    )


def dumps(obj: Any) -> dict[str, Any]:
    """Contract -> plain dict suitable for an Envelope payload."""
    return json.loads(obj.model_dump_json())
