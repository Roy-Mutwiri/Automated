"""Market data feeds.

The engine consumes ticks and does not care where they come from. Three
implementations:

  ReplayFeed     a recorded session from CSV, replayed at any speed. This is
                 what soak tests and detector development run against -- a
                 recorded London session is reproducible in a way a live feed
                 never is.
  WebSocketFeed  a generic JSON websocket with reconnect and backoff.
  SyntheticFeed  a random walk, for smoke tests with no data at all.

LICENSING, before this is pointed at a real provider: most XAUUSD feeds licence
for INTERNAL DISPLAY only. Broadcasting a live price to a public audience is
redistribution and usually needs a different, more expensive tier. Check the
contract before launch, not after.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Tick:
    bid: float
    ask: float
    at: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


class Feed(ABC):
    name: str

    @abstractmethod
    def ticks(self) -> AsyncIterator[Tick]: ...

    async def close(self) -> None:  # pragma: no cover - trivial
        return None


class ReplayFeed(Feed):
    """Replay a CSV of ticks. Columns: timestamp,bid,ask

    `speed` multiplies real time: 1.0 replays at the original pace, 60.0 runs
    an hour a minute, 0 replays as fast as the consumer can take it.
    """

    name = "replay"

    def __init__(self, path: str | Path, speed: float = 0.0) -> None:
        self.path = Path(path)
        self.speed = speed

    async def ticks(self) -> AsyncIterator[Tick]:  # type: ignore[override]
        previous: datetime | None = None
        with self.path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                at = datetime.fromisoformat(row["timestamp"])
                if at.tzinfo is None:
                    at = at.replace(tzinfo=timezone.utc)
                if self.speed > 0 and previous is not None:
                    delay = (at - previous).total_seconds() / self.speed
                    if delay > 0:
                        await asyncio.sleep(min(delay, 5.0))
                previous = at
                yield Tick(bid=float(row["bid"]), ask=float(row["ask"]), at=at)


class SyntheticFeed(Feed):
    """Random walk with occasional volatility bursts. Smoke tests only."""

    name = "synthetic"

    def __init__(
        self,
        start_price: float = 3652.40,
        interval_s: float = 1.0,
        spread: float = 0.36,
        seed: int = 5,
    ) -> None:
        self.price = start_price
        self.interval_s = interval_s
        self.spread = spread
        self.rng = random.Random(seed)
        self._running = True

    async def ticks(self) -> AsyncIterator[Tick]:  # type: ignore[override]
        now = datetime.now(timezone.utc)
        while self._running:
            burst = self.rng.random() < 0.03
            step = self.rng.gauss(0, 0.9 if burst else 0.18)
            self.price = round(max(1.0, self.price + step), 2)
            now += timedelta(seconds=self.interval_s)
            yield Tick(
                bid=round(self.price - self.spread / 2, 2),
                ask=round(self.price + self.spread / 2, 2),
                at=now,
            )
            # Must actually wait. Sleeping 0 advanced simulated market time by
            # hours per wall-clock second, which made every time-based cooldown
            # in the engine look like it had already elapsed.
            await asyncio.sleep(self.interval_s)

    async def close(self) -> None:
        self._running = False


class WebSocketFeed(Feed):
    """Generic JSON websocket feed with reconnect and exponential backoff.

    `field_map` adapts a provider's payload without subclassing, e.g.
    {"bid": "b", "ask": "a", "timestamp": "t"}.

    Reconnect policy matters more than throughput here. A dropped feed is not
    an error to raise -- the engine's staleness clock is already running, the
    host has already stopped quoting prices, and this just needs to keep trying
    quietly until it is back.
    """

    name = "websocket"

    def __init__(
        self,
        url: str,
        subscribe: dict | None = None,
        field_map: dict[str, str] | None = None,
        max_backoff_s: float = 60.0,
    ) -> None:
        self.url = url
        self.subscribe = subscribe
        self.field_map = field_map or {"bid": "bid", "ask": "ask", "timestamp": "timestamp"}
        self.max_backoff_s = max_backoff_s
        self._running = True
        self.reconnects = 0

    def _parse(self, raw: str) -> Tick | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        try:
            ts = data[self.field_map["timestamp"]]
            at = (
                datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                if isinstance(ts, (int, float))
                else datetime.fromisoformat(str(ts))
            )
            return Tick(
                bid=float(data[self.field_map["bid"]]),
                ask=float(data[self.field_map["ask"]]),
                at=at if at.tzinfo else at.replace(tzinfo=timezone.utc),
            )
        except (KeyError, TypeError, ValueError):
            return None

    async def ticks(self) -> AsyncIterator[Tick]:  # type: ignore[override]
        import websockets

        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(self.url) as ws:
                    log.info("market feed connected: %s", self.url)
                    backoff = 1.0
                    if self.subscribe:
                        await ws.send(json.dumps(self.subscribe))
                    async for raw in ws:
                        tick = self._parse(raw if isinstance(raw, str) else raw.decode())
                        if tick is not None:
                            yield tick
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect, never surface
                self.reconnects += 1
                log.warning(
                    "market feed dropped (%s); reconnecting in %.1fs", exc, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)

    async def close(self) -> None:
        self._running = False
