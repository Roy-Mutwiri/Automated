"""Candle aggregation from a tick stream.

Deliberately boring and deterministic. Everything downstream -- structure,
detections, what the host says -- rests on these being right, so this module
has no cleverness in it and no dependency on anything above it.

Bucketing rule: a candle covers [t, t+period). A tick lands in the bucket
floor(epoch / period). That means buckets align to the epoch, so the 5m candle
boundaries are :00 :05 :10 and not "five minutes after whenever we started",
which is what you want when comparing against anyone else's chart.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(slots=True)
class Candle:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    #: False until the period has elapsed. Detectors must not fire on a
    #: forming candle -- a "break" that un-breaks before the close is noise,
    #: and reacting to it out loud is how a host loses credibility.
    closed: bool = False

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def bullish(self) -> bool:
        return self.close >= self.open


class CandleSeries:
    """A fixed-size rolling window of candles for one timeframe."""

    def __init__(self, period_s: int, maxlen: int = 500) -> None:
        self.period_s = period_s
        self.candles: deque[Candle] = deque(maxlen=maxlen)

    def _bucket(self, at: datetime) -> datetime:
        epoch = int(at.timestamp())
        return datetime.fromtimestamp(
            epoch - (epoch % self.period_s), tz=timezone.utc
        )

    def update(self, price: float, at: datetime, volume: float = 0.0) -> Candle | None:
        """Fold one tick in. Returns a candle if this tick closed the previous one."""
        bucket = self._bucket(at)
        closed: Candle | None = None

        if self.candles and self.candles[-1].open_time == bucket:
            c = self.candles[-1]
            c.high = max(c.high, price)
            c.low = min(c.low, price)
            c.close = price
            c.volume += volume
            return None

        if self.candles:
            prev = self.candles[-1]
            if not prev.closed:
                prev.closed = True
                closed = prev

        self.candles.append(
            Candle(
                open_time=bucket, open=price, high=price, low=price,
                close=price, volume=volume,
            )
        )
        return closed

    def mark_stale_closed(self, now: datetime) -> Candle | None:
        """Close the trailing candle when its period has elapsed but no tick
        has arrived to open the next one. Without this, a quiet market leaves
        the last candle forming forever and no detector ever fires."""
        if not self.candles:
            return None
        last = self.candles[-1]
        if last.closed:
            return None
        if now >= last.open_time + timedelta(seconds=self.period_s):
            last.closed = True
            return last
        return None

    @property
    def closed_candles(self) -> list[Candle]:
        return [c for c in self.candles if c.closed]

    @property
    def last(self) -> Candle | None:
        return self.candles[-1] if self.candles else None

    def __len__(self) -> int:
        return len(self.candles)
