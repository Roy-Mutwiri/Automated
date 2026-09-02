"""Deterministic market calculations.

Pure functions over closed candles. No LLM anywhere near this file -- these are
measurements, and a model that "mostly" gets a swing high right is worse than
useless because the host will state it with confidence.

Everything here returns observations, never interpretations. "The 5m range is
4.20" belongs here. "Buyers are in control" does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone

from platform_.market.candles import Candle
from shared.contracts import Structure, TradingSession, Trend


@dataclass(slots=True)
class Swing:
    index: int
    price: float
    at: datetime
    kind: str  # "high" | "low"


def true_range(current: Candle, previous: Candle | None) -> float:
    if previous is None:
        return current.range
    return max(
        current.high - current.low,
        abs(current.high - previous.close),
        abs(current.low - previous.close),
    )


def atr(candles: list[Candle], period: int = 14) -> float | None:
    """Wilder's ATR. None when there is not enough history to mean anything."""
    if len(candles) < period + 1:
        return None
    trs = [true_range(candles[i], candles[i - 1]) for i in range(1, len(candles))]
    window = trs[-period:]
    return round(sum(window) / period, 4)


def find_swings(candles: list[Candle], strength: int = 2) -> list[Swing]:
    """Fractal swings: a high with `strength` lower highs on both sides.

    `strength` is the noise filter. 2 is right for 5m gold; raise it on lower
    timeframes or the detectors fire on every wiggle and the host narrates
    noise all day.
    """
    swings: list[Swing] = []
    n = len(candles)
    for i in range(strength, n - strength):
        window = candles[i - strength : i + strength + 1]
        c = candles[i]
        if all(c.high >= o.high for o in window) and any(
            c.high > o.high for o in window if o is not c
        ):
            swings.append(Swing(i, c.high, c.open_time, "high"))
        if all(c.low <= o.low for o in window) and any(
            c.low < o.low for o in window if o is not c
        ):
            swings.append(Swing(i, c.low, c.open_time, "low"))
    return swings


def last_swing(swings: list[Swing], kind: str, before: int | None = None) -> Swing | None:
    for s in reversed(swings):
        if s.kind != kind:
            continue
        if before is not None and s.index >= before:
            continue
        return s
    return None


def classify_structure(swings: list[Swing]) -> Structure:
    """Compare the two most recent swings of each kind."""
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return Structure.CONSOLIDATION

    hh = highs[-1].price > highs[-2].price
    hl = lows[-1].price > lows[-2].price
    lh = highs[-1].price < highs[-2].price
    ll = lows[-1].price < lows[-2].price

    if hh and hl:
        return Structure.HIGHER_HIGH
    if hl and lh:
        return Structure.HIGHER_LOW
    if lh and ll:
        return Structure.LOWER_LOW
    if ll and hh:
        return Structure.LOWER_HIGH
    return Structure.CONSOLIDATION


def classify_trend(swings: list[Swing], candles: list[Candle]) -> Trend:
    structure = classify_structure(swings)
    if structure in (Structure.HIGHER_HIGH, Structure.HIGHER_LOW):
        return Trend.BULLISH
    if structure in (Structure.LOWER_LOW, Structure.LOWER_HIGH):
        return Trend.BEARISH
    return Trend.RANGING


def realised_volatility(candles: list[Candle], period: int = 20) -> float | None:
    """Standard deviation of close-to-close returns, as a percentage."""
    if len(candles) < period + 1:
        return None
    closes = [c.close for c in candles[-(period + 1) :]]
    rets = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
        if closes[i - 1]
    ]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return round((var**0.5) * 100, 4)


def session_extremes(candles: list[Candle], since: datetime) -> tuple[float, float] | None:
    window = [c for c in candles if c.open_time >= since]
    if not window:
        return None
    return min(c.low for c in window), max(c.high for c in window)


# --- trading sessions (UTC) ------------------------------------------------
# Approximate and deliberately so: these are conversational reference points
# for the host, not settlement times.

SESSION_WINDOWS: list[tuple[TradingSession, time, time]] = [
    (TradingSession.ASIAN, time(23, 0), time(7, 0)),
    (TradingSession.LONDON, time(7, 0), time(12, 0)),
    (TradingSession.NEW_YORK, time(12, 0), time(21, 0)),
]


def current_session(now: datetime) -> TradingSession:
    t = now.astimezone(timezone.utc).time()
    for name, start, end in SESSION_WINDOWS:
        if start <= end:
            if start <= t < end:
                return name
        elif t >= start or t < end:  # wraps midnight
            return name
    return TradingSession.OFF_HOURS


def session_start(now: datetime) -> datetime:
    """When the current session began, for session-high/low calculations."""
    session = current_session(now)
    now_utc = now.astimezone(timezone.utc)
    for name, start, _end in SESSION_WINDOWS:
        if name is not session:
            continue
        candidate = now_utc.replace(
            hour=start.hour, minute=start.minute, second=0, microsecond=0
        )
        if candidate > now_utc:
            candidate = candidate.replace(day=candidate.day) - __import__(
                "datetime"
            ).timedelta(days=1)
        return candidate
    return now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
