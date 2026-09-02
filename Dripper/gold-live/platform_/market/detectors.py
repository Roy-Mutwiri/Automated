"""Event detection rules.

Each detector is a pure function returning zero or more MarketEvents, and every
event carries the evidence that produced it. That evidence is what makes the
observability requirement answerable: "why did Session 4 say there was a break
of structure?" resolves to a rule id, its inputs, and its threshold.

Severity is assigned here, deterministically, and feeds the Director's priority.
A model never sets it -- if it did, how loudly the host reacts would drift with
sampling temperature.

Detectors only ever look at CLOSED candles. A level that is broken intrabar and
recovered before the close is not a break; narrating it is how a host ends up
contradicting themselves ninety seconds later.
"""

from __future__ import annotations

from dataclasses import dataclass

from platform_.market.candles import Candle
from platform_.market.indicators import Swing, last_swing
from shared.contracts import MarketEvent, MarketEventKind, Trend


@dataclass(slots=True)
class DetectorContext:
    candles: list[Candle]
    swings: list[Swing]
    timeframe: str
    trend: Trend
    atr_value: float | None
    #: Rolling median ATR, for deciding what counts as an expansion.
    atr_baseline: float | None
    #: Levels already reported, so the same break is not announced twice.
    reported_levels: set[float]


def _round_level(price: float) -> float:
    return round(price, 2)


def detect_break_of_structure(ctx: DetectorContext) -> list[MarketEvent]:
    """Close beyond the most recent opposing swing, in the trend direction."""
    if len(ctx.candles) < 3:
        return []
    last = ctx.candles[-1]
    events: list[MarketEvent] = []

    prior_high = last_swing(ctx.swings, "high", before=len(ctx.candles) - 1)
    prior_low = last_swing(ctx.swings, "low", before=len(ctx.candles) - 1)

    if prior_high and last.close > prior_high.price:
        level = _round_level(prior_high.price)
        if level not in ctx.reported_levels:
            ctx.reported_levels.add(level)
            events.append(
                MarketEvent(
                    kind=MarketEventKind.BOS, timeframe=ctx.timeframe,
                    occurred_at=last.open_time, severity=5, direction="up",
                    price_level=level,
                    evidence={
                        "rule": "bos_close_beyond_swing_high",
                        "prior_swing_high": level,
                        "close": round(last.close, 2),
                        "margin": round(last.close - prior_high.price, 2),
                    },
                    narrative_hint="Price closed above the prior swing high.",
                )
            )

    if prior_low and last.close < prior_low.price:
        level = _round_level(prior_low.price)
        if level not in ctx.reported_levels:
            ctx.reported_levels.add(level)
            events.append(
                MarketEvent(
                    kind=MarketEventKind.BOS, timeframe=ctx.timeframe,
                    occurred_at=last.open_time, severity=5, direction="down",
                    price_level=level,
                    evidence={
                        "rule": "bos_close_beyond_swing_low",
                        "prior_swing_low": level,
                        "close": round(last.close, 2),
                        "margin": round(prior_low.price - last.close, 2),
                    },
                    narrative_hint="Price closed below the prior swing low.",
                )
            )
    return events


def detect_change_of_character(ctx: DetectorContext) -> list[MarketEvent]:
    """A break AGAINST the prevailing trend -- the first sign it is turning.

    Distinct from a BOS: a break with the trend continues the story, a break
    against it changes who is in control. Lower severity than a BOS because it
    is the earlier and less reliable signal.
    """
    if len(ctx.candles) < 3 or ctx.trend is Trend.RANGING:
        return []
    last = ctx.candles[-1]

    if ctx.trend is Trend.BULLISH:
        prior_low = last_swing(ctx.swings, "low", before=len(ctx.candles) - 1)
        if prior_low and last.close < prior_low.price:
            return [
                MarketEvent(
                    kind=MarketEventKind.CHOCH, timeframe=ctx.timeframe,
                    occurred_at=last.open_time, severity=4, direction="down",
                    price_level=_round_level(prior_low.price),
                    evidence={
                        "rule": "choch_against_uptrend",
                        "prevailing_trend": "bullish",
                        "broken_low": _round_level(prior_low.price),
                    },
                    narrative_hint="First lower low after an uptrend.",
                )
            ]
    else:
        prior_high = last_swing(ctx.swings, "high", before=len(ctx.candles) - 1)
        if prior_high and last.close > prior_high.price:
            return [
                MarketEvent(
                    kind=MarketEventKind.CHOCH, timeframe=ctx.timeframe,
                    occurred_at=last.open_time, severity=4, direction="up",
                    price_level=_round_level(prior_high.price),
                    evidence={
                        "rule": "choch_against_downtrend",
                        "prevailing_trend": "bearish",
                        "broken_high": _round_level(prior_high.price),
                    },
                    narrative_hint="First higher high after a downtrend.",
                )
            ]
    return []


def detect_liquidity_sweep(ctx: DetectorContext, wick_atr_ratio: float = 0.6) -> list[MarketEvent]:
    """Wick beyond a prior swing, close back inside.

    The close is the whole point. Trading beyond a level and accepting there is
    a break; trading beyond it and being rejected is a sweep. Same extreme,
    opposite meaning.
    """
    if len(ctx.candles) < 3 or not ctx.atr_value:
        return []
    last = ctx.candles[-1]
    events: list[MarketEvent] = []

    prior_high = last_swing(ctx.swings, "high", before=len(ctx.candles) - 1)
    prior_low = last_swing(ctx.swings, "low", before=len(ctx.candles) - 1)

    if (
        prior_high
        and last.high > prior_high.price
        and last.close < prior_high.price
        and last.upper_wick >= wick_atr_ratio * ctx.atr_value
    ):
        events.append(
            MarketEvent(
                kind=MarketEventKind.LIQUIDITY_SWEEP, timeframe=ctx.timeframe,
                occurred_at=last.open_time, severity=4, direction="down",
                price_level=_round_level(prior_high.price),
                evidence={
                    "rule": "sweep_above_swing_high",
                    "swept_level": _round_level(prior_high.price),
                    "wick": round(last.upper_wick, 2),
                    "atr": ctx.atr_value,
                    "threshold": round(wick_atr_ratio * ctx.atr_value, 2),
                },
                narrative_hint="Price ran the stops above the prior high and was rejected.",
            )
        )

    if (
        prior_low
        and last.low < prior_low.price
        and last.close > prior_low.price
        and last.lower_wick >= wick_atr_ratio * ctx.atr_value
    ):
        events.append(
            MarketEvent(
                kind=MarketEventKind.LIQUIDITY_SWEEP, timeframe=ctx.timeframe,
                occurred_at=last.open_time, severity=4, direction="up",
                price_level=_round_level(prior_low.price),
                evidence={
                    "rule": "sweep_below_swing_low",
                    "swept_level": _round_level(prior_low.price),
                    "wick": round(last.lower_wick, 2),
                    "atr": ctx.atr_value,
                    "threshold": round(wick_atr_ratio * ctx.atr_value, 2),
                },
                narrative_hint="Price ran the stops below the prior low and snapped back.",
            )
        )
    return events


def detect_volatility_expansion(
    ctx: DetectorContext, ratio: float = 1.8
) -> list[MarketEvent]:
    if not ctx.atr_value or not ctx.atr_baseline or ctx.atr_baseline <= 0:
        return []
    observed = ctx.atr_value / ctx.atr_baseline
    if observed < ratio:
        return []
    return [
        MarketEvent(
            kind=MarketEventKind.VOL_EXPANSION, timeframe=ctx.timeframe,
            occurred_at=ctx.candles[-1].open_time, severity=3,
            evidence={
                "rule": "atr_expansion",
                "atr": ctx.atr_value,
                "baseline": ctx.atr_baseline,
                "ratio": round(observed, 2),
                "threshold": ratio,
            },
            narrative_hint="Ranges are widening; volatility just expanded.",
        )
    ]


ALL_DETECTORS = [
    detect_break_of_structure,
    detect_change_of_character,
    detect_liquidity_sweep,
    detect_volatility_expansion,
]


def run_all(ctx: DetectorContext) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    for detector in ALL_DETECTORS:
        try:
            events.extend(detector(ctx))
        except Exception:
            import logging

            logging.getLogger(__name__).exception("detector %s failed", detector.__name__)
    return events
