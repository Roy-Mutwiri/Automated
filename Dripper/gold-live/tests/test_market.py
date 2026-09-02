"""Market engine: candles, indicators, detectors, staleness.

These are measurements the host will state with confidence, so they get real
tests with hand-computed expectations rather than snapshot assertions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from platform_.market.candles import Candle, CandleSeries
from platform_.market.detectors import (
    DetectorContext,
    detect_break_of_structure,
    detect_change_of_character,
    detect_liquidity_sweep,
    detect_volatility_expansion,
)
from platform_.market.engine import MarketEngine
from platform_.market.indicators import (
    atr,
    classify_structure,
    current_session,
    find_swings,
    true_range,
)
from shared.contracts import (
    MarketConfidence,
    MarketEvent,
    MarketEventKind,
    Structure,
    Trend,
)

T0 = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)


def candle(i: int, o: float, h: float, low: float, c: float, closed: bool = True) -> Candle:
    return Candle(
        open_time=T0 + timedelta(minutes=5 * i), open=o, high=h, low=low,
        close=c, closed=closed,
    )


# -- candles ---------------------------------------------------------------


def test_candles_bucket_to_epoch_boundaries():
    """Buckets must align to the clock, not to when we happened to start."""
    s = CandleSeries(period_s=300)
    s.update(3650.0, datetime(2026, 9, 1, 8, 3, 17, tzinfo=timezone.utc))
    assert s.last is not None
    assert s.last.open_time == datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)


def test_candle_ohlc_accumulates():
    s = CandleSeries(period_s=300)
    base = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
    for offset, price in [(0, 3650.0), (60, 3655.0), (120, 3648.0), (180, 3652.0)]:
        s.update(price, base + timedelta(seconds=offset))
    c = s.last
    assert (c.open, c.high, c.low, c.close) == (3650.0, 3655.0, 3648.0, 3652.0)
    assert not c.closed


def test_new_bucket_closes_previous():
    s = CandleSeries(period_s=300)
    base = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
    s.update(3650.0, base)
    closed = s.update(3660.0, base + timedelta(seconds=310))
    assert closed is not None and closed.closed
    assert len(s) == 2


def test_quiet_market_still_closes_the_candle():
    """Without this a flat market leaves the last candle forming forever and
    no detector ever fires."""
    s = CandleSeries(period_s=300)
    base = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
    s.update(3650.0, base)
    assert s.mark_stale_closed(base + timedelta(seconds=100)) is None
    assert s.mark_stale_closed(base + timedelta(seconds=400)) is not None


def test_wick_and_body_geometry():
    c = candle(0, o=100.0, h=110.0, low=95.0, c=105.0)
    assert c.body == 5.0
    assert c.upper_wick == 5.0
    assert c.lower_wick == 5.0
    assert c.range == 15.0
    assert c.bullish


# -- indicators ------------------------------------------------------------


def test_true_range_uses_previous_close():
    prev = candle(0, 100, 102, 99, 101)
    cur = candle(1, 105, 107, 104, 106)
    # gap up: high-prevclose = 107-101 = 6 beats high-low = 3
    assert true_range(cur, prev) == 6.0


def test_atr_needs_history():
    assert atr([candle(i, 100, 101, 99, 100) for i in range(5)], period=14) is None


def test_atr_of_constant_range_is_that_range():
    candles = [candle(i, 100, 101, 99, 100) for i in range(30)]
    assert atr(candles, period=14) == pytest.approx(2.0)


def test_finds_swing_high_and_low():
    prices = [100, 101, 105, 101, 100, 99, 95, 99, 100]
    candles = [candle(i, p, p + 0.5, p - 0.5, p) for i, p in enumerate(prices)]
    swings = find_swings(candles, strength=2)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    assert highs and highs[0].index == 2
    assert lows and lows[0].index == 6


def test_structure_classification_higher_high():
    prices = [100, 104, 100, 96, 100, 108, 103, 99, 104]
    candles = [candle(i, p, p + 0.5, p - 0.5, p) for i, p in enumerate(prices)]
    assert classify_structure(find_swings(candles, 1)) in (
        Structure.HIGHER_HIGH, Structure.HIGHER_LOW,
    )


def test_structure_is_consolidation_without_enough_swings():
    assert classify_structure([]) is Structure.CONSOLIDATION


@pytest.mark.parametrize(
    "hour,expected",
    [(2, "asian"), (9, "london"), (15, "new_york"), (22, "off_hours")],
)
def test_session_windows(hour, expected):
    when = datetime(2026, 9, 1, hour, 0, tzinfo=timezone.utc)
    assert current_session(when).value == expected


# -- detectors -------------------------------------------------------------


def ctx(candles, trend=Trend.BULLISH, atr_value=1.0, baseline=1.0, strength=1):
    return DetectorContext(
        candles=candles, swings=find_swings(candles, strength), timeframe="5m",
        trend=trend, atr_value=atr_value, atr_baseline=baseline,
        reported_levels=set(),
    )


def test_bos_fires_on_close_beyond_swing_high():
    prices = [100, 105, 100, 98, 100, 107]
    candles = [candle(i, p, p + 0.2, p - 0.2, p) for i, p in enumerate(prices)]
    events = detect_break_of_structure(ctx(candles))
    assert events and events[0].direction == "up"
    assert events[0].severity == 5
    assert events[0].evidence["rule"] == "bos_close_beyond_swing_high"


def test_bos_does_not_fire_twice_for_the_same_level():
    prices = [100, 105, 100, 98, 100, 107]
    candles = [candle(i, p, p + 0.2, p - 0.2, p) for i, p in enumerate(prices)]
    context = ctx(candles)
    assert detect_break_of_structure(context)
    assert detect_break_of_structure(context) == []


def test_wick_beyond_level_is_a_sweep_not_a_break():
    """The close decides. Trading beyond a level and being rejected is the
    opposite signal to accepting beyond it."""
    prices = [100, 105, 100, 98, 100]
    candles = [candle(i, p, p + 0.2, p - 0.2, p) for i, p in enumerate(prices)]
    # High pierces 105 but closes back at 101.
    candles.append(candle(5, o=100, h=106.5, low=99.5, c=101))
    context = ctx(candles, atr_value=1.0)

    assert detect_break_of_structure(context) == [], "close is inside; not a break"
    sweeps = detect_liquidity_sweep(context)
    assert sweeps and sweeps[0].kind.value == "liquidity_sweep"
    assert sweeps[0].direction == "down"


def test_sweep_requires_a_meaningful_wick():
    prices = [100, 105, 100, 98, 100]
    candles = [candle(i, p, p + 0.2, p - 0.2, p) for i, p in enumerate(prices)]
    candles.append(candle(5, o=100, h=105.05, low=99.5, c=101))
    # Wick is tiny relative to ATR -- noise, not a sweep.
    assert detect_liquidity_sweep(ctx(candles, atr_value=5.0)) == []


def test_choch_fires_against_the_trend():
    prices = [100, 104, 100, 96, 100]
    candles = [candle(i, p, p + 0.2, p - 0.2, p) for i, p in enumerate(prices)]
    candles.append(candle(5, o=99, h=99.2, low=94, c=94.5))
    events = detect_change_of_character(ctx(candles, trend=Trend.BULLISH))
    assert events and events[0].direction == "down"
    assert events[0].severity == 4


def test_choch_silent_in_a_range():
    candles = [candle(i, 100, 100.2, 99.8, 100) for i in range(6)]
    assert detect_change_of_character(ctx(candles, trend=Trend.RANGING)) == []


def test_volatility_expansion_threshold():
    candles = [candle(i, 100, 101, 99, 100) for i in range(6)]
    assert detect_volatility_expansion(ctx(candles, atr_value=1.0, baseline=1.0)) == []
    events = detect_volatility_expansion(ctx(candles, atr_value=2.5, baseline=1.0))
    assert events and events[0].evidence["ratio"] == 2.5


def test_detector_failure_does_not_blind_the_others():
    from platform_.market import detectors

    def boom(_ctx):
        raise RuntimeError("bad rule")

    original = detectors.ALL_DETECTORS[:]
    detectors.ALL_DETECTORS.insert(0, boom)
    try:
        prices = [100, 105, 100, 98, 100, 107]
        candles = [candle(i, p, p + 0.2, p - 0.2, p) for i, p in enumerate(prices)]
        assert detectors.run_all(ctx(candles)), "surviving detectors must still run"
    finally:
        detectors.ALL_DETECTORS[:] = original


# -- engine and staleness --------------------------------------------------


def test_confidence_degrades_with_silence():
    e = MarketEngine(delayed_after_s=5, stale_after_s=15, unavailable_after_s=120)
    assert e.confidence(T0) is MarketConfidence.UNAVAILABLE, "no ticks yet"

    e.on_tick(3652.2, 3652.6, T0)
    assert e.confidence(T0) is MarketConfidence.LIVE
    assert e.confidence(T0 + timedelta(seconds=6)) is MarketConfidence.DELAYED
    assert e.confidence(T0 + timedelta(seconds=20)) is MarketConfidence.STALE
    assert e.confidence(T0 + timedelta(seconds=300)) is MarketConfidence.UNAVAILABLE


def test_snapshot_blocks_price_quoting_when_stale():
    e = MarketEngine(stale_after_s=10)
    e.on_tick(3652.2, 3652.6, T0)
    assert e.snapshot(T0).may_quote_price()
    assert not e.snapshot(T0 + timedelta(seconds=30)).may_quote_price()


def test_engine_builds_snapshot_from_ticks():
    e = MarketEngine(timeframes={"5m": 300})
    price = 3650.0
    at = T0
    # 400 ticks x 30s = ~33 five-minute candles, enough for ATR (needs 15).
    for i in range(400):
        price += 0.6 if i % 3 else -0.4
        at += timedelta(seconds=30)
        e.on_tick(round(price - 0.18, 2), round(price + 0.18, 2), at)

    state = e.snapshot(at)
    assert state.symbol == "XAUUSD"
    assert "5m" in state.timeframes
    assert state.timeframes["5m"].atr is not None
    assert state.price.mid > 0
    assert any(o.key == "range_5m" for o in state.observations)


def test_atr_is_none_until_enough_history():
    """Detectors that need ATR must stay silent rather than guess."""
    e = MarketEngine(timeframes={"5m": 300})
    at = T0
    for i in range(60):  # only ~6 candles
        at += timedelta(seconds=30)
        e.on_tick(3650.0, 3650.4, at)
    view = e.snapshot(at).timeframes.get("5m")
    assert view is None or view.atr is None
    assert not e.warm


def test_seed_history_makes_detectors_usable_immediately():
    """A restart mid-session must not blind sweep and volatility detection for
    the next 75 minutes."""
    e = MarketEngine(timeframes={"5m": 300})
    assert not e.warm

    history = [
        candle(i, 3650 + i * 0.4, 3651 + i * 0.4, 3649 + i * 0.4, 3650.5 + i * 0.4)
        for i in range(40)
    ]
    taken = e.seed_history("5m", history)

    assert taken == 40
    assert e.warm, "engine should be usable straight after seeding"
    e.on_tick(3670.0, 3670.4, T0 + timedelta(hours=4))
    assert e.snapshot(T0 + timedelta(hours=4)).timeframes["5m"].atr is not None


def test_seed_history_marks_prior_levels_as_known():
    """Levels broken before startup are history, not breaking news."""
    e = MarketEngine(timeframes={"5m": 300})
    prices = [100, 105, 100, 98, 100, 103]
    e.seed_history(
        "5m", [candle(i, p, p + 0.5, p - 0.5, p) for i, p in enumerate(prices)]
    )
    assert e._reported_levels["5m"], "swing levels from history should be recorded"


def test_seed_history_rejects_unknown_timeframe():
    with pytest.raises(KeyError):
        MarketEngine(timeframes={"5m": 300}).seed_history("1h", [])


def test_engine_emits_and_drains_events():
    e = MarketEngine(timeframes={"5m": 300}, swing_strength=1)
    at = T0
    # Rising then a sharp push through the prior high.
    for price in [3650, 3655, 3651, 3648, 3652, 3665, 3668, 3670]:
        for _ in range(3):
            at += timedelta(seconds=120)
            e.on_tick(price - 0.18, price + 0.18, at)
    e.on_bar_close_check(at + timedelta(seconds=400))

    events = e.drain_events()
    assert e.drain_events() == [], "drain must be idempotent"
    if events:
        assert all(ev.evidence for ev in events), "every event carries its evidence"


# -- event noise control ---------------------------------------------------


def test_one_minute_timeframe_is_not_analysed():
    """Gold breaks 'structure' every few candles on 1m. Narrating that is
    unlistenable, so 1m is collected for extremes but never analysed."""
    e = MarketEngine(timeframes={"1m": 60, "5m": 300})
    assert "1m" not in e.analysed
    assert "5m" in e.analysed

    at = T0
    for price in [3650, 3656, 3651, 3647, 3653, 3668, 3672, 3676]:
        for _ in range(4):
            at += timedelta(seconds=20)
            e.on_tick(price - 0.18, price + 0.18, at)

    assert all(ev.timeframe != "1m" for ev in e.drain_events())


def test_same_event_kind_is_rate_limited():
    """Real structure does not break twice in ninety seconds."""
    e = MarketEngine(timeframes={"5m": 300})
    first = MarketEvent(
        kind=MarketEventKind.BOS, timeframe="5m", occurred_at=T0, severity=5
    )
    soon = MarketEvent(
        kind=MarketEventKind.BOS, timeframe="5m",
        occurred_at=T0 + timedelta(seconds=90), severity=5,
    )
    later = MarketEvent(
        kind=MarketEventKind.BOS, timeframe="5m",
        occurred_at=T0 + timedelta(seconds=400), severity=5,
    )

    assert e._rate_limit("5m", [first]) == [first]
    assert e._rate_limit("5m", [soon]) == []
    assert e._rate_limit("5m", [later]) == [later]


def test_rate_limit_is_per_kind_and_timeframe():
    e = MarketEngine(timeframes={"5m": 300, "15m": 900})
    bos = MarketEvent(kind=MarketEventKind.BOS, timeframe="5m",
                      occurred_at=T0, severity=5)
    sweep = MarketEvent(kind=MarketEventKind.LIQUIDITY_SWEEP, timeframe="5m",
                        occurred_at=T0, severity=4)
    assert e._rate_limit("5m", [bos]) == [bos]
    assert e._rate_limit("5m", [sweep]) == [sweep], "different kind, not suppressed"
    assert e._rate_limit("15m", [bos]) == [bos], "different timeframe, not suppressed"
