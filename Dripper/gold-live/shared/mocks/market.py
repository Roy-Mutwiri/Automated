"""Mock market engine: a scripted XAUUSD session with real structure.

Produces the same MarketState/MarketEvent contracts the real engine will.
The scenario deliberately includes a stale-data window so the safety gate
gets exercised on every dry run rather than only in a dedicated test.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from shared.contracts import (
    Detection,
    MarketConfidence,
    MarketContext,
    MarketEvent,
    MarketEventKind,
    MarketState,
    Observation,
    Price,
    Structure,
    TimeframeView,
    TradingSession,
    Trend,
)


class MockMarketEngine:
    """Replays a scripted London-session move in Gold.

    Beat structure (one beat ~= one tick):
      0-4    quiet consolidation
      5      volatility expansion
      6      liquidity sweep below the session low
      7-9    reversal, higher lows forming
      10     break of structure to the upside
      11-13  continuation
      14-16  FEED DEGRADES -> delayed -> stale   (safety gate must fire)
      17+    recovery, back to live
    """

    #: Beats before the scenario repeats. Soak runs need continuous event flow;
    #: a one-shot script produces three events and then 24 hours of nothing,
    #: which tells you nothing about sustained behaviour.
    CYCLE = 18

    def __init__(self, seed: int = 7, base_price: float = 3652.40, cyclic: bool = False) -> None:
        self.rng = random.Random(seed)
        self.base = base_price
        self.cyclic = cyclic
        self.beat = 0
        self.t0 = datetime.now(timezone.utc)
        self.session_low = base_price - 6.0
        self.session_high = base_price + 3.5
        self._pending: list[MarketEvent] = []

    # -- helpers ----------------------------------------------------------

    @property
    def phase_beat(self) -> int:
        """Position within the scenario. Wraps when cyclic."""
        return self.beat % self.CYCLE if self.cyclic else self.beat

    def _confidence(self) -> MarketConfidence:
        b = self.phase_beat
        if b in (14, 15):
            return MarketConfidence.DELAYED
        if b == 16:
            return MarketConfidence.STALE
        return MarketConfidence.LIVE

    def _drift(self) -> float:
        b = self.phase_beat
        if b <= 4:
            return self.rng.uniform(-0.6, 0.6)
        if b == 5:
            return self.rng.uniform(-2.5, -1.5)
        if b == 6:
            return -3.2
        if b <= 9:
            return self.rng.uniform(0.8, 2.0)
        if b == 10:
            return 2.6
        if b <= 13:
            return self.rng.uniform(0.2, 1.4)
        return self.rng.uniform(-0.5, 0.5)

    # -- public API -------------------------------------------------------

    def tick(self) -> MarketState:
        self.base = round(self.base + self._drift(), 2)
        now = self.t0 + timedelta(seconds=self.beat * 20)
        conf = self._confidence()

        # A stale feed means as_of falls behind computed_at -- staleness_ms
        # is derived from that gap, exactly as it will be in production.
        lag = {MarketConfidence.LIVE: 0.4, MarketConfidence.DELAYED: 6.0}.get(conf, 45.0)
        as_of = now - timedelta(seconds=lag)

        self.session_low = min(self.session_low, self.base)
        self.session_high = max(self.session_high, self.base)

        pb = self.phase_beat
        trend = (
            Trend.RANGING
            if pb <= 4
            else Trend.BEARISH
            if pb in (5, 6)
            else Trend.BULLISH
        )
        structure = (
            Structure.CONSOLIDATION
            if pb <= 4
            else Structure.LOWER_LOW
            if pb in (5, 6)
            else Structure.HIGHER_LOW
            if pb <= 9
            else Structure.HIGHER_HIGH
        )

        tf = TimeframeView(
            timeframe="5m",
            open=round(self.base - 1.1, 2),
            high=round(self.base + 0.9, 2),
            low=round(self.base - 1.6, 2),
            close=self.base,
            trend=trend,
            structure=structure,
            swing_high=round(self.session_high, 2),
            swing_low=round(self.session_low, 2),
            # Deliberately crosses the quiet/active threshold so a soak run
            # exercises both phases rather than assuming the market is always
            # interesting. Consolidation beats sit below it.
            atr=1.35 if pb <= 4 else (4.0 if pb <= 13 else 1.6),
        )

        detections: list[Detection] = []
        if self.beat == 6:
            detections.append(
                Detection(
                    rule_id="liq_sweep_v1",
                    label="Sell-side liquidity swept below session low",
                    timeframe="5m",
                    price_level=round(self.session_low, 2),
                    evidence={"prior_low": round(self.session_low + 0.4, 2), "wick_atr": 1.8},
                )
            )
        if self.beat == 10:
            detections.append(
                Detection(
                    rule_id="bos_v1",
                    label="Break of structure above prior swing high",
                    timeframe="5m",
                    price_level=round(self.session_high, 2),
                    evidence={"prior_high": round(self.session_high - 0.3, 2), "close_beyond": True},
                )
            )

        state = MarketState(
            as_of=as_of,
            computed_at=now,
            confidence=conf,
            price=Price(bid=round(self.base - 0.18, 2), ask=round(self.base + 0.18, 2)),
            session=TradingSession.LONDON,
            timeframes={"5m": tf},
            observations=[
                Observation(key="range_5m", value=round(tf.high - tf.low, 2), unit="usd",
                            timeframe="5m"),
                Observation(key="session_low", value=round(self.session_low, 2), unit="usd"),
                Observation(key="session_high", value=round(self.session_high, 2), unit="usd"),
            ],
            detections=detections,
            context=[
                MarketContext(kind="session", label="London session", detail="mid-session"),
                MarketContext(kind="calendar", label="US CPI", detail="in 2 hours",
                              at=now + timedelta(hours=2)),
            ],
        )

        self._queue_events(state)
        self.beat += 1
        return state

    def _queue_events(self, state: MarketState) -> None:
        b = self.phase_beat
        if b == 5:
            self._pending.append(
                MarketEvent(
                    kind=MarketEventKind.VOL_EXPANSION, timeframe="5m",
                    occurred_at=state.as_of, severity=3, direction="down",
                    evidence={"atr_ratio": 1.9},
                    narrative_hint="Volatility just expanded; ranges are widening.",
                    market_state_id=state.state_id,
                )
            )
        if b == 6:
            self._pending.append(
                MarketEvent(
                    kind=MarketEventKind.LIQUIDITY_SWEEP, timeframe="5m",
                    occurred_at=state.as_of, severity=4, direction="down",
                    price_level=round(self.session_low, 2),
                    evidence={"rule": "liq_sweep_v1"},
                    narrative_hint="Price ran the stops below the session low and snapped back.",
                    market_state_id=state.state_id,
                )
            )
        if b == 10:
            self._pending.append(
                MarketEvent(
                    kind=MarketEventKind.BOS, timeframe="5m",
                    occurred_at=state.as_of, severity=5, direction="up",
                    price_level=round(self.session_high, 2),
                    evidence={"rule": "bos_v1"},
                    narrative_hint="Break of structure to the upside after the sweep.",
                    market_state_id=state.state_id,
                )
            )

    def drain_events(self) -> list[MarketEvent]:
        out, self._pending = self._pending, []
        return out
