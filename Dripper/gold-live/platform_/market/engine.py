"""The Gold market engine.

Computed once on the central machine and fanned out to every session. Seven
sessions must never each derive their own view of the market -- besides the
waste, they would disagree, and two hosts contradicting each other about where
the session low is destroys credibility faster than any other failure.

Owns the staleness state machine, which is a safety control rather than
bookkeeping:

    LIVE      ticks arriving normally
    DELAYED   nothing for `delayed_after_s` -- price still quotable but flagged
    STALE     nothing for `stale_after_s`   -- price quoting DISABLED downstream
    UNAVAILABLE  no feed at all (weekend, or the provider is gone)

The transitions are time-based and unconditional. A feed that silently stops
delivering while the socket stays open is the common failure, and it is
indistinguishable from a quiet market except by elapsed time.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime, timedelta, timezone

from platform_.market.candles import CandleSeries
from platform_.market.detectors import DetectorContext, run_all
from platform_.market.indicators import (
    atr,
    classify_structure,
    classify_trend,
    current_session,
    find_swings,
    realised_volatility,
    session_start,
)
from shared.contracts import (
    Detection,
    MarketConfidence,
    MarketContext,
    MarketEvent,
    MarketEventKind,
    MarketState,
    Observation,
    Price,
    TimeframeView,
)

log = logging.getLogger(__name__)

DEFAULT_TIMEFRAMES = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}


class MarketEngine:
    #: Timeframes the detectors run on. 1m is collected for session extremes and
    #: price display but is NOT analysed: at that resolution gold produces a
    #: "structure break" every few candles, and a host reacting to each one is
    #: unlistenable. Structure means something on 5m and above.
    ANALYSED_TIMEFRAMES = frozenset({"5m", "15m", "1h"})

    #: Minimum gap between two events of the same kind on the same timeframe.
    #: Real structure does not break twice in ninety seconds; a detector saying
    #: it did is noise, and narrating noise is how a host loses credibility.
    EVENT_COOLDOWN_S = {
        MarketEventKind.BOS: 300,
        MarketEventKind.CHOCH: 600,
        MarketEventKind.LIQUIDITY_SWEEP: 300,
        MarketEventKind.VOL_EXPANSION: 900,
    }

    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframes: dict[str, int] | None = None,
        delayed_after_s: float = 5.0,
        stale_after_s: float = 15.0,
        unavailable_after_s: float = 120.0,
        swing_strength: int = 2,
        analysed_timeframes: frozenset[str] | None = None,
    ) -> None:
        self.symbol = symbol
        self.series = {
            name: CandleSeries(period_s)
            for name, period_s in (timeframes or DEFAULT_TIMEFRAMES).items()
        }
        self.analysed = analysed_timeframes or self.ANALYSED_TIMEFRAMES
        self._last_event_at: dict[tuple[str, str], datetime] = {}
        self.delayed_after_s = delayed_after_s
        self.stale_after_s = stale_after_s
        self.unavailable_after_s = unavailable_after_s
        self.swing_strength = swing_strength

        self.last_tick_at: datetime | None = None
        self.bid: float | None = None
        self.ask: float | None = None
        self._atr_history: dict[str, list[float]] = {k: [] for k in self.series}
        self._reported_levels: dict[str, set[float]] = {k: set() for k in self.series}
        self._pending: list[MarketEvent] = []
        #: What the price actually comes from. Reaches the generator via
        #: MarketState.context so the host can be accurate about its source
        #: rather than implying an interbank quote.
        self.price_source_note: str | None = None

    # -- warm-up ----------------------------------------------------------

    def seed_history(self, timeframe: str, candles: list) -> int:
        """Preload closed candles so detectors work immediately after a restart.

        Without this the engine starts blind: ATR needs 15 candles, so on a 5m
        timeframe nothing that depends on it -- liquidity sweeps, volatility
        expansion -- can fire for the first ~75 minutes of uptime. A process
        restart mid-session would otherwise leave the host unable to react to
        the two event types viewers most notice.

        Call with recent history from the provider's REST endpoint at startup,
        before connecting the live feed. Returns how many candles were taken.
        """
        series = self.series.get(timeframe)
        if series is None:
            raise KeyError(f"no series for timeframe {timeframe!r}")

        taken = 0
        for c in candles:
            c.closed = True
            series.candles.append(c)
            taken += 1

        closed = series.closed_candles
        for end in range(15, len(closed) + 1):
            value = atr(closed[:end])
            if value:
                self._atr_history[timeframe].append(value)

        # Levels already broken before we started are history, not news.
        swings = find_swings(closed, self.swing_strength)
        for s in swings:
            self._reported_levels[timeframe].add(round(s.price, 2))

        log.info(
            "seeded %d %s candles (%d atr samples, %d known levels)",
            taken, timeframe, len(self._atr_history[timeframe]),
            len(self._reported_levels[timeframe]),
        )
        return taken

    @property
    def warm(self) -> bool:
        """Have the ATR-dependent detectors got enough history to fire?"""
        return any(len(h) >= 1 for h in self._atr_history.values())

    # -- ingest -----------------------------------------------------------

    def on_tick(self, bid: float, ask: float, at: datetime) -> None:
        self.bid, self.ask = bid, ask
        self.last_tick_at = at
        mid = (bid + ask) / 2
        for name, series in self.series.items():
            if series.update(mid, at) is not None:
                self._on_candle_close(name, at)

    def on_bar_close_check(self, now: datetime) -> None:
        """Close candles whose period elapsed with no ticks. Without this a
        quiet market never fires a detector."""
        for name, series in self.series.items():
            if series.mark_stale_closed(now) is not None:
                self._on_candle_close(name, now)

    def _rate_limit(self, timeframe: str, events: list[MarketEvent]) -> list[MarketEvent]:
        kept: list[MarketEvent] = []
        for event in events:
            key = (timeframe, event.kind.value)
            cooldown = self.EVENT_COOLDOWN_S.get(event.kind, 300)
            last = self._last_event_at.get(key)
            if last is not None and (event.occurred_at - last).total_seconds() < cooldown:
                log.debug(
                    "suppressed %s on %s (%.0fs since last, cooldown %ds)",
                    event.kind.value, timeframe,
                    (event.occurred_at - last).total_seconds(), cooldown,
                )
                continue
            self._last_event_at[key] = event.occurred_at
            kept.append(event)
        return kept

    def _on_candle_close(self, timeframe: str, now: datetime) -> None:
        if timeframe not in self.analysed:
            return
        series = self.series[timeframe]
        candles = series.closed_candles
        if len(candles) < 5:
            return

        value = atr(candles)
        if value:
            history = self._atr_history[timeframe]
            history.append(value)
            if len(history) > 200:
                del history[:-200]

        swings = find_swings(candles, self.swing_strength)
        ctx = DetectorContext(
            candles=candles,
            swings=swings,
            timeframe=timeframe,
            trend=classify_trend(swings, candles),
            atr_value=value,
            atr_baseline=self._atr_baseline(timeframe),
            reported_levels=self._reported_levels[timeframe],
        )
        events = self._rate_limit(timeframe, run_all(ctx))
        if events:
            log.info(
                "[%s] %s: %s", self.symbol, timeframe,
                ", ".join(f"{e.kind.value}(sev={e.severity})" for e in events),
            )
        self._pending.extend(events)

    def _atr_baseline(self, timeframe: str) -> float | None:
        history = self._atr_history[timeframe]
        if len(history) < 20:
            return None
        return round(statistics.median(history[-100:]), 4)

    # -- confidence -------------------------------------------------------

    def confidence(self, now: datetime) -> MarketConfidence:
        if self.last_tick_at is None:
            return MarketConfidence.UNAVAILABLE
        age = (now - self.last_tick_at).total_seconds()
        if age >= self.unavailable_after_s:
            return MarketConfidence.UNAVAILABLE
        if age >= self.stale_after_s:
            return MarketConfidence.STALE
        if age >= self.delayed_after_s:
            return MarketConfidence.DELAYED
        return MarketConfidence.LIVE

    # -- snapshot ---------------------------------------------------------

    def snapshot(self, now: datetime | None = None) -> MarketState:
        now = now or datetime.now(timezone.utc)
        conf = self.confidence(now)
        as_of = self.last_tick_at or (now - timedelta(seconds=self.unavailable_after_s))

        views: dict[str, TimeframeView] = {}
        observations: list[Observation] = []
        detections: list[Detection] = []

        for name, series in self.series.items():
            candles = series.closed_candles
            if len(candles) < 3:
                continue
            swings = find_swings(candles, self.swing_strength)
            last = candles[-1]
            highs = [s for s in swings if s.kind == "high"]
            lows = [s for s in swings if s.kind == "low"]

            views[name] = TimeframeView(
                timeframe=name,
                open=round(last.open, 2), high=round(last.high, 2),
                low=round(last.low, 2), close=round(last.close, 2),
                volume=last.volume or None,
                trend=classify_trend(swings, candles),
                structure=classify_structure(swings),
                swing_high=round(highs[-1].price, 2) if highs else None,
                swing_low=round(lows[-1].price, 2) if lows else None,
                atr=atr(candles),
            )

            if name == "5m":
                observations.append(
                    Observation(key="range_5m", value=round(last.range, 2),
                                unit="usd", timeframe="5m")
                )
                vol = realised_volatility(candles)
                if vol is not None:
                    observations.append(
                        Observation(key="realised_vol_5m", value=vol, unit="pct",
                                    timeframe="5m")
                    )

        # Session extremes come from the finest timeframe available.
        fine = self.series.get("1m") or self.series.get("5m")
        if fine and fine.closed_candles:
            since = session_start(now)
            window = [c for c in fine.closed_candles if c.open_time >= since]
            if window:
                observations.append(
                    Observation(key="session_low",
                                value=round(min(c.low for c in window), 2), unit="usd")
                )
                observations.append(
                    Observation(key="session_high",
                                value=round(max(c.high for c in window), 2), unit="usd")
                )

        for event in self._pending[-6:]:
            detections.append(
                Detection(
                    rule_id=str(event.evidence.get("rule", event.kind.value)),
                    label=event.narrative_hint or event.kind.value,
                    timeframe=event.timeframe,
                    price_level=event.price_level,
                    evidence=event.evidence,
                )
            )

        session = current_session(now)
        return MarketState(
            symbol=self.symbol,
            as_of=as_of,
            computed_at=now,
            confidence=conf,
            price=Price(
                bid=round(self.bid, 2) if self.bid is not None else 0.0,
                ask=round(self.ask, 2) if self.ask is not None else 0.0,
            ),
            session=session,
            timeframes=views,
            observations=observations,
            detections=detections,
            context=[
                MarketContext(kind="session", label=f"{session.value} session"),
                *(
                    [MarketContext(kind="news", label="price source",
                                   detail=self.price_source_note)]
                    if self.price_source_note
                    else []
                ),
            ],
        )

    def drain_events(self) -> list[MarketEvent]:
        out, self._pending = self._pending, []
        return out
