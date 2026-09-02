"""Free real-time gold price via tokenized gold on a public exchange stream.

No API key, no rate limit, no bill, sub-second updates. Binance publishes
market data on a public websocket that requires no authentication, and
tokenized gold trades there against USDT:

    PAXG  Paxos Gold      1 token = 1 fine troy ounce, redeemable
    XAUT  Tether Gold     same idea, different issuer

Both are backed 1:1 by allocated physical gold and track spot closely. This is
the only genuinely free source of real-time gold pricing found -- every REST
API cheap enough to be free is rate-limited to minutes or hours per poll,
which is not live commentary.

WHAT THIS IS NOT, and the host should never claim otherwise:

  - It is NOT the LBMA fix or an interbank XAUUSD quote. It is the price of a
    gold-backed token on a crypto exchange.
  - It is quoted in USDT, not USD. Close, not identical.
  - Short-term divergence from spot happens: crypto liquidity is thinner and
    fragmented, and issuers charge fees that show up in the tracking.

For what this system does -- discussing structure, levels, momentum, and where
price is relative to a range -- that is entirely usable, and the numbers are
close enough that the commentary is honest. For quoting an official gold fix,
it is not, and `describe_source()` exists so the persona can say what it is
watching rather than implying a spot quote.

ONE REAL ADVANTAGE OVER SPOT: this trades 24/7. Spot gold is shut roughly 48
hours a week, and the soak tests showed that weekend gap is the hardest part of
running continuously. Tokenized gold keeps printing through it.

ONE PRICE SERIES, NOT TWO. The tokens do not trade at the same number -- a
live check showed PAXG and XAUT about $11 apart, which is normal given
different issuers, liquidity and fee structures. Emitting both as ticks would
hand the engine an $11 sawtooth on alternate updates, and the detectors would
read that as constant volatility and phantom sweeps. So one token is the price
series and the other is only ever a sanity check. Averaging them was rejected:
it produces a synthetic series belonging to neither instrument, with its own
artifacts around each side's liquidity gaps.

Cross-checking: a quote is rejected when the secondary disagrees with the
primary by more than a tolerance. A bad print, a thin book or one side lagging
is exactly the input that would make the host say something wrong with
confidence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from platform_.market.feeds import Feed, Tick

log = logging.getLogger(__name__)

# Public market data endpoint. Explicitly documented as requiring no API key.
BINANCE_PUBLIC_WS = "wss://data-stream.binance.vision/stream"

TOKENS = {
    "paxg": {
        "stream": "paxgusdt@bookTicker",
        "label": "PAX Gold (PAXG/USDT)",
        "issuer": "Paxos",
    },
    "xaut": {
        "stream": "xautusdt@bookTicker",
        "label": "Tether Gold (XAUT/USDT)",
        "issuer": "Tether",
    },
}


class ExchangeGoldFeed(Feed):
    """Real-time gold via tokenized gold on Binance's public stream."""

    name = "exchange"

    # Measured against the live stream: PAXG top-of-book updates come in
    # bursts with multi-second gaps between them. Treating a 7s gap as
    # "delayed" made the host stop quoting prices during entirely normal
    # quiet periods.
    expected_interval_s = 5.0

    def __init__(
        self,
        symbols: list[str] | None = None,
        url: str = BINANCE_PUBLIC_WS,
        #: Reject a quote if the two tokens disagree by more than this, in USD.
        #: Normal divergence is small; a large gap means one side is bad.
        cross_check_tolerance: float = 25.0,
        max_backoff_s: float = 60.0,
        stale_after_s: float = 90.0,
    ) -> None:
        self.symbols = [s.lower() for s in (symbols or ["paxg", "xaut"])]
        unknown = [s for s in self.symbols if s not in TOKENS]
        if unknown:
            raise ValueError(f"unknown token(s) {unknown}; known: {sorted(TOKENS)}")

        # The first symbol is the price series; any others are sanity checks
        # only. See the module docstring -- interleaving them produces a
        # sawtooth the detectors read as volatility.
        self.primary = self.symbols[0]

        self.url = url
        self.cross_check_tolerance = cross_check_tolerance
        self.max_backoff_s = max_backoff_s
        self.stale_after_s = stale_after_s

        self._running = True
        self.reconnects = 0
        self.rejected_divergent = 0
        self.ticks_emitted = 0
        #: Latest quote per token, for cross-checking.
        self.latest: dict[str, tuple[float, float, datetime]] = {}

    # -- description ------------------------------------------------------

    def describe_source(self) -> str:
        """What the host is actually watching. Goes into the market context so
        the persona can be accurate rather than implying a spot quote."""
        primary = TOKENS[self.primary]["label"]
        checks = [TOKENS[s]["label"] for s in self.symbols if s != self.primary]
        checked = f", cross-checked against {' and '.join(checks)}" if checks else ""
        return (
            f"Price is tracked from tokenized gold ({primary}){checked} on a "
            "public exchange feed, not an interbank XAUUSD quote. It follows "
            "spot closely but is not an official fix."
        )

    @property
    def stream_url(self) -> str:
        streams = "/".join(TOKENS[s]["stream"] for s in self.symbols)
        return f"{self.url}?streams={streams}"

    # -- parsing ----------------------------------------------------------

    @staticmethod
    def _token_for(stream: str) -> str | None:
        for token, meta in TOKENS.items():
            if meta["stream"] == stream:
                return token
        return None

    def parse(self, raw: str) -> tuple[str, float, float] | None:
        """One bookTicker frame -> (token, bid, ask)."""
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            return None

        # Combined streams wrap the payload; single streams do not.
        data = envelope.get("data", envelope)
        stream = envelope.get("stream", "")
        token = self._token_for(stream)
        if token is None:
            symbol = str(data.get("s", "")).lower()
            token = next(
                (t for t in TOKENS if symbol.startswith(t)), None
            )
        if token is None:
            return None

        try:
            bid = float(data["b"])
            ask = float(data["a"])
        except (KeyError, TypeError, ValueError):
            return None
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        return token, bid, ask

    def _cross_check(self, token: str, bid: float, ask: float, at: datetime) -> bool:
        """Reject a quote the other token disagrees with.

        A thin book, a bad print or one exchange lagging is precisely the input
        that produces a confident host saying something wrong.
        """
        self.latest[token] = (bid, ask, at)
        if len(self.symbols) < 2 or len(self.latest) < 2:
            return True

        mids = [
            (b + a) / 2
            for _t, (b, a, seen) in self.latest.items()
            if (at - seen).total_seconds() < self.stale_after_s
        ]
        if len(mids) < 2:
            return True  # the other side is stale; nothing to compare against

        if max(mids) - min(mids) > self.cross_check_tolerance:
            self.rejected_divergent += 1
            log.warning(
                "tokenized gold quotes diverge by %.2f (%s); rejecting the tick",
                max(mids) - min(mids),
                ", ".join(f"{t}={(b + a) / 2:.2f}" for t, (b, a, _s) in self.latest.items()),
            )
            return False
        return True

    # -- loop -------------------------------------------------------------

    async def ticks(self) -> AsyncIterator[Tick]:  # type: ignore[override]
        import websockets

        log.info("free gold feed: %s", self.describe_source())
        backoff = 1.0

        while self._running:
            reason: str | None = None
            try:
                async with websockets.connect(self.stream_url) as ws:
                    log.info("connected to %s", self.stream_url)
                    received = 0
                    async for raw in ws:
                        parsed = self.parse(
                            raw if isinstance(raw, str) else raw.decode()
                        )
                        if parsed is None:
                            continue
                        token, bid, ask = parsed
                        at = datetime.now(timezone.utc)
                        if not self._cross_check(token, bid, ask, at):
                            continue

                        # Only the primary token becomes a tick. The others are
                        # recorded by _cross_check for sanity checking and then
                        # discarded -- emitting both would hand the engine a
                        # sawtooth between two instruments priced differently.
                        if token != self.primary:
                            continue

                        received += 1
                        if received == 1:
                            # Reset backoff only once real data arrives, so a
                            # connection that opens and immediately closes does
                            # not defeat the backoff.
                            backoff = 1.0
                        self.ticks_emitted += 1
                        yield Tick(bid=round(bid, 2), ask=round(ask, 2), at=at)

                # Binance closes public connections after about 24 hours. That
                # is expected, not an error -- but it still needs backoff, or a
                # server that closes politely gets reconnected to in a loop.
                reason = "stream closed (expected roughly every 24h)"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = str(exc) or exc.__class__.__name__

            if not self._running:
                return
            self.reconnects += 1
            log.warning("gold feed dropped (%s); reconnecting in %.0fs", reason, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff_s)

    async def close(self) -> None:
        self._running = False
