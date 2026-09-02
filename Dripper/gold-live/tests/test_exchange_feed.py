"""Free real-time gold feed via tokenized gold.

The only genuinely free real-time source found, so it is likely to be the one
actually used -- which makes its failure handling worth more attention than the
paid alternatives.

Most of these cover the ways a market feed lies rather than fails: a thin book,
a bad print, one side lagging. A feed that goes down is obvious and handled; a
feed that quietly returns a wrong number produces a confident host saying
something false, which is the failure that matters here.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from platform_.market.exchange_feed import TOKENS, ExchangeGoldFeed

websockets = pytest.importorskip("websockets")


def frame(token: str, bid: float, ask: float) -> str:
    return json.dumps({
        "stream": TOKENS[token]["stream"],
        "data": {"s": f"{token.upper()}USDT", "b": str(bid), "a": str(ask)},
    })


class StubExchange:
    """A websocket server that pushes bookTicker frames."""

    def __init__(self, messages: list[str], drop_after: int | None = None) -> None:
        self.messages = messages
        self.drop_after = drop_after
        self.connections = 0
        self._server = None
        self.url = ""

    async def _handler(self, ws):
        self.connections += 1
        try:
            for n, msg in enumerate(self.messages):
                if self.drop_after is not None and n >= self.drop_after:
                    await ws.close()
                    return
                await ws.send(msg)
            await asyncio.sleep(0.3)
        except Exception:
            pass

    async def __aenter__(self):
        self._server = await websockets.serve(self._handler, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc):
        if self._server:
            self._server.close()
            await self._server.wait_closed()


async def take(feed, n: int, timeout: float = 5.0):
    got = []

    async def run():
        async for tick in feed.ticks():
            got.append(tick)
            if len(got) >= n:
                return

    try:
        await asyncio.wait_for(run(), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        pass
    return got


# -- configuration ---------------------------------------------------------


def test_requires_no_api_key():
    """The entire reason this feed exists."""
    feed = ExchangeGoldFeed()
    assert "key" not in feed.stream_url.lower()
    assert "token=" not in feed.stream_url


def test_subscribes_to_both_tokens_by_default():
    url = ExchangeGoldFeed().stream_url
    assert "paxgusdt@bookTicker" in url
    assert "xautusdt@bookTicker" in url


def test_can_use_a_single_token():
    feed = ExchangeGoldFeed(symbols=["paxg"])
    assert "xaut" not in feed.stream_url


def test_unknown_token_fails_loudly():
    with pytest.raises(ValueError, match="unknown token"):
        ExchangeGoldFeed(symbols=["dogecoin"])


def test_source_description_is_honest():
    """The host must be able to say what it is watching rather than implying
    an interbank spot quote."""
    text = ExchangeGoldFeed().describe_source()
    assert "tokenized gold" in text
    assert "not an official fix" in text or "not an interbank" in text


# -- parsing ---------------------------------------------------------------


def test_parses_a_book_ticker_frame():
    feed = ExchangeGoldFeed()
    assert feed.parse(frame("paxg", 3652.20, 3652.60)) == ("paxg", 3652.20, 3652.60)


def test_parses_a_single_stream_frame_without_an_envelope():
    feed = ExchangeGoldFeed()
    raw = json.dumps({"s": "PAXGUSDT", "b": "3652.20", "a": "3652.60"})
    assert feed.parse(raw) == ("paxg", 3652.20, 3652.60)


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps({"stream": "paxgusdt@bookTicker", "data": {}}),
        json.dumps({"stream": "paxgusdt@bookTicker",
                    "data": {"s": "PAXGUSDT", "b": "x", "a": "1"}}),
        json.dumps({"stream": "unknown@thing", "data": {"b": "1", "a": "2"}}),
    ],
)
def test_bad_frames_are_skipped_not_fatal(raw):
    assert ExchangeGoldFeed().parse(raw) is None


def test_rejects_nonsensical_quotes():
    feed = ExchangeGoldFeed()
    assert feed.parse(frame("paxg", 0, 10)) is None, "zero bid"
    assert feed.parse(frame("paxg", -5, 10)) is None, "negative"
    assert feed.parse(frame("paxg", 100, 50)) is None, "ask below bid"


# -- cross-checking --------------------------------------------------------


def test_agreeing_quotes_pass():
    feed = ExchangeGoldFeed(cross_check_tolerance=25.0)
    now = datetime.now(timezone.utc)
    assert feed._cross_check("paxg", 3650.0, 3650.4, now)
    assert feed._cross_check("xaut", 3651.0, 3651.4, now)
    assert feed.rejected_divergent == 0


def test_divergent_quotes_are_rejected():
    """A thin book or a bad print on one exchange is exactly the input that
    makes a host say something wrong with confidence."""
    feed = ExchangeGoldFeed(cross_check_tolerance=25.0)
    now = datetime.now(timezone.utc)
    feed._cross_check("paxg", 3650.0, 3650.4, now)
    assert not feed._cross_check("xaut", 3900.0, 3900.4, now)
    assert feed.rejected_divergent == 1


def test_a_stale_counterpart_does_not_block_good_data():
    """If one token stops printing, the other must still be usable -- half a
    feed beats none."""
    feed = ExchangeGoldFeed(cross_check_tolerance=25.0, stale_after_s=30)
    old = datetime.now(timezone.utc) - timedelta(seconds=300)
    now = datetime.now(timezone.utc)
    feed._cross_check("paxg", 3000.0, 3000.4, old)
    assert feed._cross_check("xaut", 3650.0, 3650.4, now)


def test_single_token_mode_skips_cross_checking():
    feed = ExchangeGoldFeed(symbols=["paxg"])
    now = datetime.now(timezone.utc)
    assert feed._cross_check("paxg", 3650.0, 3650.4, now)


# -- streaming -------------------------------------------------------------


async def test_streams_ticks_over_a_real_websocket():
    messages = [frame("paxg", 3652.20, 3652.60), frame("paxg", 3653.00, 3653.40)]
    async with StubExchange(messages) as server:
        feed = ExchangeGoldFeed(symbols=["paxg"], url=server.url)
        got = await take(feed, 2)
        await feed.close()

    assert len(got) == 2
    assert got[0].bid == 3652.20
    assert got[0].mid == pytest.approx(3652.40)
    assert got[0].at.tzinfo is not None


async def test_reconnects_when_the_stream_closes():
    """Binance closes public connections roughly daily. Expected, not an error
    -- but it still needs backoff."""
    messages = [frame("paxg", 3650 + i, 3650.4 + i) for i in range(6)]
    async with StubExchange(messages, drop_after=2) as server:
        feed = ExchangeGoldFeed(symbols=["paxg"], url=server.url, max_backoff_s=0.2)
        got = await take(feed, 4, timeout=8.0)
        await feed.close()

    assert server.connections >= 2
    assert feed.reconnects >= 1
    assert len(got) >= 2


async def test_close_stops_the_reconnect_loop():
    feed = ExchangeGoldFeed(url="ws://127.0.0.1:1", max_backoff_s=0.1)
    task = asyncio.create_task(take(feed, 1, timeout=1.5))
    await asyncio.sleep(0.4)
    await feed.close()
    await task
    assert not feed._running


# -- into the engine -------------------------------------------------------


async def test_feeds_the_engine_and_unlocks_price_quoting():
    from platform_.market.engine import MarketEngine

    async with StubExchange([frame("paxg", 3652.20, 3652.60)]) as server:
        feed = ExchangeGoldFeed(symbols=["paxg"], url=server.url)
        got = await take(feed, 1)
        await feed.close()

    assert got
    engine = MarketEngine(timeframes={"5m": 300})
    engine.on_tick(got[0].bid, got[0].ask, got[0].at)
    state = engine.snapshot(got[0].at)

    assert state.may_quote_price(), "a fresh real tick unlocks price quoting"
    assert state.price.mid == pytest.approx(3652.40, abs=0.05)


def test_trades_through_the_weekend():
    """Spot gold is shut ~48h a week and the soak tests showed that gap is the
    hardest part of running continuously. Tokenized gold keeps printing."""
    from intelligence.content import market_is_closed

    saturday = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
    assert market_is_closed(saturday), "spot gold is shut"
    # The feed has no trading-hours logic at all, by design: the exchange runs
    # continuously, so there is nothing to gate on.
    assert not hasattr(ExchangeGoldFeed, "market_hours")


# -- one price series ------------------------------------------------------


async def test_only_the_primary_token_becomes_ticks():
    """The tokens trade about $11 apart. Emitting both would hand the engine a
    sawtooth between two differently-priced instruments, which the detectors
    would read as constant volatility and phantom sweeps."""
    messages = [
        frame("paxg", 4383.90, 4383.92),
        frame("xaut", 4373.00, 4373.02),
        frame("paxg", 4384.10, 4384.12),
        frame("xaut", 4373.20, 4373.22),
    ]
    async with StubExchange(messages) as server:
        feed = ExchangeGoldFeed(
            symbols=["paxg", "xaut"], url=server.url, cross_check_tolerance=50.0
        )
        got = await take(feed, 2, timeout=4.0)
        await feed.close()

    assert len(got) == 2
    # Every emitted tick is PAXG; XAUT only ever informed the cross-check.
    assert all(4383 < t.mid < 4385 for t in got), [t.mid for t in got]
    assert "xaut" in feed.latest, "the secondary is still tracked for checking"


async def test_price_series_has_no_artificial_sawtooth():
    """The regression this guards: consecutive ticks must not jump between two
    instruments' price levels."""
    messages = []
    for i in range(6):
        messages.append(frame("paxg", 4383.90 + i * 0.1, 4383.92 + i * 0.1))
        messages.append(frame("xaut", 4373.00, 4373.02))

    async with StubExchange(messages) as server:
        feed = ExchangeGoldFeed(
            symbols=["paxg", "xaut"], url=server.url, cross_check_tolerance=50.0
        )
        got = await take(feed, 5, timeout=5.0)
        await feed.close()

    jumps = [abs(b.mid - a.mid) for a, b in zip(got, got[1:], strict=False)]
    assert jumps and max(jumps) < 1.0, f"sawtooth detected: {jumps}"


def test_the_first_symbol_is_the_price_series():
    assert ExchangeGoldFeed(symbols=["xaut", "paxg"]).primary == "xaut"
    assert ExchangeGoldFeed(symbols=["paxg"]).primary == "paxg"


def test_source_description_names_the_primary_and_the_check():
    text = ExchangeGoldFeed(symbols=["paxg", "xaut"]).describe_source()
    assert "PAX Gold" in text
    assert "cross-checked against" in text
    assert "Tether Gold" in text


def test_staleness_thresholds_suit_the_instrument():
    """PAXG prints on top-of-book changes, so multi-second gaps are normal.
    A 5s 'delayed' threshold made the host stop quoting prices during entirely
    ordinary quiet periods."""
    feed = ExchangeGoldFeed()
    delayed, stale, unavailable = feed.staleness_thresholds()

    assert delayed > 20, "a 7s gap between book updates must not read as delayed"
    assert stale > delayed
    assert unavailable > stale


def test_a_faster_feed_gets_tighter_thresholds():
    from platform_.market.feeds import SyntheticFeed

    tick_by_tick = SyntheticFeed(interval_s=1.0)
    assert tick_by_tick.staleness_thresholds()[0] < ExchangeGoldFeed().staleness_thresholds()[0]
