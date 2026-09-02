"""Market feeds against real transports.

WebSocketFeed had never executed -- the same gap that hid a session-killing bug
in the Redis bus and a minute-long freeze in the LLM client. These run it
against an actual websocket server, including the failure modes that matter:
a provider that drops mid-session, sends malformed frames, or changes its
field names.

A feed dropping is not an exception to raise. The engine's staleness clock is
already running, the host has already stopped quoting prices, and this layer
just needs to keep trying quietly until data returns.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from datetime import datetime, timezone

import pytest

from platform_.market.feeds import ReplayFeed, SyntheticFeed, Tick, WebSocketFeed

websockets = pytest.importorskip("websockets")


class StubFeedServer:
    """A websocket server that pushes ticks and can misbehave on demand."""

    def __init__(
        self,
        messages: list[str] | None = None,
        drop_after: int | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.messages = messages or []
        self.drop_after = drop_after
        self.delay_s = delay_s
        self.connections = 0
        self.subscribed: list[dict] = []
        self._server = None
        self.url = ""

    async def _handler(self, ws):
        self.connections += 1
        try:
            # A subscribe frame may or may not arrive; do not block on it.
            try:
                first = await asyncio.wait_for(ws.recv(), timeout=0.05)
                self.subscribed.append(json.loads(first))
            except (TimeoutError, asyncio.TimeoutError, Exception):
                pass

            for n, message in enumerate(self.messages):
                if self.drop_after is not None and n >= self.drop_after:
                    await ws.close()
                    return
                await ws.send(message)
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
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


def tick_json(bid: float, ask: float, ts: str | int) -> str:
    return json.dumps({"bid": bid, "ask": ask, "timestamp": ts})


async def take(feed, n: int, timeout: float = 5.0) -> list[Tick]:
    got: list[Tick] = []

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


# -- websocket -------------------------------------------------------------


async def test_receives_ticks_over_a_real_websocket():
    messages = [
        tick_json(3652.2, 3652.6, "2026-09-01T08:00:00+00:00"),
        tick_json(3652.9, 3653.3, "2026-09-01T08:00:01+00:00"),
    ]
    async with StubFeedServer(messages) as server:
        feed = WebSocketFeed(server.url)
        got = await take(feed, 2)
        await feed.close()

    assert len(got) == 2
    assert got[0].bid == 3652.2
    assert got[0].mid == pytest.approx(3652.4)
    assert got[0].at.tzinfo is not None, "timestamps must be timezone-aware"


async def test_accepts_epoch_millisecond_timestamps():
    """Providers split roughly evenly between ISO strings and epoch millis."""
    async with StubFeedServer([tick_json(1.0, 2.0, 1788249600000)]) as server:
        feed = WebSocketFeed(server.url)
        got = await take(feed, 1)
        await feed.close()

    assert len(got) == 1
    assert got[0].at.year == 2026
    assert got[0].at.tzinfo is timezone.utc


async def test_field_map_adapts_a_providers_naming():
    """Adapting a provider must not require subclassing."""
    payload = json.dumps({"b": 3652.2, "a": 3652.6, "t": "2026-09-01T08:00:00Z"})
    async with StubFeedServer([payload]) as server:
        feed = WebSocketFeed(
            server.url, field_map={"bid": "b", "ask": "a", "timestamp": "t"}
        )
        got = await take(feed, 1)
        await feed.close()

    assert len(got) == 1 and got[0].bid == 3652.2


async def test_sends_the_subscribe_frame():
    subscribe = {"op": "subscribe", "args": ["XAUUSD"]}
    async with StubFeedServer([tick_json(1.0, 2.0, 1788249600000)]) as server:
        feed = WebSocketFeed(server.url, subscribe=subscribe)
        await take(feed, 1)
        await feed.close()

    assert server.subscribed and server.subscribed[0] == subscribe


async def test_malformed_frames_are_skipped_not_fatal():
    """One bad frame must not end the session's market data."""
    messages = [
        "not json at all",
        json.dumps({"bid": "oops", "ask": 2.0, "timestamp": 1788249600000}),
        json.dumps({"missing": "fields"}),
        tick_json(3652.2, 3652.6, 1788249600000),
    ]
    async with StubFeedServer(messages) as server:
        feed = WebSocketFeed(server.url)
        got = await take(feed, 1, timeout=4.0)
        await feed.close()

    assert len(got) == 1, "the one valid tick should still arrive"
    assert got[0].bid == 3652.2


async def test_reconnects_after_the_provider_drops():
    """The common failure: the provider closes mid-session. Reconnect quietly
    rather than surfacing an exception -- staleness handling has already
    stopped the host quoting prices."""
    messages = [tick_json(1.0 + i, 2.0 + i, 1788249600000 + i) for i in range(6)]
    async with StubFeedServer(messages, drop_after=2) as server:
        feed = WebSocketFeed(server.url, max_backoff_s=0.2)
        got = await take(feed, 4, timeout=8.0)
        await feed.close()

    assert server.connections >= 2, "must have reconnected after the drop"
    assert feed.reconnects >= 1
    assert len(got) >= 2


async def test_close_stops_the_reconnect_loop():
    """A feed that keeps reconnecting after shutdown prevents a clean exit."""
    feed = WebSocketFeed("ws://127.0.0.1:1", max_backoff_s=0.1)
    task = asyncio.create_task(take(feed, 1, timeout=1.5))
    await asyncio.sleep(0.4)
    await feed.close()
    await task
    assert not feed._running


# -- replay ----------------------------------------------------------------


async def test_replays_a_csv(tmp_path):
    path = tmp_path / "ticks.csv"
    path.write_text(
        "timestamp,bid,ask\n"
        "2026-09-01T08:00:00+00:00,3652.20,3652.60\n"
        "2026-09-01T08:00:01+00:00,3652.90,3653.30\n",
        encoding="utf-8",
    )
    got = await take(ReplayFeed(path), 2)
    assert len(got) == 2
    assert got[0].bid == 3652.20
    assert got[1].at > got[0].at


async def test_replay_assumes_utc_for_naive_timestamps(tmp_path):
    """A recorded file without offsets is the common case, and a naive
    datetime downstream breaks every staleness comparison."""
    path = tmp_path / "naive.csv"
    path.write_text(
        "timestamp,bid,ask\n2026-09-01T08:00:00,3652.20,3652.60\n", encoding="utf-8"
    )
    got = await take(ReplayFeed(path), 1)
    assert got[0].at.tzinfo is timezone.utc


async def test_replay_feeds_the_engine_end_to_end(tmp_path):
    """The point of replay: a recorded session is reproducible in a way a live
    feed never is, so detector work can be done against known data."""
    from platform_.market.engine import MarketEngine

    rows = ["timestamp,bid,ask"]
    base = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
    price = 3650.0
    for i in range(400):
        price += 0.6 if i % 3 else -0.4
        at = base.timestamp() + i * 30
        stamp = datetime.fromtimestamp(at, tz=timezone.utc).isoformat()
        rows.append(f"{stamp},{price - 0.18:.2f},{price + 0.18:.2f}")

    path = tmp_path / "session.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    engine = MarketEngine(timeframes={"5m": 300})
    last_at = base
    async for tick in ReplayFeed(path).ticks():
        engine.on_tick(tick.bid, tick.ask, tick.at)
        last_at = tick.at

    state = engine.snapshot(last_at)
    assert state.may_quote_price(), "replayed ticks are fresh relative to their own clock"
    assert "5m" in state.timeframes
    assert state.timeframes["5m"].atr is not None


# -- synthetic -------------------------------------------------------------


async def test_synthetic_feed_respects_its_interval():
    """It advanced simulated time by hours per wall second when this was
    wrong, which made every engine cooldown look already-elapsed."""
    feed = SyntheticFeed(interval_s=0.05)
    got = await take(feed, 4, timeout=3.0)
    await feed.close()

    assert len(got) >= 4
    gaps = [
        (b.at - a.at).total_seconds()
        for a, b in itertools.pairwise(got)
    ]
    assert all(abs(g - 0.05) < 1e-6 for g in gaps), "tick clock must match the interval"


async def test_clean_server_close_still_backs_off():
    """A provider that closes politely -- auth rejection, rate limiting -- must
    not be reconnected to in a hot loop. The stream ending without an exception
    is still a disconnect."""
    async with StubFeedServer([], drop_after=0) as server:
        feed = WebSocketFeed(server.url, max_backoff_s=0.3)
        await take(feed, 1, timeout=2.0)
        await feed.close()

    assert feed.reconnects >= 1, "a clean close must be counted as a disconnect"
    # Backoff bounds how often we can reconnect; without it this would be
    # hundreds of connections in two seconds.
    assert server.connections < 15, f"hot reconnect loop: {server.connections} connections"
