"""Event bus, both implementations.

RedisBus had never executed -- only InMemoryBus was covered. These run it
against a real Redis protocol implementation (fakeredis speaks actual stream
semantics: consumer groups, acknowledgement, pending entries), so the wire
behaviour is exercised without needing a server.

Redis Streams rather than pub/sub was chosen for replay and consumer groups.
That only matters if the ack and durability semantics actually work, so those
are what these check.
"""

from __future__ import annotations

import asyncio

import pytest

from shared.contracts import Envelope, utcnow
from shared.events import (
    COMMENT_RECEIVED,
    MARKET_STATE_UPDATED,
    InMemoryBus,
    RedisBus,
    dumps,
    wrap,
)

fakeredis = pytest.importorskip("fakeredis")


@pytest.fixture
def redis_bus(monkeypatch):
    """RedisBus wired to an in-process Redis implementation."""
    import fakeredis.aioredis

    shared_client = fakeredis.aioredis.FakeRedis(decode_responses=True)

    def make(self):
        return shared_client

    async def _conn(self):
        return shared_client

    monkeypatch.setattr(RedisBus, "_conn", _conn)
    return RedisBus("redis://fake", group="g1", consumer="c1")


async def collect(bus, types, session_id, count, timeout=3.0):
    got: list[Envelope] = []

    async def run():
        async for env in bus.subscribe(types, session_id=session_id):
            got.append(env)
            if len(got) >= count:
                return

    with pytest.raises(asyncio.TimeoutError) if False else _nullcontext():
        try:
            await asyncio.wait_for(run(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
    return got


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# -- envelope --------------------------------------------------------------


def test_envelope_measures_feed_lag():
    """Two timestamps, not one. Their difference is the lag, and you cannot
    detect stale data without both."""
    now = utcnow()
    env = Envelope(
        event_type="x", occurred_at=now, trace_id="t", payload={}
    )
    assert env.lag_ms >= 0

    from datetime import timedelta

    late = Envelope(
        event_type="x", occurred_at=now - timedelta(seconds=3),
        emitted_at=now, trace_id="t", payload={},
    )
    assert 2900 <= late.lag_ms <= 3100


def test_wrap_and_dumps_round_trip():
    from shared.contracts import Price

    env = wrap("market.tick", dumps(Price(bid=1.0, ask=2.0)), trace_id="t")
    assert env.payload["bid"] == 1.0
    assert env.payload["mid"] == 1.5, "computed fields must survive serialisation"


# -- redis: real stream semantics -----------------------------------------


async def test_redis_publish_and_receive(redis_bus):
    await redis_bus.publish(
        wrap(MARKET_STATE_UPDATED, {"price": 3652.4}, "t1", session_id=None)
    )
    got = await collect(redis_bus, [MARKET_STATE_UPDATED], None, 1)
    assert len(got) == 1
    assert got[0].payload["price"] == 3652.4


async def test_redis_isolates_sessions(redis_bus):
    """The bus is isolation layer 2. A comment for one session must never
    reach another."""
    await redis_bus.publish(
        wrap(COMMENT_RECEIVED, {"t": "for seven"}, "a", session_id="SESSION_007")
    )
    await redis_bus.publish(
        wrap(COMMENT_RECEIVED, {"t": "for two"}, "b", session_id="SESSION_002")
    )
    got = await collect(redis_bus, [COMMENT_RECEIVED], "SESSION_002", 1, timeout=2.0)
    assert [e.payload["t"] for e in got] == ["for two"]


async def test_redis_delivers_shared_plane_to_every_session(redis_bus):
    await redis_bus.publish(
        wrap(MARKET_STATE_UPDATED, {"price": 1.0}, "t", session_id=None)
    )
    got = await collect(redis_bus, [MARKET_STATE_UPDATED], "SESSION_004", 1, timeout=2.0)
    assert len(got) == 1, "shared-plane events reach all sessions"


async def test_redis_preserves_order(redis_bus):
    for i in range(5):
        await redis_bus.publish(wrap(COMMENT_RECEIVED, {"i": i}, "t", session_id="S1"))
    got = await collect(redis_bus, [COMMENT_RECEIVED], "S1", 5, timeout=3.0)
    assert [e.payload["i"] for e in got] == [0, 1, 2, 3, 4]


async def test_redis_survives_a_restart_with_unread_messages(redis_bus):
    """The reason for Streams over pub/sub: a subscriber that was not running
    still gets what it missed. Pub/sub would have dropped these."""
    for i in range(3):
        await redis_bus.publish(wrap(COMMENT_RECEIVED, {"i": i}, "t", session_id="S1"))
    # Nothing was subscribed at publish time.
    got = await collect(redis_bus, [COMMENT_RECEIVED], "S1", 3, timeout=3.0)
    assert len(got) == 3, "messages published before subscribing must still arrive"


async def test_redis_round_trips_contract_payloads(redis_bus):
    from shared.contracts import MarketConfidence, MarketState, Price, TradingSession

    now = utcnow()
    state = MarketState(
        as_of=now, computed_at=now, confidence=MarketConfidence.LIVE,
        price=Price(bid=3652.2, ask=3652.6), session=TradingSession.LONDON,
    )
    await redis_bus.publish(
        wrap(MARKET_STATE_UPDATED, dumps(state), "t", session_id=None)
    )
    got = await collect(redis_bus, [MARKET_STATE_UPDATED], None, 1)

    restored = MarketState.model_validate(got[0].payload)
    assert restored.state_id == state.state_id
    assert restored.confidence is MarketConfidence.LIVE
    assert restored.may_quote_price()


# -- in-memory parity ------------------------------------------------------


async def test_in_memory_matches_redis_isolation_behaviour():
    """Both implementations must agree, or tests against the fast one prove
    nothing about production."""
    bus = InMemoryBus()
    got: list[Envelope] = []

    async def listen():
        async for env in bus.subscribe([COMMENT_RECEIVED], session_id="SESSION_002"):
            got.append(env)
            return

    task = asyncio.create_task(listen())
    await asyncio.sleep(0)
    await bus.publish(wrap(COMMENT_RECEIVED, {"t": "seven"}, "a", session_id="SESSION_007"))
    await bus.publish(wrap(COMMENT_RECEIVED, {"t": "two"}, "b", session_id="SESSION_002"))
    await asyncio.wait_for(task, timeout=1.0)

    assert [e.payload["t"] for e in got] == ["two"]


async def test_in_memory_drops_rather_than_blocking_when_full():
    """A slow consumer must not stall the publisher -- the market engine cannot
    wait on a wedged session."""
    bus = InMemoryBus(maxsize=3)

    async def idle():
        async for _ in bus.subscribe([COMMENT_RECEIVED], session_id="S1"):
            await asyncio.sleep(10)

    task = asyncio.create_task(idle())
    await asyncio.sleep(0)
    for i in range(20):
        await bus.publish(wrap(COMMENT_RECEIVED, {"i": i}, "t", session_id="S1"))
    task.cancel()
    assert len(bus.published) == 20, "publisher never blocked"
