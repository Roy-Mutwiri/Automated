"""REST price feed.

Providers disagree about response shape far more than about content, so most
of these are about extracting the right number from the wrong-looking JSON --
and about refusing to guess when it cannot.

A dotted path that silently resolves to the wrong field is the dangerous
failure: it produces a confident host quoting a number that is not the gold
price. That is why the paths are configuration and why check_feed exists.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from platform_.market.rest_feed import (
    ProviderSpec,
    RestPollingFeed,
    dig,
    load_providers,
)
from shared.paths import config_path


def spec(**kw) -> ProviderSpec:
    base = {
        "name": "test", "url": "http://example/price",
        "price_path": "price", "monthly_limit": 999_999_999,
        "min_interval_s": 0.01,
    }
    return ProviderSpec(**{**base, **kw})


def feed(response: dict, **kw) -> RestPollingFeed:
    async def transport(_spec, _key):
        return response

    return RestPollingFeed(spec(**kw), api_key="k", transport=transport)


# -- dotted paths ----------------------------------------------------------


@pytest.mark.parametrize(
    "data,path,expected",
    [
        ({"price": 3652.4}, "price", 3652.4),
        ({"rates": {"XAU": 3652.4}}, "rates.XAU", 3652.4),
        ({"data": [{"p": 1.5}]}, "data.0.p", 1.5),
        ({"a": {"b": {"c": 7}}}, "a.b.c", 7),
        ({"price": 1}, "missing", None),
        ({"a": 1}, "a.b", None),
        ({"data": []}, "data.0.p", None),
    ],
)
def test_dig(data, path, expected):
    assert dig(data, path) == expected


# -- extraction ------------------------------------------------------------


async def test_uses_bid_and_ask_when_the_provider_gives_them():
    f = feed(
        {"bid": 3652.2, "ask": 3652.6}, bid_path="bid", ask_path="ask"
    )
    tick = f.parse({"bid": 3652.2, "ask": 3652.6})
    assert tick is not None
    assert (tick.bid, tick.ask) == (3652.2, 3652.6)
    assert tick.mid == pytest.approx(3652.4)


async def test_synthesises_a_spread_from_a_single_price():
    """Most gold REST APIs quote one number. Downstream code should not have to
    special-case a missing bid."""
    f = feed({"price": 3652.40}, assumed_spread=0.40)
    tick = f.parse({"price": 3652.40})
    assert tick is not None
    assert tick.bid == 3652.20
    assert tick.ask == 3652.60
    assert tick.mid == pytest.approx(3652.40)


async def test_reads_a_nested_price():
    f = feed({}, price_path="rates.USDXAU")
    tick = f.parse({"rates": {"USDXAU": 3651.0}})
    assert tick is not None and tick.mid == pytest.approx(3651.0)


@pytest.mark.parametrize(
    "body",
    [
        {},                       # nothing at all
        {"price": None},          # explicit null
        {"price": "not a number"},
        {"price": 0},             # zero is never a valid quote
        {"price": -5},
    ],
)
async def test_refuses_to_guess(body):
    """Returning None means the engine's staleness clock keeps running and the
    host stops quoting prices. Inventing a number would be far worse."""
    assert feed(body).parse(body) is None


# -- timestamps ------------------------------------------------------------


async def test_parses_epoch_seconds():
    f = feed({}, timestamp_path="ts")
    tick = f.parse({"price": 3652.4, "ts": 1788249600})
    assert tick is not None and tick.at.year == 2026


async def test_parses_epoch_milliseconds():
    f = feed({}, timestamp_path="ts")
    tick = f.parse({"price": 3652.4, "ts": 1788249600000})
    assert tick is not None and tick.at.year == 2026


async def test_parses_iso_timestamps():
    f = feed({}, timestamp_path="ts")
    tick = f.parse({"price": 3652.4, "ts": "2026-09-01T08:00:00Z"})
    assert tick is not None
    assert tick.at == datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)


async def test_falls_back_to_now_for_an_unparseable_timestamp():
    """A bad timestamp must not lose the price; the staleness machine can
    still do its job from arrival time."""
    f = feed({}, timestamp_path="ts")
    tick = f.parse({"price": 3652.4, "ts": "yesterday-ish"})
    assert tick is not None and tick.at.tzinfo is not None


async def test_timestamps_are_always_timezone_aware():
    """A naive datetime downstream breaks every staleness comparison, which is
    the safety gate's input."""
    f = feed({}, timestamp_path="ts")
    tick = f.parse({"price": 3652.4, "ts": "2026-09-01T08:00:00"})
    assert tick is not None and tick.at.tzinfo is not None


# -- pacing ----------------------------------------------------------------


def test_monthly_limit_sets_the_cadence():
    assert ProviderSpec(name="a", url="u", monthly_limit=1000).sustainable_interval_s == \
        pytest.approx(2592.0)  # ~43 min
    fast = ProviderSpec(name="b", url="u", monthly_limit=100_000)
    assert 20 < fast.sustainable_interval_s < 30


def test_per_minute_plans_override_the_monthly_maths():
    s = ProviderSpec(name="c", url="u", monthly_limit=100, min_interval_s=1.0)
    assert s.sustainable_interval_s == 1.0


def test_cannot_be_asked_to_poll_faster_than_the_plan():
    """Otherwise a well-meaning --interval burns the month's allowance in a
    day and the host goes blind."""
    s = spec(monthly_limit=1000, min_interval_s=None)

    async def transport(_s, _k):
        return {"price": 1.0}

    f = RestPollingFeed(s, api_key="k", interval_s=0.5, transport=transport)
    assert f.interval_s == pytest.approx(s.sustainable_interval_s)


def test_cadence_is_described_in_human_terms():
    assert "min" in ProviderSpec(name="a", url="u", monthly_limit=10_000).describe_cadence()
    assert "hours" in ProviderSpec(name="b", url="u", monthly_limit=500).describe_cadence()
    assert "s" in ProviderSpec(name="c", url="u", min_interval_s=5).describe_cadence()


# -- loop ------------------------------------------------------------------


async def test_yields_ticks():
    f = feed({"price": 3652.4})
    got = []
    async for tick in f.ticks():
        got.append(tick)
        if len(got) >= 3:
            break
    await f.close()
    assert len(got) == 3
    assert all(t.mid == pytest.approx(3652.4) for t in got)


async def test_a_failing_request_is_retried_not_raised():
    calls = {"n": 0}

    async def flaky(_spec, _key):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("provider down")
        return {"price": 3652.4}

    f = RestPollingFeed(spec(), api_key="k", transport=flaky)
    got = []

    async def run():
        async for tick in f.ticks():
            got.append(tick)
            return

    await asyncio.wait_for(run(), timeout=5.0)
    await f.close()
    assert f.errors >= 2
    assert len(got) == 1, "recovers once the provider comes back"


# -- shipped config --------------------------------------------------------


def test_shipped_providers_load():
    providers = load_providers(config_path("market_providers.yaml"))
    assert providers, "no providers configured"
    for name, s in providers.items():
        assert s.url.startswith("http"), f"{name} has no usable url"
        assert s.price_path, f"{name} has no price path"
        assert s.sustainable_interval_s > 0


def test_every_shipped_provider_documents_its_cadence():
    for name, s in load_providers(config_path("market_providers.yaml")).items():
        assert s.describe_cadence(), name


def test_feeds_into_the_market_engine():
    """The whole point: real ticks reaching the engine make prices quotable."""
    from platform_.market.engine import MarketEngine

    engine = MarketEngine(timeframes={"5m": 300})
    f = feed({"price": 3652.4})
    tick = f.parse({"price": 3652.4})
    assert tick is not None

    engine.on_tick(tick.bid, tick.ask, tick.at)
    state = engine.snapshot(tick.at)
    assert state.may_quote_price(), "a fresh real tick should unlock price quoting"
    assert state.price.mid == pytest.approx(3652.4, abs=0.05)
