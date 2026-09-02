"""REST polling feed for XAUUSD.

Most accessible gold price APIs are REST rather than websocket -- streaming
usually sits behind a paid or trial plan. This polls one, on a schedule it can
actually sustain, and emits the same Tick every other feed does.

RATE LIMIT IS THE DESIGN CONSTRAINT, exactly as quota was for YouTube, and the
arithmetic is worth doing before choosing a plan:

    a 1,000 requests/month plan  -> one call every ~45 minutes
    a 10,000/month plan          -> one call every ~4.5 minutes
    a 100,000/month plan         -> one call every ~26 seconds
    a per-minute plan (60/min)   -> one call per second

For a host that reacts to price action, anything slower than about 30 seconds
is not really live commentary -- the staleness machine will mark the data
`delayed` or `stale` and the host will correctly stop quoting levels. That is
the system behaving properly, but it makes for a worse stream, so pick a plan
that matches the cadence you want rather than discovering it at 3am.

This feed self-paces to make its allowance last the whole billing period, the
same way the YouTube adapter does. Running slower all month beats running fast
and going blind on the 9th.

Providers are configured in `configs/market_providers.yaml`, not in code, so
adding one is an edit rather than a release.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from platform_.market.feeds import Feed, Tick

log = logging.getLogger(__name__)

SECONDS_PER_MONTH = 30 * 86_400


def dig(data: Any, path: str) -> Any:
    """Pull a value out by dotted path, e.g. 'rates.XAU' or 'data.0.price'.

    Providers disagree about response shape far more than about content, so the
    shape lives in config rather than in a class per provider.
    """
    current = data
    for part in path.split("."):
        if current is None:
            return None
        if part.isdigit() and isinstance(current, list):
            index = int(part)
            current = current[index] if index < len(current) else None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


@dataclass
class ProviderSpec:
    """How to talk to one price API. Loaded from YAML."""

    name: str
    url: str
    #: Dotted paths into the response.
    price_path: str = "price"
    bid_path: str | None = None
    ask_path: str | None = None
    timestamp_path: str | None = None
    #: "header" puts the key in `auth_name`; "query" puts it in the query string.
    auth_style: str = "query"
    auth_name: str = "api_key"
    headers: dict[str, str] = field(default_factory=dict)
    #: Requests allowed per month on the chosen plan. Drives self-pacing.
    monthly_limit: int = 1000
    #: Some plans are per-minute instead; this wins when set.
    min_interval_s: float | None = None
    #: When a provider quotes a single price with no bid/ask, synthesise a
    #: spread so downstream code has both. Typical XAUUSD retail spread.
    assumed_spread: float = 0.36
    notes: str = ""

    @property
    def sustainable_interval_s(self) -> float:
        if self.min_interval_s is not None:
            return self.min_interval_s
        return SECONDS_PER_MONTH / max(1, self.monthly_limit)

    def describe_cadence(self) -> str:
        s = self.sustainable_interval_s
        if s < 60:
            return f"one poll every {s:.0f}s"
        if s < 3600:
            return f"one poll every {s / 60:.1f} min"
        return f"one poll every {s / 3600:.1f} hours"


class RestPollingFeed(Feed):
    name = "rest"

    def __init__(
        self,
        spec: ProviderSpec,
        api_key: str | None = None,
        interval_s: float | None = None,
        transport=None,
    ) -> None:
        self.spec = spec
        self.api_key = api_key or os.environ.get("MARKET_API_KEY", "")
        # Never poll faster than the plan sustains, even if asked to.
        self.interval_s = max(
            interval_s or spec.sustainable_interval_s, spec.sustainable_interval_s
        )
        self._transport = transport
        self._running = True
        self._client: Any = None

        self.polls = 0
        self.errors = 0
        self.last_tick_at: datetime | None = None

    # -- transport --------------------------------------------------------

    async def _fetch(self) -> dict:
        if self._transport is not None:
            return await self._transport(self.spec, self.api_key)

        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0, connect=5.0)
            )

        headers = dict(self.spec.headers)
        params: dict[str, str] = {}
        if self.api_key:
            if self.spec.auth_style == "header":
                headers[self.spec.auth_name] = self.api_key
            else:
                params[self.spec.auth_name] = self.api_key

        resp = await self._client.get(self.spec.url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()

    # -- parsing ----------------------------------------------------------

    def parse(self, body: dict) -> Tick | None:
        bid = dig(body, self.spec.bid_path) if self.spec.bid_path else None
        ask = dig(body, self.spec.ask_path) if self.spec.ask_path else None

        if bid is None or ask is None:
            price = dig(body, self.spec.price_path)
            if price is None:
                log.warning(
                    "%s: no price at %r in response", self.spec.name, self.spec.price_path
                )
                return None
            try:
                mid = float(price)
            except (TypeError, ValueError):
                log.warning("%s: price %r is not a number", self.spec.name, price)
                return None
            # A single quote with no spread is normal for gold REST APIs.
            # Synthesising one keeps every downstream consumer uniform rather
            # than making them special-case a missing bid.
            half = self.spec.assumed_spread / 2
            bid, ask = mid - half, mid + half

        try:
            bid_f, ask_f = float(bid), float(ask)
        except (TypeError, ValueError):
            return None
        if bid_f <= 0 or ask_f <= 0:
            return None

        at = datetime.now(timezone.utc)
        if self.spec.timestamp_path:
            raw = dig(body, self.spec.timestamp_path)
            parsed = _parse_timestamp(raw)
            if parsed is not None:
                at = parsed

        return Tick(bid=round(bid_f, 2), ask=round(ask_f, 2), at=at)

    # -- loop -------------------------------------------------------------

    async def ticks(self) -> AsyncIterator[Tick]:  # type: ignore[override]
        log.info(
            "market feed %s: %s (%s)",
            self.spec.name, self.spec.describe_cadence(), self.spec.url,
        )
        backoff = self.interval_s
        while self._running:
            started = time.perf_counter()
            try:
                tick = self.parse(await self._fetch())
                self.polls += 1
                backoff = self.interval_s
                if tick is not None:
                    self.last_tick_at = tick.at
                    yield tick
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.errors += 1
                log.warning(
                    "%s poll failed (%s); retrying in %.0fs",
                    self.spec.name, exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 600.0)
                continue

            elapsed = time.perf_counter() - started
            await asyncio.sleep(max(0.0, self.interval_s - elapsed))

    async def close(self) -> None:
        self._running = False
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _parse_timestamp(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        # Providers use both seconds and milliseconds; tell them apart by size.
        seconds = raw / 1000 if raw > 1e11 else raw
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_providers(path) -> dict[str, ProviderSpec]:
    import yaml
    from pathlib import Path

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: dict[str, ProviderSpec] = {}
    for name, cfg in (data.get("providers") or {}).items():
        out[name] = ProviderSpec(name=name, **cfg)
    return out
