"""Verify a price provider before trusting it with a live stream.

Fetches once, shows the raw response and what was extracted from it, and says
plainly whether the number looks like gold. Providers disagree about response
shape constantly, and a dotted path that silently resolves to the wrong field
gives you a confident host quoting a number that is not the gold price.

    python -m scripts.check_feed --provider goldapi
    python -m scripts.check_feed --provider metalprice --key YOUR_KEY
    python -m scripts.check_feed --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from platform_.market.rest_feed import RestPollingFeed, load_providers
from shared.paths import config_path

# Spot gold has traded in four figures for years. A value outside this is
# almost certainly the wrong field -- an ounces-per-dollar inverse, a rate in
# another currency, or a different metal entirely.
PLAUSIBLE_LOW = 500.0
PLAUSIBLE_HIGH = 20_000.0


async def check(provider: str, key: str | None, show_raw: bool) -> int:
    providers = load_providers(config_path("market_providers.yaml"))
    spec = providers.get(provider)
    if spec is None:
        print(f"Unknown provider {provider!r}. Known: {', '.join(sorted(providers))}")
        return 2

    api_key = key or os.environ.get("MARKET_API_KEY", "")
    print(f"\n  provider  {spec.name}")
    print(f"  url       {spec.url}")
    print(f"  cadence   {spec.describe_cadence()}")
    print(f"  key       {'set' if api_key else 'NOT SET'}")
    if spec.notes:
        print(f"  notes     {' '.join(spec.notes.split())}")
    print()

    feed = RestPollingFeed(spec, api_key=api_key)
    try:
        body = await feed._fetch()
    except Exception as exc:
        print(f"  REQUEST FAILED: {exc}\n")
        if not api_key:
            print("  No API key was sent. Set MARKET_API_KEY or pass --key.\n")
        return 1
    finally:
        await feed.close()

    if show_raw:
        print("  raw response:")
        print("    " + json.dumps(body, indent=2)[:1500].replace("\n", "\n    "))
        print()

    tick = feed.parse(body)
    if tick is None:
        print("  COULD NOT EXTRACT A PRICE.")
        print(f"  Looked for {spec.price_path!r} (and bid/ask if configured).")
        print("  Re-run with --raw and correct the paths in")
        print("  configs/market_providers.yaml.\n")
        return 1

    print(f"  bid       {tick.bid}")
    print(f"  ask       {tick.ask}")
    print(f"  mid       {tick.mid:.2f}")
    print(f"  spread    {tick.ask - tick.bid:.2f}")
    print(f"  as of     {tick.at.isoformat()}")
    print()

    if not (PLAUSIBLE_LOW < tick.mid < PLAUSIBLE_HIGH):
        print(f"  FAIL: {tick.mid:.2f} is not a plausible gold price.")
        print("  The path is almost certainly resolving to the wrong field --")
        print("  an inverted rate, another currency, or a different metal.")
        print("  Re-run with --raw and fix the paths.\n")
        return 1

    if spec.bid_path is None or spec.ask_path is None:
        print("  Note: this provider quotes a single price, so the spread above")
        print(f"  is synthesised ({spec.assumed_spread}). Fine for commentary;")
        print("  do not present it as a real dealable spread.\n")

    print(f"  PASS: {tick.mid:.2f} looks like spot gold.\n")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a market data provider")
    ap.add_argument("--provider")
    ap.add_argument("--key", help="overrides MARKET_API_KEY")
    ap.add_argument("--raw", action="store_true", help="print the full response")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    providers = load_providers(config_path("market_providers.yaml"))
    if args.list or not args.provider:
        print("\n  Configured providers:\n")
        for name, spec in sorted(providers.items()):
            print(f"    {name:<14} {spec.describe_cadence():<28} {spec.url[:52]}")
        print("\n  Check one:  python -m scripts.check_feed --provider <name>\n")
        raise SystemExit(0 if args.list else 2)

    raise SystemExit(asyncio.run(check(args.provider, args.key, args.raw)))


if __name__ == "__main__":
    main()
