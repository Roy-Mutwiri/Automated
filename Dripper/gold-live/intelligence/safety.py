"""Financial-content safety gate.

This is a control with tests, not a tone-of-voice instruction. Two jobs:

  1. Block numeric price claims when market data is not LIVE. This is the
     mechanism that stops the AI confidently quoting a nine-minute-old price.
  2. Block outcome-certainty language ("gold WILL go up", "guaranteed").

The gate runs AFTER generation and BEFORE audio. A failed utterance is dropped
and optionally regenerated -- it is never spoken. Silence is always safer than
a confident false claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from shared.contracts import MarketState, SafetyReport

# A number that looks like a Gold price: 4 digits, optional decimals.
PRICE_RE = re.compile(r"\b\d{4}(?:\.\d{1,2})?\b")

# Outcome certainty. These are the phrasings that turn analysis into a promise.
CERTAINTY_PATTERNS = [
    (re.compile(r"\b(will|is going to|gonna)\s+(definitely\s+)?(hit|reach|go|rise|fall|drop|break)\b", re.I),
     "outcome stated as certain"),
    (re.compile(r"\bguarantee(d|s)?\b", re.I), "guarantee language"),
    (re.compile(r"\b(100%|certain(ly)?|no doubt|for sure)\b", re.I), "certainty language"),
    (re.compile(r"\b(buy|sell|long|short)\s+now\b", re.I), "direct trade instruction"),
    (re.compile(r"\byou should (buy|sell|go long|go short)\b", re.I), "personalised advice"),
]

# Language that correctly frames a scenario rather than a prediction.
HEDGE_RE = re.compile(
    r"\b(scenario|if |watching|one case|becomes more|invalidat|could|might|may |"
    r"tends to|historically|not advice)\b",
    re.I,
)


@dataclass(slots=True)
class SafetyVerdict:
    report: SafetyReport
    #: Text with violations removed where that is possible, else None.
    repaired: str | None


def check(text: str, market: MarketState, *, require_hedge_on_direction: bool = True) -> SafetyVerdict:
    violations: list[str] = []
    stated_price = bool(PRICE_RE.search(text))

    if stated_price and not market.may_quote_price():
        violations.append(
            f"quoted a price while market confidence is {market.confidence.value} "
            f"(stale by {market.staleness_ms}ms)"
        )

    for pattern, label in CERTAINTY_PATTERNS:
        if pattern.search(text):
            violations.append(label)

    directional = re.search(r"\b(bullish|bearish|upside|downside|higher|lower)\b", text, re.I)
    if require_hedge_on_direction and directional and not HEDGE_RE.search(text):
        violations.append("directional claim with no scenario framing")

    report = SafetyReport(
        passed=not violations,
        stated_price=stated_price,
        has_disclaimer=bool(HEDGE_RE.search(text)),
        violations=violations,
    )
    return SafetyVerdict(report=report, repaired=None)


def price_policy_note(market: MarketState) -> str:
    """The instruction injected into the prompt for the current data state."""
    if market.may_quote_price():
        return (
            f"Live pricing is available (bid {market.price.bid}, ask {market.price.ask}). "
            "You may reference exact levels."
        )
    return (
        f"MARKET DATA IS {market.confidence.value.upper()} "
        f"({market.staleness_ms}ms old). Do NOT state any specific price or level. "
        "Speak about structure, concepts, education or viewer questions instead, "
        "and if it is natural to do so, say plainly that you are waiting on a "
        "clean data feed before quoting levels."
    )
