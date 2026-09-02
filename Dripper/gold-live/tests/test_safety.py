"""The safety gate is a control, so it gets real tests."""

from __future__ import annotations

from datetime import timedelta

import pytest

from intelligence import safety
from shared.contracts import (
    MarketConfidence,
    MarketState,
    Price,
    TradingSession,
    utcnow,
)


def make_state(confidence: MarketConfidence, stale_s: float = 0.4) -> MarketState:
    now = utcnow()
    return MarketState(
        as_of=now - timedelta(seconds=stale_s),
        computed_at=now,
        confidence=confidence,
        price=Price(bid=3652.22, ask=3652.58),
        session=TradingSession.LONDON,
    )


def test_price_allowed_when_live():
    state = make_state(MarketConfidence.LIVE)
    v = safety.check("We're trading around 3652.40 right now, watching for a reaction.", state)
    assert v.report.stated_price
    assert v.report.passed


@pytest.mark.parametrize(
    "confidence", [MarketConfidence.DELAYED, MarketConfidence.STALE, MarketConfidence.UNAVAILABLE]
)
def test_price_blocked_when_not_live(confidence):
    """The core protection: no price quoting on stale data."""
    state = make_state(confidence, stale_s=45)
    v = safety.check("Gold is at 3652.40 as we speak.", state)
    assert not v.report.passed
    assert any("confidence" in x for x in v.report.violations)


def test_no_price_no_problem_when_stale():
    state = make_state(MarketConfidence.STALE, stale_s=45)
    v = safety.check(
        "One scenario I'm watching is acceptance above the prior high; "
        "if we reject there the bearish case becomes more relevant.",
        state,
    )
    assert v.report.passed


@pytest.mark.parametrize(
    "text",
    [
        "Gold will hit 3700 today.",
        "This is a guaranteed setup.",
        "You should buy now.",
        "It is 100% going higher.",
    ],
)
def test_certainty_language_blocked(text):
    v = safety.check(text, make_state(MarketConfidence.LIVE))
    assert not v.report.passed


def test_directional_claim_requires_hedging():
    state = make_state(MarketConfidence.LIVE)
    bare = safety.check("The bias is bullish from here.", state)
    assert not bare.report.passed

    hedged = safety.check(
        "If price accepts above that level the bullish scenario gets more interesting.", state
    )
    assert hedged.report.passed


def test_price_policy_note_changes_with_confidence():
    live = safety.price_policy_note(make_state(MarketConfidence.LIVE))
    stale = safety.price_policy_note(make_state(MarketConfidence.STALE, 45))
    assert "may reference exact levels" in live
    assert "Do NOT state any specific price" in stale
