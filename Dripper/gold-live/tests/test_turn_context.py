"""How the per-turn context is framed.

Written after measuring the real host against the real model. Production data
showed the failure plainly: of 1,439 utterances, 78% were discarded as
repetitive, and the openings clustered hard -- "the bid is now" 64 times, "the
price is now" 22, one stale level quoted verbatim 16 times.

The cause was framing, not model size. The market block put a concrete number
at the top of every turn, so a small model treated it as the subject and
narrated it -- it would discuss the spread no matter which topic it had been
asked for. Measured against llama3.2:3b, reframing that block took price-led
openings from 50% to 0% and distinct openings from 83% to 100%.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intelligence.director import SpeechIntent
from intelligence.generation import build_turn_context
from shared.contracts import (
    MarketConfidence,
    Price,
    Priority,
    TradingSession,
    TriggerType,
)
from shared.contracts import MarketState


@pytest.fixture
def live_market() -> MarketState:
    """A live, quotable market -- the state that produced the real failure."""
    now = datetime.now(timezone.utc)
    return MarketState(
        as_of=now, computed_at=now, confidence=MarketConfidence.LIVE,
        price=Price(bid=4434.61, ask=4434.62), session=TradingSession.NEW_YORK,
    )


def intent(trigger=TriggerType.EDUCATION, **payload) -> SpeechIntent:
    return SpeechIntent(
        trigger=trigger, priority=Priority.LOW, topic="edu:spreads",
        seed_text="spreads", ttl_s=300.0, created_at=datetime.now(timezone.utc),
        payload=payload or {"instruction": "Explain what a spread is."},
    )


def context(market, transcript=None, **kw) -> str:
    return build_turn_context(intent(**kw), market, transcript or [])


def test_market_data_is_framed_as_reference_not_as_the_subject(live_market):
    """The model narrated whatever concrete number sat at the top of this
    block. It has to be told the block is background."""
    text = context(live_market)
    assert "NOT the topic" in text
    assert "do not narrate" in text.lower()


def test_the_instruction_is_labelled_as_the_subject(live_market):
    text = context(live_market)
    assert "this is your subject" in text


def test_recent_openings_are_listed_so_they_can_be_avoided(live_market):
    """Listing whole previous utterances told the model what content to avoid
    but nothing about shape, so it varied content and reused construction."""
    transcript = [
        "The bid is now at 4434.61 and the ask is just above it.",
        "The price is now testing the session high after a quiet stretch.",
    ]
    text = context(live_market, transcript)
    assert "SENTENCE OPENINGS YOU HAVE ALREADY USED" in text
    assert "The bid is now at" in text
    assert "The price is now testing" in text


def test_no_opening_list_when_nothing_has_been_said(live_market):
    assert "SENTENCE OPENINGS" not in context(live_market)


def test_the_host_is_told_not_to_open_with_a_price(live_market):
    text = context(live_market)
    assert "HOW TO OPEN" in text
    for word in ("bid", "ask", "spread"):
        assert word in text.split("HOW TO OPEN")[1]


def test_a_viewer_question_is_exempt_from_the_opening_rule(live_market):
    """Answering "where is gold right now" by refusing to say a number would
    be absurd. The rule is about unprompted commentary."""
    text = context(
        live_market, trigger=TriggerType.COMMENT,
        text="where is gold right now?", intent="market_q", is_risk_sensitive=False,
    )
    assert "HOW TO OPEN" not in text


def test_the_price_is_still_available_when_it_may_be_quoted(live_market):
    """De-emphasising the number must not remove it: 72% of real utterances
    cite a price, and that grounding is the point of the product."""
    text = context(live_market)
    assert str(live_market.price.bid) in text
