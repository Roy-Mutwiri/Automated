"""Contract invariants. These are the interfaces between two people's work."""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from shared.contracts import (
    EXPORTED,
    AudioRequest,
    CommentEvent,
    MarketConfidence,
    MarketEvent,
    MarketEventKind,
    MarketState,
    Price,
    TradingSession,
    utcnow,
)


def test_every_contract_exports_a_json_schema():
    for name, model in EXPORTED.items():
        schema = model.model_json_schema()
        assert schema.get("properties"), f"{name} has no properties"


def test_price_derives_mid_and_spread():
    p = Price(bid=3652.22, ask=3652.58)
    assert p.mid == pytest.approx(3652.40)
    assert p.spread == pytest.approx(0.36)


def test_staleness_is_derived_not_supplied():
    now = utcnow()
    s = MarketState(
        as_of=now - timedelta(seconds=9),
        computed_at=now,
        confidence=MarketConfidence.STALE,
        price=Price(bid=1.0, ask=2.0),
        session=TradingSession.LONDON,
    )
    assert 8900 <= s.staleness_ms <= 9100
    assert not s.may_quote_price()


def test_may_quote_price_only_when_live():
    now = utcnow()
    kw = dict(
        as_of=now, computed_at=now, price=Price(bid=1.0, ask=2.0), session=TradingSession.ASIAN
    )
    assert MarketState(confidence=MarketConfidence.LIVE, **kw).may_quote_price()
    for c in (MarketConfidence.DELAYED, MarketConfidence.STALE, MarketConfidence.UNAVAILABLE):
        assert not MarketState(confidence=c, **kw).may_quote_price()


def test_severity_is_bounded():
    kw = dict(kind=MarketEventKind.BOS, timeframe="5m", occurred_at=utcnow())
    MarketEvent(severity=1, **kw)
    MarketEvent(severity=5, **kw)
    with pytest.raises(ValidationError):
        MarketEvent(severity=6, **kw)
    with pytest.raises(ValidationError):
        MarketEvent(severity=0, **kw)


def test_contracts_are_frozen():
    p = Price(bid=1.0, ask=2.0)
    with pytest.raises(ValidationError):
        p.bid = 5.0  # type: ignore[misc]


def test_extra_fields_rejected():
    """A typo in a field name must fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        Price(bid=1.0, ask=2.0, spred=0.1)  # type: ignore[call-arg]


def test_synth_msg_id_is_stable_and_collision_resistant():
    a = CommentEvent.synth_msg_id("zara", "where is resistance?")
    b = CommentEvent.synth_msg_id("zara", "where is resistance?")
    c = CommentEvent.synth_msg_id("zarawhere", " is resistance?")
    assert a == b, "same author+text must produce the same id across frames"
    assert a != c, "the null separator must prevent boundary collisions"


def test_author_is_hashed_not_stored():
    h = CommentEvent.hash_author("real_username", "salt")
    assert "real_username" not in h
    assert len(h) == 16


def test_audio_request_expires():
    req = AudioRequest(
        utterance_id=EXPORTED["ai_response"].model_fields["utterance_id"].default_factory(),
        session_id="SESSION_001",
        trace_id="t",
        segments=["hello"],
        voice_id="v",
        deadline_ms=1000,
    )
    assert not req.expired(req.created_at + timedelta(milliseconds=500))
    assert req.expired(req.created_at + timedelta(seconds=5))
