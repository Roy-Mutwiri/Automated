"""Dynamic topic proposal and coverage memory.

The point of this layer is that nothing is pre-written. These tests check that
the model's proposals are actually filtered against what has been covered, and
that a dead model degrades rather than silencing the stream.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone


from intelligence.personas import Persona
from intelligence.proposer import CoverageMemory, TopicProposer
from platform_.llm.base import ChatMessage, LLMBackend, LLMResult
from shared.contracts import MarketConfidence, MarketState, Price, TradingSession

NOW = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)


class FakeLLM(LLMBackend):
    name = "fake"

    def __init__(self, reply: str = "", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls = 0

    async def stream(self, messages, **kw) -> AsyncIterator[str]:  # type: ignore[override]
        yield self.reply

    async def complete(self, messages: list[ChatMessage], **kw) -> LLMResult:
        self.calls += 1
        if self.fail:
            raise ConnectionError("model server down")
        return LLMResult(text=self.reply, model=self.name)

    async def health(self) -> bool:
        return not self.fail


def persona() -> Persona:
    return Persona(
        persona_id="p", display_name="P", audience="beginners",
        primary_timeframe="5m", focus=["structure"], avoid=[], voice_id="v", style="s",
    )


def market() -> MarketState:
    return MarketState(
        as_of=NOW, computed_at=NOW, confidence=MarketConfidence.LIVE,
        price=Price(bid=3652.2, ask=3652.6), session=TradingSession.LONDON,
    )


GOOD_REPLY = """
{"topic_key": "round_number_stops", "brief": "why a stop at a round number gets hit more often", "rationale": "quiet market", "category": "risk"}
{"topic_key": "asian_range", "brief": "how the Asian range becomes a London reference", "rationale": "session handover", "category": "liquidity"}
{"topic_key": "sizing_first", "brief": "deciding size before you are emotionally involved", "rationale": "evergreen", "category": "psychology"}
"""


# -- parsing --------------------------------------------------------------


async def test_parses_proposals():
    p = TopicProposer(FakeLLM(GOOD_REPLY), CoverageMemory())
    out = await p.propose(persona(), market(), "quiet", now=NOW)
    assert len(out) == 3
    assert out[0].topic_key == "round_number_stops"
    assert "round number" in out[0].brief


async def test_tolerates_prose_around_the_json():
    reply = "Here are some ideas:\n" + GOOD_REPLY + "\nHope that helps."
    p = TopicProposer(FakeLLM(reply), CoverageMemory())
    assert len(await p.propose(persona(), market(), "quiet", now=NOW)) == 3


async def test_ignores_malformed_entries():
    reply = '{"topic_key": "ok", "brief": "a usable brief"}\n{"broken": \n{"topic_key": "", "brief": "x"}'
    p = TopicProposer(FakeLLM(reply), CoverageMemory())
    out = await p.propose(persona(), market(), "quiet", now=NOW)
    assert [c.topic_key for c in out] == ["ok"]


def test_topic_keys_are_sanitised():
    from intelligence.proposer import TopicProposer as TP

    out = TP._parse('{"topic_key": "Round Number / Stops!", "brief": "b"}')
    assert out[0].topic_key == "round_number_stops_"


# -- coverage filtering ---------------------------------------------------


async def test_already_covered_proposals_are_rejected():
    """The model is told not to repeat. Being told is not a guarantee."""
    coverage = CoverageMemory()
    coverage.record(
        "stops", "why a stop at a round number gets hit more often", NOW
    )
    p = TopicProposer(FakeLLM(GOOD_REPLY), coverage)
    out = await p.propose(persona(), market(), "quiet", now=NOW)
    keys = [c.topic_key for c in out]
    assert "round_number_stops" not in keys
    assert len(out) == 2


async def test_topic_cooldown_rejects_recent_key():
    coverage = CoverageMemory(topic_cooldown=timedelta(hours=3))
    coverage.record("asian_range", "something unrelated entirely about ranges", NOW)
    p = TopicProposer(FakeLLM(GOOD_REPLY), coverage)
    out = await p.propose(persona(), market(), "quiet", now=NOW + timedelta(minutes=30))
    assert "asian_range" not in [c.topic_key for c in out]

    later = await p.propose(persona(), market(), "quiet", now=NOW + timedelta(hours=4))
    assert "asian_range" in [c.topic_key for c in later]


def test_coverage_survives_restart():
    a = CoverageMemory()
    a.record("k1", "first brief about liquidity pools", NOW)
    a.record("k2", "second brief about position sizing", NOW)
    b = CoverageMemory()
    b.load_state(a.export_state())
    assert b.on_cooldown("k1", NOW)
    covered, _ = b.is_covered("first brief about liquidity pools")
    assert covered


# -- failure --------------------------------------------------------------


async def test_dead_model_returns_nothing_rather_than_raising():
    """A proposal failure must never break the stream -- the caller falls back."""
    p = TopicProposer(FakeLLM(fail=True), CoverageMemory())
    assert await p.propose(persona(), market(), "quiet", now=NOW) == []
    assert p.failures == 1


async def test_consecutive_failures_are_counted():
    p = TopicProposer(FakeLLM(fail=True), CoverageMemory())
    for _ in range(3):
        await p.propose(persona(), market(), "quiet", now=NOW)
    assert p.failures == 3


# -- prompt context -------------------------------------------------------


async def test_closed_market_context_forbids_price_talk():
    llm = FakeLLM(GOOD_REPLY)
    p = TopicProposer(llm, CoverageMemory())

    captured: list[str] = []
    original = llm.complete

    async def spy(messages, **kw):
        captured.append(messages[-1].content)
        return await original(messages, **kw)

    llm.complete = spy  # type: ignore[method-assign]
    await p.propose(persona(), market(), "closed", now=NOW)
    assert "Market is closed" in captured[0]
    assert "no price action" in captured[0]


async def test_covered_briefs_are_shown_to_the_model():
    coverage = CoverageMemory()
    coverage.record("k", "a very distinctive brief about tokyo ranges", NOW)
    llm = FakeLLM(GOOD_REPLY)
    p = TopicProposer(llm, coverage)

    captured: list[str] = []
    original = llm.complete

    async def spy(messages, **kw):
        captured.append(messages[-1].content)
        return await original(messages, **kw)

    llm.complete = spy  # type: ignore[method-assign]
    await p.propose(persona(), market(), "quiet", now=NOW)
    assert "tokyo ranges" in captured[0]
    assert "ALREADY COVERED" in captured[0]
