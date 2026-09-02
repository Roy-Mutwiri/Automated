"""End-to-end runtime behaviour: repetition policy and the safety gate.

These cover the two failure modes found by actually running the dry run:
a critical event being silently dropped as repetitive, and a stale-data
price quote reaching audio.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from intelligence.generation import GenerationResult, Generator
from intelligence.personas import Persona
from runtime.session import SessionRuntime
from shared.contracts import (
    MarketConfidence,
    MarketEvent,
    MarketEventKind,
    MarketState,
    PlatformBinding,
    Price,
    SessionState,
    TradingSession,
    utcnow,
)
from shared.mocks.tts import MockTTS


class FixedGenerator(Generator):
    """Always returns the same text -- forces the repetition path."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def generate(self, persona, intent, market, transcript) -> GenerationResult:
        self.calls += 1
        return GenerationResult(text=self.text, segments=[self.text])


def persona() -> Persona:
    return Persona(
        persona_id="t", display_name="T", audience="a", primary_timeframe="5m",
        focus=["x"], avoid=[], voice_id="v", style="s",
    )


def market(confidence=MarketConfidence.LIVE, stale_s=0.4) -> MarketState:
    now = utcnow()
    return MarketState(
        as_of=now - timedelta(seconds=stale_s), computed_at=now, confidence=confidence,
        price=Price(bid=3652.22, ask=3652.58), session=TradingSession.LONDON,
    )


def runtime(gen: Generator, tmp: Path) -> SessionRuntime:
    return SessionRuntime(
        state=SessionState(
            session_id="SESSION_001", persona_id="t",
            platform_binding=PlatformBinding(platform="mock", channel_id="c"),
        ),
        persona=persona(), generator=gen, tts=MockTTS(), out_dir=tmp,
    )


def event(severity: int) -> MarketEvent:
    return MarketEvent(
        kind=MarketEventKind.BOS, timeframe="5m", occurred_at=utcnow(),
        severity=severity, narrative_hint="Break of structure.",
    )


async def test_critical_event_spoken_even_if_repetitive(tmp_path):
    """A severity-5 event must never be silently dropped for repetition."""
    gen = FixedGenerator("If price accepts above there the bullish scenario is live.")
    rt = runtime(gen, tmp_path)
    now = utcnow()

    rt.on_market_event(event(5), now)
    first = await rt.tick(market(), now)
    assert first is not None

    # Same text again, now definitely repetitive.
    rt.on_market_event(event(5), now + timedelta(seconds=60))
    second = await rt.tick(market(), now + timedelta(seconds=60))
    assert second is not None, "CRITICAL must speak anyway"
    assert gen.calls == 3, "should have retried once before speaking anyway"


async def test_low_priority_repetition_is_dropped(tmp_path):
    gen = FixedGenerator("If price accepts above there the bullish scenario is live.")
    rt = runtime(gen, tmp_path)
    now = utcnow()

    rt.on_market_event(event(1), now)
    assert await rt.tick(market(), now) is not None

    rt.offer_filler("position sizing", now + timedelta(seconds=200))
    result = await rt.tick(market(), now + timedelta(seconds=200))
    assert result is None
    assert len(rt.dropped_repetitive) == 1


async def test_stale_market_blocks_price_quote_end_to_end(tmp_path):
    """The whole point of MarketState.confidence, tested through the runtime."""
    gen = FixedGenerator("Gold is sitting at 3652.40 right now.")
    rt = runtime(gen, tmp_path)
    now = utcnow()

    rt.on_market_event(event(5), now)
    result = await rt.tick(market(MarketConfidence.STALE, stale_s=45), now)

    assert result is None, "a price quote on stale data must never be spoken"
    assert len(rt.dropped_unsafe) == 1
    assert any("confidence" in v for v in rt.dropped_unsafe[0][1])


async def test_same_utterance_is_safe_when_data_is_live(tmp_path):
    gen = FixedGenerator("Gold is sitting at 3652.40 right now.")
    rt = runtime(gen, tmp_path)
    now = utcnow()
    rt.on_market_event(event(5), now)
    result = await rt.tick(market(MarketConfidence.LIVE), now)
    assert result is not None
    assert result.safety.stated_price
