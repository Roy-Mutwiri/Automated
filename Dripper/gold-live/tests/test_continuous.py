"""Continuous mode: the host should not go quiet unless stopped.

The original design treated silence as an acceptable outcome -- candidates were
scored and the host spoke only when one cleared a floor. Measured over a soak
that produced roughly one utterance every three minutes, which is right for a
system that speaks when it has something to say and wrong for a live stream
where someone is watching.

Continuous mode waives the floor and the minimum gap once the host has been
quiet too long. Two things still hold, because breaking them makes the stream
worse rather than better: the host never talks over itself, and nothing unsafe
is spoken to fill time.
"""

from __future__ import annotations

from datetime import timedelta

from intelligence.director import Director, SpeechIntent
from intelligence.memory import SessionMemory
from shared.contracts import Priority, TriggerType, utcnow


def build(max_silence_s: float | None = 12.0, **kw):
    memory = SessionMemory("SESSION_001")
    return Director("SESSION_001", memory, max_silence_s=max_silence_s, **kw), memory


def intent(priority=Priority.LOW, topic="edu:filler", **kw) -> SpeechIntent:
    return SpeechIntent(
        trigger=TriggerType.EDUCATION, priority=priority, topic=topic, **kw
    )


# -- the core behaviour ----------------------------------------------------


def test_weak_candidate_is_spoken_once_overdue():
    """The whole point: a low-scoring filler beats dead air."""
    director, memory = build(max_silence_s=10.0)
    now = utcnow()
    memory.record_utterance("Something earlier.", "mkt:a", now)

    weak = intent(Priority.LOW, ttl_s=600.0, created_at=now)
    director.offer(weak)

    # Before the deadline the floor still applies.
    assert director.tick(now + timedelta(seconds=3)) is None

    decision = director.tick(now + timedelta(seconds=20))
    assert decision is not None
    assert any("OVERDUE" in r for r in decision.reasons)


def test_minimum_gap_is_waived_when_overdue():
    director, memory = build(max_silence_s=10.0, min_gap_s=60.0)
    now = utcnow()
    memory.record_utterance("Just spoke.", "mkt:a", now)
    director.offer(intent(Priority.LOW, ttl_s=600.0, created_at=now))

    assert director.tick(now + timedelta(seconds=5)) is None, "gap still applies early"
    assert director.tick(now + timedelta(seconds=30)) is not None


def test_silence_is_still_allowed_when_disabled():
    """Continuous mode is opt-in; the original behaviour must be reachable."""
    director, memory = build(max_silence_s=None)
    now = utcnow()
    memory.record_utterance("Something.", "mkt:a", now)
    director.offer(intent(Priority.LOW, ttl_s=600.0, created_at=now))
    assert director.tick(now + timedelta(seconds=600)) is None


# -- what continuous mode must NOT break -----------------------------------


def test_never_talks_over_itself_even_when_overdue():
    """Filling silence is good; interrupting yourself to do it is not."""
    director, memory = build(max_silence_s=5.0)
    now = utcnow()
    speaking = intent(Priority.HIGH, topic="mkt:live")
    director.mark_speaking(speaking, duration_ms=30_000, now=now)
    director.offer(intent(Priority.LOW, ttl_s=600.0, created_at=now))

    assert director.tick(now + timedelta(seconds=20)) is None


def test_critical_still_preempts_when_overdue():
    director, memory = build(max_silence_s=5.0)
    now = utcnow()
    director.mark_speaking(intent(Priority.MEDIUM), duration_ms=30_000, now=now)
    director.offer(
        intent(Priority.CRITICAL, topic="mkt:bos", ttl_s=600.0, created_at=now)
    )
    decision = director.tick(now + timedelta(seconds=20))
    assert decision is not None and decision.preempts is not None


def test_expired_intents_are_still_dropped_when_overdue():
    """Being overdue is not a reason to say something stale -- a reaction to a
    move that has passed is worse than a pause."""
    director, memory = build(max_silence_s=5.0)
    now = utcnow()
    memory.record_utterance("Earlier.", "mkt:a", now)
    director.offer(intent(Priority.HIGH, ttl_s=10.0, created_at=now))

    assert director.tick(now + timedelta(seconds=120)) is None
    assert director.queue_depth == 0


def test_an_empty_queue_still_yields_nothing():
    """Waiving the floor cannot invent a candidate. This is why the runtime
    tops the queue up rather than relying on the Director alone."""
    director, _memory = build(max_silence_s=1.0)
    assert director.tick(utcnow() + timedelta(seconds=600)) is None


# -- the runtime keeps the queue fed ---------------------------------------


async def test_runtime_tops_up_the_queue_in_continuous_mode(tmp_path):
    from intelligence.content import ContentPlanner, MarketPhase, load_content
    from intelligence.generation import GenerationResult, Generator
    from intelligence.personas import Persona
    from runtime.session import SessionRuntime
    from shared.contracts import (
        MarketConfidence,
        MarketState,
        PlatformBinding,
        Price,
        SessionState,
        TradingSession,
    )
    from shared.mocks.tts import MockTTS
    from shared.paths import resource_path

    class Fixed(Generator):
        async def generate(self, persona, intent, market, transcript):
            return GenerationResult(text="A line.", segments=["A line."])

    runtime = SessionRuntime(
        state=SessionState(
            session_id="SESSION_001", persona_id="p",
            platform_binding=PlatformBinding(platform="mock", channel_id="c"),
        ),
        persona=Persona(
            persona_id="p", display_name="P", audience="a", primary_timeframe="5m",
            focus=["x"], avoid=[], voice_id="v", style="s",
        ),
        generator=Fixed(), tts=MockTTS(), out_dir=tmp_path,
        planner=ContentPlanner(load_content(resource_path("configs", "content.yaml"))),
        max_silence_s=10.0,
    )

    now = utcnow()
    market = MarketState(
        as_of=now, computed_at=now, confidence=MarketConfidence.LIVE,
        price=Price(bid=4380.0, ask=4380.4), session=TradingSession.LONDON,
    )

    assert runtime.director.queue_depth == 0
    await runtime.keep_fed(MarketPhase.QUIET, market, now)
    assert runtime.director.queue_depth > 0, "a quiet market must still produce material"


async def test_no_top_up_when_continuous_is_off(tmp_path):
    from intelligence.content import MarketPhase
    from intelligence.generation import GenerationResult, Generator
    from intelligence.personas import Persona
    from runtime.session import SessionRuntime
    from shared.contracts import (
        MarketConfidence,
        MarketState,
        PlatformBinding,
        Price,
        SessionState,
        TradingSession,
    )
    from shared.mocks.tts import MockTTS

    class Fixed(Generator):
        async def generate(self, persona, intent, market, transcript):
            return GenerationResult(text="A line.", segments=["A line."])

    runtime = SessionRuntime(
        state=SessionState(
            session_id="SESSION_001", persona_id="p",
            platform_binding=PlatformBinding(platform="mock", channel_id="c"),
        ),
        persona=Persona(
            persona_id="p", display_name="P", audience="a", primary_timeframe="5m",
            focus=["x"], avoid=[], voice_id="v", style="s",
        ),
        generator=Fixed(), tts=MockTTS(), out_dir=tmp_path, max_silence_s=None,
    )
    now = utcnow()
    market = MarketState(
        as_of=now, computed_at=now, confidence=MarketConfidence.LIVE,
        price=Price(bid=1.0, ask=2.0), session=TradingSession.LONDON,
    )
    await runtime.keep_fed(MarketPhase.QUIET, market, now)
    assert runtime.director.queue_depth == 0
