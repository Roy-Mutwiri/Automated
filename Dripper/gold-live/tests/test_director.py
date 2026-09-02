"""Director behaviour. This is the component that decides how the stream sounds."""

from __future__ import annotations

from datetime import timedelta

from intelligence.director import Director, SpeechIntent
from intelligence.memory import SessionMemory
from shared.contracts import Priority, TriggerType, utcnow


def make_director(**kw) -> tuple[Director, SessionMemory]:
    mem = SessionMemory("SESSION_001")
    return Director("SESSION_001", mem, **kw), mem


def intent(priority=Priority.MEDIUM, topic="mkt:bos", trigger=TriggerType.MARKET_EVENT, **kw):
    return SpeechIntent(trigger=trigger, priority=priority, topic=topic, **kw)


def test_empty_queue_says_nothing():
    d, _ = make_director()
    assert d.tick() is None


def test_higher_priority_wins():
    d, _ = make_director()
    d.offer(intent(Priority.LOW, topic="edu:sizing"))
    d.offer(intent(Priority.CRITICAL, topic="mkt:bos"))
    decision = d.tick()
    assert decision is not None
    assert decision.intent.topic == "mkt:bos"


def test_expired_intents_are_dropped_not_spoken():
    """A reaction that arrives 40s late is worse than silence."""
    d, _ = make_director()
    d.offer(intent(Priority.HIGH, ttl_s=10.0))
    later = utcnow() + timedelta(seconds=30)
    assert d.tick(later) is None
    assert d.queue_depth == 0


def test_topic_cooldown_suppresses_repeat_topic():
    d, mem = make_director()
    now = utcnow()
    mem.record_utterance("Break of structure above the prior high.", "mkt:bos", now)
    d.offer(intent(Priority.MEDIUM, topic="mkt:bos"))
    # Scored well below the floor because the topic is on cooldown.
    score, reasons = d.score(d._queue[0], now)
    assert any("cooldown" in r for r in reasons)
    assert score < Director.SCORE_FLOOR


def test_semantic_repetition_suppressed():
    d, mem = make_director()
    now = utcnow()
    mem.record_utterance("Gold broke the previous high.", "mkt:other", now)
    candidate = intent(
        Priority.HIGH, topic="mkt:new", seed_text="Gold broke the previous high."
    )
    d.offer(candidate)
    _score, reasons = d.score(candidate, now)
    assert any("repetitive" in r for r in reasons)


def test_does_not_interrupt_itself():
    d, _ = make_director()
    now = utcnow()
    speaking = intent(Priority.HIGH, topic="mkt:a")
    d.mark_speaking(speaking, duration_ms=10_000, now=now)
    d.offer(intent(Priority.HIGH, topic="mkt:b"))
    assert d.tick(now) is None, "a non-critical intent must not interrupt"


def test_critical_preempts_lower_priority_speech():
    d, _ = make_director()
    now = utcnow()
    d.mark_speaking(intent(Priority.MEDIUM, topic="edu:x"), duration_ms=10_000, now=now)
    d.offer(intent(Priority.CRITICAL, topic="mkt:bos"))
    decision = d.tick(now)
    assert decision is not None
    assert decision.preempts is not None


def test_critical_does_not_preempt_critical():
    d, _ = make_director()
    now = utcnow()
    d.mark_speaking(intent(Priority.CRITICAL, topic="mkt:a"), duration_ms=10_000, now=now)
    d.offer(intent(Priority.CRITICAL, topic="mkt:b"))
    assert d.tick(now) is None


def test_min_gap_between_utterances():
    d, mem = make_director(min_gap_s=30.0)
    now = utcnow()
    mem.record_utterance("Something.", "mkt:a", now)
    d.offer(intent(Priority.HIGH, topic="mkt:b"))
    assert d.tick(now + timedelta(seconds=5)) is None
    assert d.tick(now + timedelta(seconds=45)) is not None


def test_silence_boosts_weak_candidates():
    """After a long quiet stretch, low-priority filler must be able to clear
    the floor on its own. Dead air is worse than ordinary commentary."""
    d, mem = make_director(silence_boost_after_s=20.0)
    now = utcnow()
    mem.record_utterance("Something.", "edu:old", now)
    # ttl matches how SessionRuntime.offer_filler actually creates these.
    weak = intent(Priority.LOW, topic="edu:new", trigger=TriggerType.SILENCE, ttl_s=300.0)
    d.offer(weak)
    quiet = now + timedelta(seconds=180)
    score, reasons = d.score(weak, quiet)
    assert any("silence" in r for r in reasons)
    assert score > Director.SCORE_FLOOR


def test_expired_filler_still_dropped_despite_silence():
    """The silence boost must not resurrect genuinely stale intents."""
    d, mem = make_director(silence_boost_after_s=20.0)
    now = utcnow()
    mem.record_utterance("Something.", "edu:old", now)
    d.offer(intent(Priority.LOW, topic="edu:new", trigger=TriggerType.SILENCE, ttl_s=60.0))
    assert d.tick(now + timedelta(seconds=300)) is None


def test_full_queue_drops_weakest_not_newest():
    d, _ = make_director(max_queue=4)
    for _ in range(4):
        d.offer(intent(Priority.LOW, topic="edu:filler"))
    d.offer(intent(Priority.CRITICAL, topic="mkt:bos"))
    assert any(i.priority is Priority.CRITICAL for i in d._queue)
