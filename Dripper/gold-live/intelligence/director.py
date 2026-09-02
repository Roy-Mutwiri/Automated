"""The Director: decides whether to speak, about what, and whether to interrupt.

This is deterministic code. The LLM writes the words; it does not choose the
moment. Letting a model decide when to talk makes latency, cost and behaviour
all unpredictable, and it is the difference between a stream that sounds alive
and one that sounds like a bot on a timer.

How it works:

  offer()  -- triggers push candidate SpeechIntents onto a queue
  tick()   -- called on a fixed cadence; scores every live candidate and
              returns at most one to speak, or None

Scoring is: base priority, decayed by age, penalised by topic cooldown and
semantic repetition, boosted by silence. A candidate that scores below the
floor is held; one that expires is dropped. Nothing is spoken twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from shared.contracts import Priority, TriggerType, utcnow

log = logging.getLogger(__name__)


@dataclass
class SpeechIntent:
    """A candidate thing to say. Not yet words -- just an intent and its context."""

    trigger: TriggerType
    priority: Priority
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    intent_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)
    #: After this, saying it is worse than silence.
    ttl_s: float = 90.0
    source_event_id: UUID | None = None
    #: Preview text used for repetition scoring before generation. Cheap.
    seed_text: str = ""

    def age_s(self, now: datetime) -> float:
        return (now - self.created_at).total_seconds()

    def expired(self, now: datetime) -> bool:
        return self.age_s(now) > self.ttl_s


@dataclass(slots=True)
class Decision:
    intent: SpeechIntent
    score: float
    preempts: UUID | None
    reasons: list[str]


class Director:
    """One per session. Holds no reference to any other session."""

    #: Below this, hold rather than speak. Prevents filler chatter.
    SCORE_FLOOR = 0.35

    BASE = {
        Priority.CRITICAL: 1.00,
        Priority.HIGH: 0.72,
        Priority.MEDIUM: 0.48,
        Priority.LOW: 0.26,
    }

    def __init__(
        self,
        session_id: str,
        memory: Any,
        *,
        min_gap_s: float = 8.0,
        silence_boost_after_s: float = 25.0,
        max_queue: int = 64,
    ) -> None:
        self.session_id = session_id
        self.memory = memory
        self.min_gap_s = min_gap_s
        self.silence_boost_after_s = silence_boost_after_s
        self._queue: list[SpeechIntent] = []
        self._max_queue = max_queue
        self._speaking: SpeechIntent | None = None
        self._speaking_until: datetime | None = None
        self._spoken_count = 0

    # -- input ------------------------------------------------------------

    def offer(self, intent: SpeechIntent) -> None:
        if len(self._queue) >= self._max_queue:
            # Drop the weakest rather than the newest -- a critical market
            # event arriving into a full queue must not be discarded.
            self._queue.sort(key=lambda i: i.priority)
            dropped = self._queue.pop(0)
            log.debug("queue full, dropped %s/%s", dropped.trigger, dropped.topic)
        self._queue.append(intent)

    # -- speaking state ---------------------------------------------------

    def mark_speaking(self, intent: SpeechIntent, duration_ms: int, now: datetime | None = None) -> None:
        now = now or utcnow()
        self._speaking = intent
        self._speaking_until = now + timedelta(milliseconds=duration_ms)
        self._spoken_count += 1

    def is_speaking(self, now: datetime | None = None) -> bool:
        if self._speaking_until is None:
            return False
        return (now or utcnow()) < self._speaking_until

    def _may_interrupt(self, candidate: SpeechIntent) -> bool:
        """Only a CRITICAL event interrupts, and only a lower-priority utterance.

        Without this the host talks over itself constantly, which is the single
        most bot-like failure mode there is.
        """
        if self._speaking is None:
            return True
        if candidate.priority is not Priority.CRITICAL:
            return False
        return self._speaking.priority < Priority.CRITICAL

    # -- scoring ----------------------------------------------------------

    def score(self, intent: SpeechIntent, now: datetime) -> tuple[float, list[str]]:
        reasons: list[str] = []
        s = self.BASE[intent.priority]
        reasons.append(f"base={s:.2f} ({intent.priority.name})")

        # Age decay: relevance falls off over the intent's lifetime.
        decay = max(0.0, 1.0 - (intent.age_s(now) / intent.ttl_s) * 0.6)
        s *= decay
        reasons.append(f"age_decay x{decay:.2f}")

        # Topic cooldown: we covered this recently.
        if self.memory.topic_on_cooldown(intent.topic, now):
            s *= 0.30
            reasons.append("topic_cooldown x0.30")

        # Semantic repetition against what we have actually said.
        if intent.seed_text:
            repetitive, sim = self.memory.is_repetitive(intent.seed_text)
            if repetitive:
                s *= 0.25
                reasons.append(f"repetitive x0.25 (sim={sim:.2f})")

        # Silence boost: dead air is its own problem, and it compounds. The cap
        # is deliberately high enough that a LOW-priority filler can clear the
        # floor on its own after a long quiet stretch -- a stream that says
        # nothing for three minutes is worse than one that says something
        # ordinary. Anything below ~2.2 here makes that impossible arithmetically.
        quiet_s = self.memory.seconds_since_last_utterance(now)
        if quiet_s > self.silence_boost_after_s:
            boost = min(2.5, 1.0 + (quiet_s - self.silence_boost_after_s) / 60.0)
            s *= boost
            reasons.append(f"silence x{boost:.2f} ({quiet_s:.0f}s quiet)")

        # Direct viewer questions matter more than ambient commentary.
        if intent.trigger is TriggerType.COMMENT:
            s *= 1.15
            reasons.append("viewer_question x1.15")

        return min(s, 2.0), reasons

    # -- the decision -----------------------------------------------------

    def tick(self, now: datetime | None = None) -> Decision | None:
        now = now or utcnow()

        before = len(self._queue)
        self._queue = [i for i in self._queue if not i.expired(now)]
        if (dropped := before - len(self._queue)) > 0:
            log.debug("dropped %d expired intents", dropped)
        if not self._queue:
            return None

        scored = [(self.score(i, now), i) for i in self._queue]
        scored.sort(key=lambda pair: pair[0][0], reverse=True)
        (best_score, reasons), best = scored[0]

        if best_score < self.SCORE_FLOOR:
            return None

        speaking = self.is_speaking(now)
        if speaking and not self._may_interrupt(best):
            return None

        # Minimum gap between utterances, unless this is critical.
        if not speaking and best.priority is not Priority.CRITICAL:
            quiet = self.memory.seconds_since_last_utterance(now)
            if quiet < self.min_gap_s:
                return None

        self._queue.remove(best)
        preempts = (
            self._speaking.intent_id if speaking and self._speaking is not None else None
        )
        if preempts:
            reasons.append("PREEMPTS current utterance")
        return Decision(intent=best, score=best_score, preempts=preempts, reasons=reasons)

    # -- introspection ----------------------------------------------------

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def spoken_count(self) -> int:
        return self._spoken_count
