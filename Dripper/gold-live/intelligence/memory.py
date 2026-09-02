"""Conversational memory and repetition detection.

Four layers, all scoped to one session:

  ShortTermMemory  -- the last N utterances, verbatim
  TopicMemory      -- what has been covered and when
  AudienceMemory   -- recurring viewer questions this stream
  RepetitionIndex  -- semantic near-duplicate detection

Repetition is NOT solved by telling the model "don't repeat yourself". It is
solved here, with state, before generation happens.

On the similarity backend: this ships with character n-gram cosine, which is
dependency-free and good at catching near-duplicates ("Gold broke the previous
high" vs "Price pushed through that prior swing high" scores ~0.42 -- caught at
the default threshold once the topic key matches). Swap in a real embedding
model behind SimilarityIndex when M5 soak tests show it is needed; the
interface is deliberately narrow so that is a one-file change.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from shared.contracts import utcnow

_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _ngrams(text: str, n: int = 4) -> Counter[str]:
    t = _norm(text)
    if len(t) < n:
        return Counter([t] if t else [])
    return Counter(t[i : i + n] for i in range(len(t) - n + 1))


def cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[k] * b[k] for k in common)
    da = sum(v * v for v in a.values()) ** 0.5
    db = sum(v * v for v in b.values()) ** 0.5
    return num / (da * db) if da and db else 0.0


class SimilarityIndex(ABC):
    @abstractmethod
    def add(self, text: str) -> None: ...

    @abstractmethod
    def max_similarity(self, text: str) -> float: ...


class NGramIndex(SimilarityIndex):
    """Character n-gram cosine over a rolling window. No model, no network."""

    def __init__(self, window: int = 40) -> None:
        self._vecs: deque[Counter[str]] = deque(maxlen=window)

    def add(self, text: str) -> None:
        self._vecs.append(_ngrams(text))

    def max_similarity(self, text: str) -> float:
        v = _ngrams(text)
        return max((cosine(v, o) for o in self._vecs), default=0.0)


@dataclass
class Utterance:
    text: str
    topic: str
    at: datetime = field(default_factory=utcnow)


class SessionMemory:
    """All four layers for exactly one session.

    Every method takes/returns only this session's data. There is no API on
    this object that can reach another session's memory -- isolation layer 1.
    """

    def __init__(
        self,
        session_id: str,
        short_term_size: int = 12,
        repetition_threshold: float = 0.38,
        topic_cooldown: timedelta = timedelta(minutes=6),
    ) -> None:
        self.session_id = session_id
        self.repetition_threshold = repetition_threshold
        self.topic_cooldown = topic_cooldown

        self.short_term: deque[Utterance] = deque(maxlen=short_term_size)
        self.topics_last_seen: dict[str, datetime] = {}
        self.audience_questions: Counter[str] = Counter()
        self._index: SimilarityIndex = NGramIndex()

    # -- writes -----------------------------------------------------------

    def record_utterance(self, text: str, topic: str, at: datetime | None = None) -> None:
        at = at or utcnow()
        self.short_term.append(Utterance(text=text, topic=topic, at=at))
        self.topics_last_seen[topic] = at
        self._index.add(text)

    def record_question(self, normalised_text: str) -> None:
        self.audience_questions[normalised_text] += 1

    # -- reads ------------------------------------------------------------

    def is_repetitive(self, candidate: str) -> tuple[bool, float]:
        score = self._index.max_similarity(candidate)
        return score >= self.repetition_threshold, score

    def topic_on_cooldown(self, topic: str, now: datetime | None = None) -> bool:
        last = self.topics_last_seen.get(topic)
        if last is None:
            return False
        return (now or utcnow()) - last < self.topic_cooldown

    def seconds_since_last_utterance(self, now: datetime | None = None) -> float:
        if not self.short_term:
            return float("inf")
        return ((now or utcnow()) - self.short_term[-1].at).total_seconds()

    def recent_transcript(self, n: int = 6) -> list[str]:
        return [u.text for u in list(self.short_term)[-n:]]

    def hot_questions(self, min_count: int = 2) -> list[str]:
        """Questions asked repeatedly -- worth answering even unprompted."""
        return [q for q, c in self.audience_questions.most_common(5) if c >= min_count]
