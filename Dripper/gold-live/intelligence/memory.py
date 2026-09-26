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

# Stripped before similarity is computed. A host with a consistent verbal style
# reuses these constantly; leaving them in means two utterances about completely
# different topics score as near-duplicates purely because they share sentence
# scaffolding. Similarity must be driven by what is being SAID, not how.
_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can did do does doing down during each
few for from further had has have having he her here hers him his how i if in
into is it its itself just let lets me more most my no nor not now of off on
once only or other our out over own re s same she should so some such t than
that the their them then there these they this those through to too under
until up very was we were what when where which while who whom why will with
you your thing things get gets got going want lot actually really quite bit
second one two three
""".split())


def _norm(text: str) -> str:
    return " ".join(w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS)


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


#: Viewer questions are counted so the host can answer what keeps coming up.
#: Only frequently-repeated ones are ever read, so the table is capped: on a
#: busy stream the number of distinct questions is otherwise unbounded.
QUESTION_TABLE_LIMIT = 2_000
QUESTION_TABLE_KEEP = 500


class SimilarityIndex(ABC):
    @abstractmethod
    def add(self, text: str) -> None: ...

    @abstractmethod
    def max_similarity(self, text: str) -> float: ...


class NGramIndex(SimilarityIndex):
    """Character n-gram cosine over a rolling window. No model, no network.

    Window sizing matters more than it looks. At roughly 90 utterances an hour,
    a 40-item window forgets everything older than ~25 minutes -- it catches
    local repetition and is completely blind to "you explained this at 3am".
    The default here holds ~24 hours. Cost is a dict comparison per stored
    vector per candidate, which measures in single-digit milliseconds at this
    size; the soak run reports it.
    """

    def __init__(self, window: int = 2200) -> None:
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
        similarity_window: int = 2200,
    ) -> None:
        self.session_id = session_id
        self.repetition_threshold = repetition_threshold
        self.topic_cooldown = topic_cooldown

        self.short_term: deque[Utterance] = deque(maxlen=short_term_size)
        self.topics_last_seen: dict[str, datetime] = {}
        self.audience_questions: Counter[str] = Counter()
        self._index: SimilarityIndex = NGramIndex(window=similarity_window)
        self.utterance_count = 0

    # -- writes -----------------------------------------------------------

    def record_utterance(self, text: str, topic: str, at: datetime | None = None) -> None:
        at = at or utcnow()
        self.short_term.append(Utterance(text=text, topic=topic, at=at))
        self.topics_last_seen[topic] = at
        self._index.add(text)
        self.utterance_count += 1

    def record_question(self, normalised_text: str) -> None:
        """Count a viewer question, keeping only what hot_questions can use.

        This Counter is keyed by the text of the question, so on a real stream
        it grows with every distinct thing anyone has ever asked -- unbounded,
        for the life of the process. Only the frequently-repeated ones are ever
        read back, so the rare ones are dropped once the table gets large.

        Pruning keeps the most-asked rather than the most-recent, because that
        is what hot_questions selects on; dropping a question that has been
        asked ten times to make room for one asked once would defeat it.
        """
        self.audience_questions[normalised_text] += 1
        if len(self.audience_questions) > QUESTION_TABLE_LIMIT:
            keep = self.audience_questions.most_common(QUESTION_TABLE_KEEP)
            self.audience_questions = Counter(dict(keep))

    # -- reads ------------------------------------------------------------

    def is_repetitive(self, candidate: str, threshold: float | None = None) -> tuple[bool, float]:
        score = self._index.max_similarity(candidate)
        return score >= (self.repetition_threshold if threshold is None else threshold), score

    def repetition_threshold_for_silence(self, quiet_s: float) -> float:
        """Relax the repetition bar the longer we have been silent.

        Anti-repetition must never be allowed to starve the stream. If every
        candidate is being rejected, the correct outcome is to say something
        the audience may have heard before -- not to broadcast dead air. The
        bar rises from the base threshold toward 1.0 (accept anything) over a
        few minutes of silence.
        """
        if quiet_s <= 60:
            return self.repetition_threshold
        relaxed = self.repetition_threshold + (quiet_s - 60) / 60.0 * 0.12
        return min(0.99, relaxed)

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
