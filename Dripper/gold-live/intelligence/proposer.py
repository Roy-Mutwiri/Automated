"""Dynamic topic proposal.

The previous design walked a fixed YAML curriculum. Even with good generation on
top, a stream working through a pre-written topic list reads as scripted --
the sequencing is the tell, not the wording. Regulars notice that the host
covers the same ground in the same order every night.

This inverts the relationship:

    coverage memory  =  what has ALREADY been said (prevents repeats)
    proposer         =  what to say NEXT (generated from live context)

Nothing is pre-written. The model is asked what would actually be worth saying
right now given the market, the audience, and everything already covered, and
it answers differently every time because the context is different every time.

The YAML inventory survives in exactly one role: a cold-start and fallback seed
for when the LLM is unreachable. A degraded stream reading from a list beats a
silent one -- but it is the emergency path, not the design.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from intelligence.memory import NGramIndex, SimilarityIndex
from intelligence.personas import Persona
from platform_.llm.base import ChatMessage, LLMBackend
from shared.contracts import MarketState, utcnow

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TopicCandidate:
    #: Short slug, used for cooldown tracking. Model-generated, not from a list.
    topic_key: str
    #: What to actually talk about. Becomes the generation brief.
    brief: str
    #: Why this is worth saying now. Goes into the trace so "why did it say
    #: that?" has an answer even for unprompted commentary.
    rationale: str
    category: str = "general"

    @property
    def topic(self) -> str:
        return f"gen:{self.topic_key}"


class CoverageMemory:
    """What this session has already covered. Semantic, not a checklist.

    Deliberately separate from SessionMemory's utterance index: that one guards
    against repeating PHRASING over a short window, this one guards against
    revisiting SUBJECT MATTER over many hours. Different horizons, different
    thresholds.
    """

    def __init__(
        self,
        window: int = 800,
        similarity_threshold: float = 0.42,
        topic_cooldown: timedelta = timedelta(hours=3),
    ) -> None:
        self._index: SimilarityIndex = NGramIndex(window=window)
        self.threshold = similarity_threshold
        self.topic_cooldown = topic_cooldown
        self.covered: list[tuple[str, str, datetime]] = []  # key, brief, at
        self.topic_last_seen: dict[str, datetime] = {}

    def record(self, key: str, brief: str, at: datetime | None = None) -> None:
        at = at or utcnow()
        self._index.add(brief)
        self.covered.append((key, brief, at))
        self.topic_last_seen[key] = at

    def is_covered(self, brief: str) -> tuple[bool, float]:
        sim = self._index.max_similarity(brief)
        return sim >= self.threshold, sim

    def on_cooldown(self, key: str, now: datetime | None = None) -> bool:
        last = self.topic_last_seen.get(key)
        if last is None:
            return False
        return (now or utcnow()) - last < self.topic_cooldown

    def recent_briefs(self, n: int = 25) -> list[str]:
        return [b for _k, b, _a in self.covered[-n:]]

    def export_state(self) -> dict:
        return {
            "covered": [(k, b, a.isoformat()) for k, b, a in self.covered[-500:]],
        }

    def load_state(self, state: dict) -> None:
        for k, b, a in state.get("covered", []):
            self.record(k, b, datetime.fromisoformat(a))


PROPOSER_SYSTEM = """You plan segments for a live Gold (XAUUSD) trading stream.

You do not write the segment. You decide WHAT is worth talking about next, and \
you are given everything the host has already covered so you can avoid \
retreading it.

Return between 3 and 5 candidates as JSON, one object per line, no other text:
{"topic_key": "short_slug", "brief": "what to talk about, one sentence", \
"rationale": "why now, one short clause", "category": "structure|liquidity|risk|\
psychology|macro|gold|process|audience"}

Rules:
- Do not repeat anything in ALREADY COVERED, in substance or in framing.
- Vary the category from the last few segments.
- Briefs must be specific enough to speak from. "Talk about risk" is useless; \
"why a stop placed at a round number gets hit more often" is usable.
- When the market is closed or flat, lean into education, audience questions, \
psychology and process. Do not invent price action to talk about.
- When something is genuinely happening in the market, prefer it.
- Never propose predicting a price, giving trade signals, or personal advice."""


class TopicProposer:
    """Asks the local model what is worth saying next.

    Not on the latency-critical path -- proposals are fetched ahead of time and
    queued, so a slow proposal never delays speech.
    """

    def __init__(
        self,
        llm: LLMBackend,
        coverage: CoverageMemory,
        max_candidates: int = 5,
    ) -> None:
        self.llm = llm
        self.coverage = coverage
        self.max_candidates = max_candidates
        self.failures = 0

    def _context(
        self,
        persona: Persona,
        market: MarketState,
        phase: str,
        audience_questions: list[str],
    ) -> str:
        tf = market.timeframes.get("5m")
        lines = [
            f"HOST: {persona.display_name} for {persona.audience}, "
            f"primary timeframe {persona.primary_timeframe}.",
            f"Covers: {', '.join(persona.focus)}.",
            "",
            f"MARKET PHASE: {phase}",
            f"DATA CONFIDENCE: {market.confidence.value}",
        ]
        if phase != "closed" and tf:
            lines += [
                f"5m trend: {tf.trend.value}, structure: {tf.structure.value}, atr {tf.atr}",
            ]
            if market.detections:
                lines.append("Detected: " + "; ".join(d.label for d in market.detections))
        else:
            lines.append("Market is closed. There is no price action to discuss.")

        if audience_questions:
            lines += ["", "RECENT AUDIENCE INTEREST:"]
            lines += [f"  - {q}" for q in audience_questions[:6]]

        covered = self.coverage.recent_briefs(25)
        lines += ["", "ALREADY COVERED (do not retread):"]
        lines += [f"  - {c}" for c in covered] if covered else ["  (nothing yet)"]
        return "\n".join(lines)

    async def propose(
        self,
        persona: Persona,
        market: MarketState,
        phase: str,
        audience_questions: list[str] | None = None,
        now: datetime | None = None,
    ) -> list[TopicCandidate]:
        now = now or utcnow()
        messages = [
            ChatMessage(role="system", content=PROPOSER_SYSTEM),
            ChatMessage(
                role="user",
                content=self._context(persona, market, phase, audience_questions or []),
            ),
        ]
        try:
            result = await self.llm.complete(
                messages, max_tokens=500, temperature=1.0
            )
            self.failures = 0
        except Exception as exc:
            self.failures += 1
            log.warning("topic proposal failed (%d consecutive): %s", self.failures, exc)
            return []

        return self._filter(self._parse(result.text), now)

    @staticmethod
    def _parse(text: str) -> list[TopicCandidate]:
        out: list[TopicCandidate] = []
        for match in re.finditer(r"\{[^{}]*\}", text):
            try:
                d = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            key = str(d.get("topic_key", "")).strip()
            brief = str(d.get("brief", "")).strip()
            if not key or not brief:
                continue
            out.append(
                TopicCandidate(
                    topic_key=re.sub(r"[^a-z0-9_]+", "_", key.lower())[:48],
                    brief=brief,
                    rationale=str(d.get("rationale", "")).strip(),
                    category=str(d.get("category", "general")).strip() or "general",
                )
            )
        return out

    def _filter(self, candidates: list[TopicCandidate], now: datetime) -> list[TopicCandidate]:
        """Drop anything already covered. The model is told not to repeat, but
        being told is not a guarantee -- this is the enforcement."""
        kept: list[TopicCandidate] = []
        for c in candidates:
            if self.coverage.on_cooldown(c.topic_key, now):
                continue
            covered, sim = self.coverage.is_covered(c.brief)
            if covered:
                log.debug("proposal rejected as covered (sim=%.2f): %s", sim, c.brief[:60])
                continue
            kept.append(c)
            if len(kept) >= self.max_candidates:
                break
        return kept
