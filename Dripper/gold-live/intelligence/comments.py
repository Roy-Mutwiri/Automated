"""Comment pipeline: normalise -> classify -> dedupe -> prioritise.

Not every comment reaches the generator. Most should not. This stage is
deliberately cheap: classification runs on Haiku, dedup and prioritisation
are plain code.
"""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass

from shared.contracts import (
    CommentClassification,
    CommentEvent,
    CommentIntent,
    Priority,
)

log = logging.getLogger(__name__)

# Risk-sensitive phrasings route to the constrained path. Detected in code
# first so a classifier failure fails SAFE rather than open.
RISK_RE = re.compile(
    r"\b(should i|shall i|do i|worth) (buy|sell|enter|go long|go short|hold)\b"
    r"|\b(buy|sell) now\b"
    r"|\bis it going to (hit|reach|go)\b"
    r"|\bwill it (hit|reach|go|drop|rise)\b"
    r"|\bgive me (a )?(signal|entry|tp|sl)\b"
    r"|\bwhat.s your (entry|target|tp|sl)\b",
    re.I,
)

SPAM_RE = re.compile(
    r"(free signals|dm me|100% win|guaranteed profit|t\.me/|whatsapp|join my)",
    re.I,
)

GREETING_RE = re.compile(r"^\s*(gm|good morning|hi|hello|hey|yo|sup)\b.{0,20}$", re.I)


@dataclass(slots=True)
class ScoredComment:
    comment: CommentEvent
    priority: Priority
    #: Higher wins when several comments are pending.
    weight: float


class Deduper:
    """Sliding-window dedupe on platform_msg_id.

    For screen capture the id is sha256(author+text), so the same comment seen
    across many frames collapses to one, and a genuine repeat by the same user
    also collapses -- acceptable, arguably desirable.
    """

    def __init__(self, window: int = 512) -> None:
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._window = window

    def is_duplicate(self, msg_id: str) -> bool:
        if msg_id in self._seen:
            self._seen.move_to_end(msg_id)
            return True
        self._seen[msg_id] = None
        if len(self._seen) > self._window:
            self._seen.popitem(last=False)
        return False


def heuristic_classify(c: CommentEvent) -> CommentClassification:
    """Offline classifier. Also the fallback when the API call fails.

    Fails safe: anything matching RISK_RE is risk-sensitive regardless of what
    a model would have said.
    """
    t = c.text_norm
    risky = bool(RISK_RE.search(t))

    if SPAM_RE.search(t):
        return CommentClassification(intent=CommentIntent.SPAM, relevance=0.0)
    if GREETING_RE.match(t):
        return CommentClassification(intent=CommentIntent.GREETING, relevance=0.15)
    if risky:
        return CommentClassification(
            intent=CommentIntent.TRADE_ADVICE_REQ, is_risk_sensitive=True, relevance=0.8
        )
    if re.search(r"\b(bot|fake|scam|lol|trash)\b", t):
        return CommentClassification(intent=CommentIntent.PROVOCATION, relevance=0.1)
    if re.search(r"\b(what is|what's|whats|how do|how does|explain|difference between)\b", t):
        return CommentClassification(intent=CommentIntent.EDUCATION_Q, relevance=0.85)
    if re.search(r"\b(resistance|support|level|structure|trend|liquidity|bos|choch)\b", t):
        return CommentClassification(intent=CommentIntent.TECHNICAL_Q, relevance=0.9)
    if re.search(r"\b(gold|xau|dollar|dxy|price|fed|cpi)\b", t):
        return CommentClassification(intent=CommentIntent.MARKET_Q, relevance=0.75)
    if "?" in c.text_raw:
        return CommentClassification(intent=CommentIntent.MARKET_Q, relevance=0.5)
    return CommentClassification(intent=CommentIntent.OFF_TOPIC, relevance=0.2)


class CommentPipeline:
    """One per session."""

    PRIORITY_BY_INTENT = {
        CommentIntent.TECHNICAL_Q: Priority.HIGH,
        CommentIntent.MARKET_Q: Priority.HIGH,
        CommentIntent.EDUCATION_Q: Priority.MEDIUM,
        CommentIntent.TRADE_ADVICE_REQ: Priority.MEDIUM,
        CommentIntent.GREETING: Priority.LOW,
        CommentIntent.JOKE: Priority.LOW,
        CommentIntent.OFF_TOPIC: Priority.LOW,
    }

    DROP = {CommentIntent.SPAM, CommentIntent.PROVOCATION}

    def __init__(self, session_id: str, classifier=None, min_ocr_confidence: float = 0.55) -> None:
        self.session_id = session_id
        self.dedupe = Deduper()
        self.classifier = classifier
        self.min_ocr_confidence = min_ocr_confidence

    async def process(self, c: CommentEvent) -> ScoredComment | None:
        # Isolation layer 4: assert before anything else touches it.
        if c.session_id != self.session_id:
            raise ValueError(
                f"comment for {c.session_id} reached pipeline for {self.session_id}"
            )

        if self.dedupe.is_duplicate(c.platform_msg_id):
            return None

        cls = heuristic_classify(c)
        if self.classifier is not None:
            try:
                cls = await self.classifier(c) or cls
                # Never let a model downgrade a code-detected risk flag.
                if RISK_RE.search(c.text_norm) and not cls.is_risk_sensitive:
                    cls = cls.model_copy(update={"is_risk_sensitive": True})
            except Exception as exc:  # noqa: BLE001 - classifier must never break ingest
                log.warning("classifier failed, using heuristic: %s", exc)

        if cls.source_confidence < self.min_ocr_confidence:
            return None  # garbled OCR must never reach the generator
        if cls.intent in self.DROP:
            return None

        scored = CommentEvent(
            **{**c.model_dump(exclude={"classification"}), "classification": cls}
        )
        priority = self.PRIORITY_BY_INTENT.get(cls.intent, Priority.LOW)
        weight = cls.relevance * (1.25 if cls.intent is CommentIntent.TECHNICAL_Q else 1.0)
        return ScoredComment(comment=scored, priority=priority, weight=weight)


def build_classifier(model: str | None = None):
    """Haiku-backed classifier. Returns None when no API key is configured."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None

    import anthropic

    client = anthropic.AsyncAnthropic()
    model = model or os.environ.get("GOLD_MODEL_CLASSIFY", "claude-haiku-4-5")

    async def classify(c: CommentEvent) -> CommentClassification:
        msg = await client.messages.parse(
            model=model,
            max_tokens=256,
            system=(
                "Classify a single live-stream comment on a Gold/XAUUSD trading "
                "stream. is_risk_sensitive must be true for any request for "
                "personalised trading advice, entries, targets, or a prediction "
                "of where price will go. Be strict about that field."
            ),
            messages=[{"role": "user", "content": c.text_raw}],
            output_config={"format": CommentClassification},
        )
        return msg.parsed_output

    return classify
