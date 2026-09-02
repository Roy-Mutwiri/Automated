"""Comment pipeline: normalise -> classify -> dedupe -> prioritise.

Not every comment reaches the generator. Most should not. This stage is
deliberately cheap: the heuristic decides most cases in code, and the local
model is consulted only where it is genuinely unsure.
"""

from __future__ import annotations

import json
import logging
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
            except Exception as exc:
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


CLASSIFIER_SYSTEM = """Classify one comment from a live Gold (XAUUSD) trading stream.

Reply with JSON only, no other text:
{"intent": "...", "is_risk_sensitive": true|false, "relevance": 0.0-1.0}

intent is one of: market_q, technical_q, education_q, greeting, joke, spam,
provocation, off_topic, trade_advice_req

is_risk_sensitive is true for ANY request for personalised trading advice, an
entry or exit, a target, a signal, or a prediction of where price will go. Be
strict: a false negative here means the host gives financial advice.

relevance is how much a Gold trading audience would benefit from this being
answered on air."""

# Which heuristic verdicts are worth a second opinion. Spam, greetings and
# clear technical questions are already decided; asking a model to re-confirm
# them is latency and GPU time spent on nothing.
AMBIGUOUS_INTENTS = {
    CommentIntent.MARKET_Q,
    CommentIntent.OFF_TOPIC,
    CommentIntent.JOKE,
}


def build_classifier(llm, escalate_only: bool = True):
    """Classification backed by the locally hosted model.

    Runs on the same server as generation. Two things keep the cost sane:

    `escalate_only` -- the heuristic decides first, and the model is consulted
    only where the heuristic is genuinely unsure. On a busy stream across seven
    sessions, classifying every comment with a model is a large amount of GPU
    time spent re-confirming that "gm" is a greeting.

    Fail-safe -- a model verdict can UPGRADE a comment to risk-sensitive but
    never downgrade one the regex already flagged. A classifier failure must
    not be the reason the host starts giving trading advice.
    """

    async def classify(c: CommentEvent) -> CommentClassification | None:
        baseline = heuristic_classify(c)
        if escalate_only and baseline.intent not in AMBIGUOUS_INTENTS:
            return baseline

        from platform_.llm.base import ChatMessage

        result = await llm.complete(
            [
                ChatMessage(role="system", content=CLASSIFIER_SYSTEM),
                ChatMessage(role="user", content=c.text_raw[:400]),
            ],
            max_tokens=120,
            temperature=0.0,  # a classifier that varies run to run is not one
        )

        match = re.search(r"\{[^{}]*\}", result.text)
        if not match:
            log.debug("classifier returned no JSON; keeping heuristic verdict")
            return baseline
        try:
            data = json.loads(match.group(0))
            intent = CommentIntent(str(data.get("intent", baseline.intent.value)))
        except (json.JSONDecodeError, ValueError):
            return baseline

        return CommentClassification(
            intent=intent,
            # Either source may raise the flag; neither may lower it.
            is_risk_sensitive=bool(data.get("is_risk_sensitive"))
            or baseline.is_risk_sensitive,
            relevance=max(0.0, min(1.0, float(data.get("relevance", baseline.relevance)))),
            source_confidence=baseline.source_confidence,
        )

    return classify
