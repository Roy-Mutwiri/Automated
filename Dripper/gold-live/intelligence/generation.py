"""Response generation.

Two implementations behind one interface:

  ClaudeGenerator  -- claude-opus-5, streaming, sentence-chunked
  OfflineGenerator -- templates, no network. Lets the whole pipeline (Director,
                      memory, safety, audio) be exercised with no API key, which
                      matters more than it sounds: it makes the dry run runnable
                      by anyone, instantly, and keeps tests free and fast.

Latency notes, because this is the critical path:
  - We STREAM and split on sentence boundaries as tokens arrive, handing
    segment 1 to TTS while the model is still writing segment 3. This is worth
    roughly two seconds and is the reason the budget closes at all.
  - effort is "low". These are three-sentence spoken utterances; deep reasoning
    buys nothing and costs time-to-first-token. Thinking stays ON (disabling it
    on Opus 5 risks tool-call and tag leakage) -- low effort is the right lever.
  - The persona system prompt is the CACHED PREFIX. It must be byte-stable.
    Volatile market state is appended as a mid-conversation system message
    instead, which Opus 5 supports and which does not invalidate the prefix.
    Editing the top-level system field on every price tick would destroy the
    cache on literally every request.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from intelligence.director import SpeechIntent
from intelligence.personas import Persona
from intelligence.safety import price_policy_note
from platform_.llm.base import ChatMessage, LLMBackend
from shared.contracts import CommentIntent, MarketState, Provenance, TriggerType

log = logging.getLogger(__name__)

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


@dataclass
class GenerationResult:
    text: str
    segments: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)


class Generator(ABC):
    @abstractmethod
    async def generate(
        self,
        persona: Persona,
        intent: SpeechIntent,
        market: MarketState,
        transcript: list[str],
    ) -> GenerationResult: ...


# ---------------------------------------------------------------------------
# Prompt construction (shared)
# ---------------------------------------------------------------------------


def build_turn_context(intent: SpeechIntent, market: MarketState, transcript: list[str]) -> str:
    """The volatile half. Goes AFTER the cache breakpoint, never in the prefix."""
    tf = market.timeframes.get("5m")
    lines = [
        # Reference data, explicitly framed as reference. Production showed the
        # model narrating whatever concrete number sat at the top of this block
        # -- 50% of utterances opened by reciting bid/ask or the spread, and it
        # would discuss the spread no matter which topic it had been asked for.
        # A small model treats the most specific thing it can see as the
        # subject, so this block has to say plainly that it is not the subject.
        "MARKET REFERENCE (background for you -- NOT the topic, do not narrate it)",
        f"  data confidence : {market.confidence.value} ({market.staleness_ms}ms old)",
        f"  trading session : {market.session.value}",
    ]
    if market.may_quote_price():
        lines.append(f"  price           : bid {market.price.bid} / ask {market.price.ask}")
    if tf:
        lines += [
            f"  5m trend        : {tf.trend.value}",
            f"  5m structure    : {tf.structure.value}",
            f"  session range   : {tf.swing_low} - {tf.swing_high}",
            f"  atr(5m)         : {tf.atr}",
        ]
    if market.detections:
        lines.append("  detected        : " + "; ".join(d.label for d in market.detections))
    if market.context:
        lines.append("  context         : " + "; ".join(
            f"{c.label}{f' ({c.detail})' if c.detail else ''}" for c in market.context
        ))

    lines.append("")
    lines.append(price_policy_note(market))
    lines.append("")

    if transcript:
        lines.append("WHAT YOU JUST SAID (do not repeat these points or their phrasing)")
        lines += [f"  - {t}" for t in transcript]
        lines.append("")

        # Listing whole previous utterances tells the model what content to
        # avoid but nothing about shape, so it varied the content and reused
        # the same construction -- "the bid is now", "the price is now", "the
        # spread has widened" over and over. Naming the openings directly is
        # what stops that.
        openings = [" ".join(t.split()[:5]) for t in transcript[-5:] if t.split()]
        if openings:
            lines.append("SENTENCE OPENINGS YOU HAVE ALREADY USED -- start differently")
            lines += [f"  - {o}..." for o in openings]
            lines.append("")

    lines.append("WHAT TO TALK ABOUT NOW  <-- this is your subject")
    if intent.trigger is TriggerType.COMMENT:
        payload = intent.payload
        lines.append(f"  A viewer asked: \"{payload.get('text', '')}\"")
        if payload.get("is_risk_sensitive"):
            lines.append(
                "  This is a request for personalised trading advice or a price "
                "prediction. Do NOT give either. Redirect to process: how someone "
                "would frame the decision, what would invalidate the idea, how risk "
                "is sized. Be warm about it, not preachy, and do not lecture."
            )
        lines.append("  Answer it naturally, as part of the conversation. Do not "
                     "announce that a viewer asked something.")
    elif intent.trigger is TriggerType.MARKET_EVENT:
        lines.append(f"  Market event: {intent.payload.get('hint', intent.topic)}")
        lines.append("  React to it the way a host watching the chart would.")
    elif intent.trigger is TriggerType.SILENCE:
        lines.append("  It has been quiet. Move the conversation somewhere useful "
                     "rather than restating the current price action.")
    elif intent.trigger is TriggerType.EDUCATION:
        instruction = intent.payload.get("instruction", intent.topic)
        lines.append(f"  {instruction}")
        if rationale := intent.payload.get("rationale"):
            lines.append(f"  (worth raising now because: {rationale})")
        if market.confidence.value != "live":
            lines.append(
                "  The market is closed or the feed is not live, so this is the "
                "main content right now. Give it proper room rather than "
                "treating it as filler between price updates."
            )
        lines.append("  Keep it concrete. One idea, well explained, beats three "
                     "mentioned in passing.")
    else:
        lines.append(f"  {intent.topic}")


    # Last, so it is the freshest instruction in the context. A viewer question
    # is exempt: answering "where is gold right now" by refusing to say a
    # number would be absurd.
    if intent.trigger is not TriggerType.COMMENT:
        lines.append("")
        lines.append(
            "HOW TO OPEN: not with the bid, the ask, the spread or the current "
            "price. Open on the subject above. Bring in a number only where it "
            "makes that point sharper -- as evidence, never as the headline."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class LocalGenerator(Generator):
    """The default. Runs against the locally hosted model on the central machine.

    Streams and yields sentence segments as they complete, so TTS can start on
    segment 1 while the model is still writing segment 3.

    Temperature is deliberately high and the backend applies presence/frequency
    penalties: the failure mode for a 24/7 host is not incoherence, it is
    sounding like the same three sentences rearranged. Determinism is the enemy
    here.
    """

    def __init__(self, llm: LLMBackend, temperature: float = 0.9) -> None:
        self.llm = llm
        self.temperature = temperature

    async def generate(
        self,
        persona: Persona,
        intent: SpeechIntent,
        market: MarketState,
        transcript: list[str],
    ) -> GenerationResult:
        t0 = time.perf_counter()
        first_token_ms: int | None = None
        chunks: list[str] = []

        messages = [
            # Byte-stable for the life of the session -- this is what the
            # server's prefix cache keys on. Nothing volatile may go here.
            ChatMessage(role="system", content=persona.system_prompt()),
            ChatMessage(
                role="user", content=build_turn_context(intent, market, transcript)
            ),
        ]

        async for delta in self.llm.stream(
            messages, max_tokens=260, temperature=self.temperature
        ):
            if first_token_ms is None:
                first_token_ms = int((time.perf_counter() - t0) * 1000)
            chunks.append(delta)

        text = _clean(("".join(chunks)).strip())
        return GenerationResult(
            text=text,
            segments=split_sentences(text),
            provenance=Provenance(
                market_state_id=market.state_id,
                market_confidence=market.confidence,
                model=self.llm.name,
                first_token_ms=first_token_ms,
                generation_ms=int((time.perf_counter() - t0) * 1000),
            ),
        )


# Smaller local models leak stage directions and formatting that a spoken
# stream must never carry. Cheap to strip, and far more reliable than asking
# the model not to do it.
_STRIP_PATTERNS = [
    re.compile(r"^\s*(host|ai|assistant|speaker)\s*:\s*", re.I),
    re.compile(r"\*[^*]{0,80}\*"),          # *clears throat*
    re.compile(r"\([^)]{0,60}(pause|laugh|beat|smiles)[^)]{0,20}\)", re.I),
    re.compile(r"^#{1,6}\s+", re.M),        # markdown headings
    re.compile(r"^[-*]\s+", re.M),          # bullet points
    re.compile(r"\[[^\]]{0,40}\]"),         # [inaudible]
]


def _clean(text: str) -> str:
    for pattern in _STRIP_PATTERNS:
        text = pattern.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


class ClaudeGenerator(Generator):
    def __init__(self, model: str | None = None, effort: str = "low") -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic()
        self.model = model or os.environ.get("GOLD_MODEL_GENERATION", "claude-opus-5")
        self.effort = effort

    async def generate(
        self,
        persona: Persona,
        intent: SpeechIntent,
        market: MarketState,
        transcript: list[str],
    ) -> GenerationResult:
        t0 = time.perf_counter()
        first_token_ms: int | None = None
        chunks: list[str] = []

        system: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": persona.system_prompt(),
                # The cache breakpoint. Everything before this is stable for the
                # life of the session; everything volatile goes in messages.
                "cache_control": {"type": "ephemeral"},
            }
        ]

        async with self.client.messages.stream(
            model=self.model,
            max_tokens=400,
            system=system,
            output_config={"effort": self.effort},
            messages=[
                {"role": "user", "content": build_turn_context(intent, market, transcript)}
            ],
        ) as stream:
            async for text in stream.text_stream:
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - t0) * 1000)
                chunks.append(text)
            final = await stream.get_final_message()

        text = "".join(chunks).strip()
        usage = getattr(final, "usage", None)
        return GenerationResult(
            text=text,
            segments=split_sentences(text),
            provenance=Provenance(
                market_state_id=market.state_id,
                market_confidence=market.confidence,
                model=self.model,
                effort=self.effort,
                prompt_tokens=getattr(usage, "input_tokens", None),
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", None),
                first_token_ms=first_token_ms,
                generation_ms=int((time.perf_counter() - t0) * 1000),
            ),
        )


# ---------------------------------------------------------------------------
# Offline
# ---------------------------------------------------------------------------


class OfflineGenerator(Generator):
    """Template-based. Exercises every other component with no API key.

    The phrasing pools are intentionally varied so the repetition detector has
    something real to work against -- a single fixed template would make the
    dry run useless as a test of it.
    """

    OPENERS = [
        "Something worth watching here.",
        "Okay, look at what just happened.",
        "This is the part that matters.",
        "Quick note on structure.",
        "Here's what I'm seeing.",
    ]

    def __init__(self, seed: int = 11) -> None:
        self.rng = random.Random(seed)

    async def generate(
        self,
        persona: Persona,
        intent: SpeechIntent,
        market: MarketState,
        transcript: list[str],
    ) -> GenerationResult:
        t0 = time.perf_counter()
        tf = market.timeframes.get("5m")
        can_price = market.may_quote_price()

        if intent.trigger is TriggerType.COMMENT:
            q = intent.payload.get("text", "")
            if intent.payload.get("is_risk_sensitive"):
                body = (
                    "I won't tell anyone when to get in or out. What I can do is "
                    "frame it: decide what would have to happen for the idea to be "
                    "wrong, and size it so that being wrong costs you very little."
                )
            else:
                level = (
                    f"around {tf.swing_high}" if (can_price and tf and tf.swing_high) else
                    "at the prior session high"
                )
                body = (
                    f"On \"{q.rstrip('?')}\" - the level I keep coming back to is {level}. "
                    "The interesting question is whether price accepts above it or gets "
                    "rejected from it."
                )
        elif intent.trigger is TriggerType.MARKET_EVENT:
            hint = intent.payload.get("hint", "structure just shifted")
            body = (
                f"{hint} If that holds, the scenario I'm watching is continuation; "
                "if we close back inside the range, that idea is invalidated."
            )
        elif intent.trigger is TriggerType.EDUCATION:
            # Both halves vary: the angle instruction (8 variants) leads, the
            # item seed (46 variants) follows. Templates can never reach the
            # variety a real model produces -- offline repetition numbers are
            # pessimistic by construction and are a plumbing signal, not a
            # content-quality signal. Run the soak with --live to judge content.
            seed = intent.payload.get("seed", "")
            angle_text = intent.payload.get("angle_instruction", "")
            body = f"{seed} {angle_text}".strip() or intent.topic
        elif intent.trigger is TriggerType.SILENCE:
            body = (
                "While it's quiet - risk per trade is the thing beginners get wrong "
                "most often. Position size is a decision you make before you're "
                "emotionally involved."
            )
        else:
            body = (
                f"On {intent.topic}: it's less about the label and more about what "
                "it tells you about who is in control here."
            )

        if not can_price:
            body += " I'm waiting on a clean data feed before I quote any exact levels."

        text = f"{self.rng.choice(self.OPENERS)} {body}"
        return GenerationResult(
            text=text,
            segments=split_sentences(text),
            provenance=Provenance(
                market_state_id=market.state_id,
                market_confidence=market.confidence,
                model="offline-template",
                first_token_ms=1,
                generation_ms=int((time.perf_counter() - t0) * 1000),
            ),
        )


async def build_generator(mode: str = "auto") -> tuple[Generator, LLMBackend | None]:
    """Resolve a generator.

    'local'   the hosted model on the central machine (the design default)
    'api'     hosted API, for A/B comparison against local output
    'offline' templates, no model at all -- structure tests only
    'auto'    local if its server answers, else offline

    Returns the backend alongside so callers can share it with the proposer and
    classifier rather than opening a second connection pool.
    """
    if mode == "offline":
        return OfflineGenerator(), None

    if mode in ("auto", "local"):
        # Probe the well-known ports rather than requiring the recipient of a
        # shared build to know that vLLM is 8000 and Ollama is 11434.
        from platform_.llm.discovery import discover, endpoint_hints

        llm = await discover()
        if llm is not None:
            log.info("using local model at %s (%s)", llm.base_url, llm.model)
            return LocalGenerator(llm), llm
        if mode == "local":
            raise RuntimeError(endpoint_hints())
        log.warning("no local model server; falling back to offline templates")
        return OfflineGenerator(), None

    if mode == "api":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("mode=api needs ANTHROPIC_API_KEY")
        return ClaudeGenerator(), None

    raise ValueError(f"unknown generator mode: {mode}")


__all__ = [
    "ClaudeGenerator",
    "CommentIntent",
    "GenerationResult",
    "Generator",
    "OfflineGenerator",
    "build_generator",
    "build_turn_context",
    "split_sentences",
]
