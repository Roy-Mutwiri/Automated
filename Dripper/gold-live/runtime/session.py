"""The session runtime: one process, one session, no shared mutable state.

Wires together comment pipeline -> Director -> generator -> safety -> audio.
Everything it touches is scoped to self.session_id.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from intelligence import safety
from intelligence.comments import CommentPipeline, ScoredComment
from intelligence.director import Decision, Director, SpeechIntent
from intelligence.generation import Generator
from intelligence.memory import SessionMemory
from intelligence.personas import Persona
from platform_.tts.base import TTSProvider
from shared.contracts import (
    AIResponse,
    AudioRequest,
    CommentEvent,
    MarketEvent,
    MarketState,
    Priority,
    SessionState,
    TriggerRef,
    TriggerType,
    utcnow,
)

log = logging.getLogger(__name__)

SEVERITY_TO_PRIORITY = {
    5: Priority.CRITICAL,
    4: Priority.HIGH,
    3: Priority.HIGH,
    2: Priority.MEDIUM,
    1: Priority.LOW,
}


class SessionRuntime:
    def __init__(
        self,
        state: SessionState,
        persona: Persona,
        generator: Generator,
        tts: TTSProvider,
        out_dir: Path,
    ) -> None:
        self.state = state
        self.persona = persona
        self.generator = generator
        self.tts = tts
        self.out_dir = out_dir / state.session_id

        self.memory = SessionMemory(state.session_id)
        self.director = Director(state.session_id, self.memory)
        self.pipeline = CommentPipeline(state.session_id)

        self.transcript: list[AIResponse] = []
        self.dropped_unsafe: list[tuple[str, list[str]]] = []
        self.dropped_repetitive: list[tuple[str, float]] = []

    # -- ingest -----------------------------------------------------------

    async def on_comment(self, c: CommentEvent, now: datetime | None = None) -> None:
        scored: ScoredComment | None = await self.pipeline.process(c)
        if scored is None:
            return
        cls = scored.comment.classification
        assert cls is not None
        self.memory.record_question(scored.comment.text_norm)
        self.director.offer(
            SpeechIntent(
                trigger=TriggerType.COMMENT,
                priority=scored.priority,
                topic=f"q:{cls.intent.value}",
                seed_text=scored.comment.text_norm,
                ttl_s=120.0,
                created_at=now or utcnow(),
                payload={
                    "text": scored.comment.text_raw,
                    "intent": cls.intent.value,
                    "is_risk_sensitive": cls.is_risk_sensitive,
                    "msg_id": scored.comment.platform_msg_id,
                },
            )
        )

    def on_market_event(self, ev: MarketEvent, now: datetime | None = None) -> None:
        self.director.offer(
            SpeechIntent(
                trigger=TriggerType.MARKET_EVENT,
                priority=SEVERITY_TO_PRIORITY.get(ev.severity, Priority.MEDIUM),
                topic=f"mkt:{ev.kind.value}",
                seed_text=ev.narrative_hint or ev.kind.value,
                ttl_s=60.0,
                created_at=now or utcnow(),
                source_event_id=ev.event_id,
                payload={"hint": ev.narrative_hint, "kind": ev.kind.value},
            )
        )

    def offer_filler(self, topic: str, now: datetime | None = None) -> None:
        """Something to fall back to during quiet periods -- education, not
        invented market movement."""
        self.director.offer(
            SpeechIntent(
                trigger=TriggerType.SILENCE,
                priority=Priority.LOW,
                topic=f"edu:{topic}",
                seed_text=topic,
                ttl_s=300.0,
                created_at=now or utcnow(),
                payload={"topic": topic},
            )
        )

    # -- the speak cycle --------------------------------------------------

    async def tick(self, market: MarketState, now: datetime | None = None) -> AIResponse | None:
        now = now or utcnow()
        decision: Decision | None = self.director.tick(now)
        if decision is None:
            return None

        trace_id = uuid4().hex[:12]
        result = await self._generate_non_repetitive(decision, market)
        if result is None:
            return None

        # Safety gate: after generation, before audio. Never spoken if it fails.
        verdict = safety.check(result.text, market)
        if not verdict.report.passed:
            log.warning(
                "[%s] utterance blocked: %s",
                self.state.session_id,
                "; ".join(verdict.report.violations),
            )
            self.dropped_unsafe.append((result.text, verdict.report.violations))
            return None

        response = AIResponse(
            session_id=self.state.session_id,
            trace_id=trace_id,
            text=result.text,
            segments=result.segments,
            trigger=TriggerRef(
                type=decision.intent.trigger,
                source_event_id=decision.intent.source_event_id,
                priority=decision.intent.priority,
            ),
            provenance=result.provenance,
            safety=verdict.report,
            interruptible=decision.intent.priority < Priority.CRITICAL,
        )

        audio_ms = await self._speak(response, decision)
        self.director.mark_speaking(decision.intent, audio_ms, now)
        self.memory.record_utterance(response.text, decision.intent.topic, now)
        self.state.last_spoke_at = now
        self.state.utterances_1h += 1
        self.transcript.append(response)
        return response

    async def _generate_non_repetitive(self, decision: Decision, market: MarketState):
        """Generate, then gate on repetition of the GENERATED text.

        The Director's pre-generation check scores the trigger (a comment, an
        event hint). That catches "we already covered this event", but not two
        different triggers producing near-identical wording -- which is exactly
        what happens when several viewer questions get the same stock framing.
        Only the output reveals that.

        Policy is priority-dependent, because dropping is not always the right
        answer. A severity-5 break of structure must never go unsaid just
        because the phrasing rhymes with something from ten minutes ago:

            CRITICAL  regenerate once, then speak regardless
            HIGH      regenerate once, drop if still repetitive
            below     drop immediately -- filler is not worth a second call
        """
        priority = decision.intent.priority
        attempts = 2 if priority >= Priority.HIGH else 1

        last = None
        for attempt in range(attempts):
            result = await self.generator.generate(
                persona=self.persona,
                intent=decision.intent,
                market=market,
                transcript=self.memory.recent_transcript(),
            )
            last = result
            repetitive, similarity = self.memory.is_repetitive(result.text)
            if not repetitive:
                return result
            log.info(
                "[%s] repetitive (sim=%.2f, attempt %d/%d, %s)",
                self.state.session_id, similarity, attempt + 1, attempts, priority.name,
            )

        assert last is not None
        _, similarity = self.memory.is_repetitive(last.text)
        if priority is Priority.CRITICAL:
            log.warning(
                "[%s] speaking a repetitive utterance anyway - CRITICAL event "
                "must not be silently dropped (sim=%.2f)",
                self.state.session_id, similarity,
            )
            return last

        self.dropped_repetitive.append((last.text, similarity))
        return None

    async def _speak(self, response: AIResponse, decision: Decision) -> int:
        req = AudioRequest(
            utterance_id=response.utterance_id,
            session_id=response.session_id,
            trace_id=response.trace_id,
            segments=response.segments,
            voice_id=self.persona.voice_id,
            priority=decision.intent.priority,
            preempts=decision.preempts,
        )
        total = 0
        for i, segment in enumerate(req.segments):
            path = self.out_dir / f"{len(self.transcript):03d}_{req.utterance_id.hex[:8]}_{i}.wav"
            res = await self.tts.synthesize(segment, req.voice_id, path)
            total += res.duration_ms
        return total
