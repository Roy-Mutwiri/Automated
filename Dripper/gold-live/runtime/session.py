"""The session runtime: one process, one session, no shared mutable state.

Wires together comment pipeline -> Director -> generator -> safety -> audio.
Everything it touches is scoped to self.session_id.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from intelligence import safety
from intelligence.comments import CommentPipeline, ScoredComment
from intelligence.content import ANGLE_INSTRUCTION, Beat, ContentPlanner, MarketPhase
from intelligence.director import Decision, Director, SpeechIntent
from intelligence.generation import Generator
from intelligence.memory import SessionMemory
from intelligence.personas import Persona
from intelligence.proposer import TopicCandidate, TopicProposer
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

#: How much recent history to keep in memory for diagnostics. Enough to inspect
#: what just happened, small enough that a session running for weeks does not
#: grow without bound. The monotonic counters beside these carry the totals.
RETAINED_UTTERANCES = 500
RETAINED_DROPS = 200

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
        planner: ContentPlanner | None = None,
        proposer: TopicProposer | None = None,
        max_silence_s: float | None = None,
        router=None,
    ) -> None:
        self.state = state
        self.persona = persona
        self.generator = generator
        self.tts = tts
        self.out_dir = out_dir / state.session_id
        self.planner = planner
        self.proposer = proposer
        #: When set, the router owns synthesis; this runtime must not also
        #: synthesise or every utterance is rendered twice.
        self.router = router
        self.coverage = proposer.coverage if proposer else None
        self._candidates: deque[TopicCandidate] = deque()
        self.content_exhausted_at: datetime | None = None
        self.fallback_used = 0

        self.memory = SessionMemory(state.session_id)
        self.director = Director(
            state.session_id, self.memory, max_silence_s=max_silence_s
        )
        self.max_silence_s = max_silence_s

        if max_silence_s is not None and planner is not None:
            # The planner paces itself to make an inventory last, which is right
            # when the host speaks every few minutes and wrong when it must
            # speak every few seconds. Continuous mode needs a beat ready
            # whenever the queue empties, so the offer interval tracks the
            # silence budget rather than a fixed 40s.
            planner.config.min_offer_interval = timedelta(
                seconds=max(2.0, max_silence_s / 3)
            )
        self.pipeline = CommentPipeline(state.session_id)

        # Bounded, with monotonic counters alongside.
        #
        # These were plain lists that only ever grew. self.transcript retained
        # every AIResponse for the life of the session purely so len() could
        # number a wav file, and dropped_repetitive grew about four times
        # faster still, because roughly three quarters of everything generated
        # is rejected. On a 24/7 host aimed at an 8 GB machine that is a leak
        # with no upper bound. export_state() never used any of it -- it reads
        # memory.short_term, which was already bounded.
        #
        # The counters are what callers actually wanted: a total that keeps
        # rising. The deques keep a recent window for diagnostics.
        self.transcript: deque[AIResponse] = deque(maxlen=RETAINED_UTTERANCES)
        self.dropped_unsafe: deque[tuple[str, list[str]]] = deque(maxlen=RETAINED_DROPS)
        self.dropped_repetitive: deque[tuple[str, float]] = deque(maxlen=RETAINED_DROPS)
        self.spoken_count = 0
        self.unsafe_count = 0
        self.repetitive_count = 0

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
        """Raw topic string. Prefer offer_planned_content()."""
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

    async def offer_next_topic(
        self,
        phase: MarketPhase,
        market: MarketState,
        now: datetime | None = None,
    ) -> TopicCandidate | Beat | None:
        """Decide what to talk about next, generated rather than scripted.

        Proposals are fetched in batches ahead of time and queued, so a slow
        proposal never delays speech. If the model is unreachable the scripted
        inventory takes over -- a degraded stream reading from a list beats a
        silent one, but `fallback_used` should be near zero in normal operation
        and belongs on the dashboard.
        """
        now = now or utcnow()
        if self.director.has_pending(TriggerType.EDUCATION):
            return None

        if self.proposer is not None:
            if not self._candidates:
                self._candidates.extend(
                    await self.proposer.propose(
                        persona=self.persona,
                        market=market,
                        phase=phase.value,
                        audience_questions=self.memory.hot_questions(min_count=1),
                        now=now,
                    )
                )
            if self._candidates:
                candidate = self._candidates.popleft()
                self._offer_candidate(candidate, phase, now)
                return candidate

        # Model unreachable or proposing nothing usable.
        self.fallback_used += 1
        return self.offer_planned_content(phase, now)

    def _offer_candidate(
        self, c: TopicCandidate, phase: MarketPhase, now: datetime
    ) -> None:
        priority = Priority.LOW if phase is MarketPhase.ACTIVE else Priority.MEDIUM
        self.director.offer(
            SpeechIntent(
                trigger=TriggerType.EDUCATION,
                priority=priority,
                topic=c.topic,
                seed_text=c.brief,
                ttl_s=600.0,
                created_at=now,
                payload={
                    "topic": c.topic_key,
                    "instruction": c.brief,
                    "rationale": c.rationale,
                    "category": c.category,
                    "generated": True,
                },
            )
        )
        if self.coverage is not None:
            self.coverage.record(c.topic_key, c.brief, now)

    def offer_planned_content(
        self, phase: MarketPhase, now: datetime | None = None
    ) -> Beat | None:
        """Pull the next beat from the content plan.

        This is what fills the market's closed hours and flat sessions without
        inventing price action. Returns None when the inventory is genuinely
        exhausted for this phase -- which is a signal worth alerting on, not a
        condition to paper over.
        """
        if self.planner is None:
            return None
        now = now or utcnow()

        # At most one unspoken planned beat at a time. Without this the planner
        # is asked for a beat on every tick and marks it used immediately, so
        # the whole inventory is spent far faster than it can be spoken -- 368
        # beats gone in 21 minutes, followed by silence.
        if self.director.has_pending(TriggerType.EDUCATION):
            return None

        beat = self.planner.next_beat(phase, now)
        if beat is None:
            # Distinguish "not yet" from "nothing left" -- only the latter is
            # a problem worth alerting on.
            if self.planner.is_exhausted(phase, now):
                self.content_exhausted_at = self.content_exhausted_at or now
            return None

        # When the market is closed or flat, planned content IS the show -- it
        # is not filler between price updates, and it must not sit at LOW where
        # it only clears the score floor after minutes of dead air.
        priority = (
            Priority.LOW if phase is MarketPhase.ACTIVE else Priority.MEDIUM
        )
        self.director.offer(
            SpeechIntent(
                trigger=TriggerType.EDUCATION,
                priority=priority,
                topic=beat.topic,
                seed_text=beat.instruction(),
                ttl_s=600.0,
                created_at=now,
                payload={
                    "topic": beat.item.title,
                    "instruction": beat.instruction(),
                    "seed": beat.item.seed,
                    "angle_instruction": ANGLE_INSTRUCTION[beat.angle],
                    "angle": beat.angle.value,
                    "category": beat.item.category,
                    "beat_key": beat.key,
                },
            )
        )
        self.planner.mark_used(beat, now)
        return beat

    # -- persistence ------------------------------------------------------

    def export_state(self) -> dict:
        """Everything that must survive a restart.

        A 24/7 system restarts -- on crash, on upgrade, on a machine reboot.
        Without this the host wakes up having forgotten every topic it covered
        and immediately repeats an hour of material, which to a regular viewer
        looks exactly like a broken bot.

        Deliberately excludes the utterance similarity index: it guards phrasing
        over a short window, and rebuilding it from the recent transcript on
        restart is both cheaper and more correct than serialising it.
        """
        state: dict = {
            "topics_last_seen": {
                k: v.isoformat() for k, v in self.memory.topics_last_seen.items()
            },
            "audience_questions": dict(self.memory.audience_questions),
            "utterance_count": self.memory.utterance_count,
            "recent_transcript": [
                {"text": u.text, "topic": u.topic, "at": u.at.isoformat()}
                for u in self.memory.short_term
            ],
        }
        if self.coverage is not None:
            state["coverage"] = self.coverage.export_state()
        if self.planner is not None:
            state["planner"] = self.planner.export_state()
        return state

    def load_state(self, state: dict) -> None:
        from collections import Counter
        from datetime import datetime as _dt

        try:
            self.memory.topics_last_seen = {
                k: _dt.fromisoformat(v)
                for k, v in state.get("topics_last_seen", {}).items()
            }
            self.memory.audience_questions = Counter(state.get("audience_questions", {}))

            # Rehydrating the transcript also rebuilds the similarity index, so
            # the host does not repeat its last few sentences verbatim.
            for entry in state.get("recent_transcript", []):
                self.memory.record_utterance(
                    entry["text"], entry["topic"], _dt.fromisoformat(entry["at"])
                )

            # Set the counter AFTER replaying the transcript. record_utterance
            # increments it, so assigning first double-counts every restored
            # utterance and the number drifts upward on every restart.
            self.memory.utterance_count = int(state.get("utterance_count", 0))

            if self.coverage is not None and "coverage" in state:
                self.coverage.load_state(state["coverage"])
            if self.planner is not None and "planner" in state:
                self.planner.load_state(state["planner"])
        except (KeyError, ValueError, TypeError) as exc:
            # Corrupt state must never stop a session starting. Losing memory
            # is recoverable; refusing to broadcast is not.
            log.warning(
                "[%s] saved state could not be restored (%s); starting fresh",
                self.state.session_id, exc,
            )

    async def keep_fed(
        self, phase: MarketPhase, market: MarketState, now: datetime | None = None
    ) -> None:
        """Make sure the Director always has something to choose from.

        Continuous mode waives the score floor when the host has been quiet too
        long, but that does nothing if the queue is empty -- tick() returns
        early with no candidates. So in continuous mode the queue is topped up
        proactively rather than only when a market event or comment arrives.
        Quiet markets are exactly when this matters and exactly when nothing
        arrives on its own.
        """
        if self.max_silence_s is None:
            return
        now = now or utcnow()
        # One spare candidate is enough: more just ages in the queue and gets
        # dropped as expired.
        if self.director.queue_depth >= 2:
            return
        await self.offer_next_topic(phase, market, now)

    # -- the speak cycle --------------------------------------------------

    async def tick(self, market: MarketState, now: datetime | None = None) -> AIResponse | None:
        now = now or utcnow()
        decision: Decision | None = self.director.tick(now)
        if decision is None:
            return None

        trace_id = uuid4().hex[:12]
        result = await self._generate_non_repetitive(decision, market, now)
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
            self.unsafe_count += 1
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
        self.spoken_count += 1
        return response

    async def _generate_non_repetitive(
        self, decision: Decision, market: MarketState, now: datetime
    ):
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

        # The longer we have been silent, the more repetition we tolerate.
        # Without this the repetition gate can starve the stream completely --
        # a 48-hour weekend where every candidate is rejected produces 44 hours
        # of dead air, which is far worse than repeating a topic.
        quiet_s = self.memory.seconds_since_last_utterance(now)
        threshold = self.memory.repetition_threshold_for_silence(quiet_s)

        last = None
        for attempt in range(attempts):
            result = await self.generator.generate(
                persona=self.persona,
                intent=decision.intent,
                market=market,
                transcript=self.memory.recent_transcript(),
            )
            last = result
            repetitive, similarity = self.memory.is_repetitive(result.text, threshold)
            if not repetitive:
                return result
            log.info(
                "[%s] repetitive (sim=%.2f > %.2f, attempt %d/%d, %s)",
                self.state.session_id, similarity, threshold, attempt + 1, attempts,
                priority.name,
            )

        assert last is not None
        _, similarity = self.memory.is_repetitive(last.text, threshold)
        if priority is Priority.CRITICAL:
            log.warning(
                "[%s] speaking a repetitive utterance anyway - CRITICAL event "
                "must not be silently dropped (sim=%.2f)",
                self.state.session_id, similarity,
            )
            return last

        self.dropped_repetitive.append((last.text, similarity))
        return None

    #: Speaking rate used to estimate an utterance's duration without
    #: synthesising it. Only needs to be close: it tells the Director how long
    #: to consider itself busy, and erring slightly long is safer than short.
    WORDS_PER_MINUTE = 165

    def _estimate_duration_ms(self, segments: list[str]) -> int:
        words = sum(len(s.split()) for s in segments)
        return int((max(1, words) / self.WORDS_PER_MINUTE) * 60_000)

    async def _speak(self, response: AIResponse, decision: Decision) -> int:
        """Hand the utterance to whoever owns audio, and report how long it
        will take so the Director knows it is busy.

        When a router is attached it owns synthesis entirely -- queueing,
        barge-in and deadline dropping all live there. This method used to
        synthesise as well, which meant every utterance was rendered TWICE:
        once here and once in the router, doubling TTS cost and leaving the
        recorded file different from the audio actually played. The duration is
        estimated instead, which is all it was ever needed for.

        Without a router (the dry run) it still synthesises directly, since
        there is nothing else to write the audio.
        """
        req = AudioRequest(
            utterance_id=response.utterance_id,
            session_id=response.session_id,
            trace_id=response.trace_id,
            segments=response.segments,
            voice_id=self.persona.voice_id,
            priority=decision.intent.priority,
            preempts=decision.preempts,
        )

        if self.router is not None:
            await self.router.submit(req)
            return self._estimate_duration_ms(req.segments)

        total = 0
        for i, segment in enumerate(req.segments):
            # spoken_count, not len(transcript): the deque stops growing at
            # its cap, which would make filenames collide after that point.
            path = self.out_dir / f"{self.spoken_count:03d}_{req.utterance_id.hex[:8]}_{i}.wav"
            res = await self.tts.synthesize(segment, req.voice_id, path)
            total += res.duration_ms
        return total
