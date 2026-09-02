"""State survives a restart.

A 24/7 system restarts -- on crash, on upgrade, on a reboot. Without this the
host wakes up having forgotten every topic it covered and immediately repeats
an hour of material, which to a regular viewer looks exactly like a bug.

The export/load methods existed for a while but nothing called them, so a
restart lost everything. These pin the wiring as well as the serialisation.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path


from intelligence.content import ContentPlanner, load_content
from intelligence.generation import GenerationResult, Generator
from intelligence.personas import Persona
from intelligence.proposer import CoverageMemory, TopicProposer
from runtime.session import SessionRuntime
from shared.contracts import (
    MarketConfidence,
    MarketEvent,
    MarketEventKind,
    MarketState,
    PlatformBinding,
    Price,
    SessionState,
    TradingSession,
    utcnow,
)
from shared.mocks.tts import MockTTS
from shared.paths import resource_path
from shared.store import TraceStore


class FixedGenerator(Generator):
    def __init__(self, text: str = "A specific observation about structure.") -> None:
        self.text = text
        self.calls = 0

    async def generate(self, persona, intent, market, transcript) -> GenerationResult:
        self.calls += 1
        return GenerationResult(text=f"{self.text} ({self.calls})", segments=[self.text])


class DeadLLM:
    name = "dead"

    async def stream(self, *a, **k):
        raise ConnectionError("down")
        yield ""

    async def complete(self, *a, **k):
        raise ConnectionError("down")

    async def health(self):
        return False


def persona() -> Persona:
    return Persona(
        persona_id="p", display_name="P", audience="a", primary_timeframe="5m",
        focus=["structure"], avoid=[], voice_id="v", style="s",
    )


def market() -> MarketState:
    now = utcnow()
    return MarketState(
        as_of=now, computed_at=now, confidence=MarketConfidence.LIVE,
        price=Price(bid=3652.2, ask=3652.6), session=TradingSession.LONDON,
    )


def build_runtime(tmp_path: Path, with_planner: bool = True) -> SessionRuntime:
    planner = (
        ContentPlanner(load_content(resource_path("configs", "content.yaml")), seed=1)
        if with_planner
        else None
    )
    proposer = TopicProposer(DeadLLM(), CoverageMemory())  # type: ignore[arg-type]
    return SessionRuntime(
        state=SessionState(
            session_id="SESSION_001", persona_id="p",
            platform_binding=PlatformBinding(platform="mock", channel_id="c"),
        ),
        persona=persona(), generator=FixedGenerator(), tts=MockTTS(),
        out_dir=tmp_path, planner=planner, proposer=proposer,
    )


def event(severity: int = 5) -> MarketEvent:
    return MarketEvent(
        kind=MarketEventKind.BOS, timeframe="5m", occurred_at=utcnow(),
        severity=severity, narrative_hint="Break of structure.",
    )


# -- round trip ------------------------------------------------------------


async def test_topics_and_transcript_survive_a_restart(tmp_path):
    before = build_runtime(tmp_path)
    now = utcnow()
    before.on_market_event(event(), now)
    assert await before.tick(market(), now) is not None
    before.memory.record_question("where is resistance")

    state = before.export_state()

    after = build_runtime(tmp_path)
    after.load_state(state)

    assert after.memory.topics_last_seen.keys() == before.memory.topics_last_seen.keys()
    assert after.memory.utterance_count == before.memory.utterance_count
    assert after.memory.recent_transcript() == before.memory.recent_transcript()
    assert after.memory.audience_questions == before.memory.audience_questions


async def test_restored_session_does_not_repeat_itself(tmp_path):
    """The reason this matters: a restart must not undo repetition memory."""
    before = build_runtime(tmp_path)
    now = utcnow()
    before.on_market_event(event(), now)
    response = await before.tick(market(), now)
    assert response is not None

    after = build_runtime(tmp_path)
    after.load_state(before.export_state())

    repetitive, similarity = after.memory.is_repetitive(response.text)
    assert repetitive, f"restored session should recognise its own words (sim={similarity:.2f})"


async def test_topic_cooldowns_survive(tmp_path):
    before = build_runtime(tmp_path)
    now = utcnow()
    before.memory.record_utterance("Something about a sweep.", "mkt:liquidity_sweep", now)

    after = build_runtime(tmp_path)
    after.load_state(before.export_state())
    assert after.memory.topic_on_cooldown("mkt:liquidity_sweep", now)


async def test_content_planner_position_survives(tmp_path):
    """Otherwise a restarted session starts the fallback inventory from the top
    and covers the same ground it just finished."""
    before = build_runtime(tmp_path)
    now = utcnow()
    from intelligence.content import MarketPhase

    for i in range(4):
        at = now + timedelta(minutes=5 * i)
        beat = before.planner.next_beat(MarketPhase.QUIET, at)
        assert beat is not None
        before.planner.mark_used(beat, at)

    after = build_runtime(tmp_path)
    after.load_state(before.export_state())
    assert after.planner.served == before.planner.served
    assert after.planner.topic_last_used == before.planner.topic_last_used


async def test_coverage_memory_survives(tmp_path):
    before = build_runtime(tmp_path)
    before.coverage.record("round_number_stops", "why stops at round numbers get hit", utcnow())

    after = build_runtime(tmp_path)
    after.load_state(before.export_state())

    assert after.coverage.on_cooldown("round_number_stops", utcnow())
    covered, _ = after.coverage.is_covered("why stops at round numbers get hit")
    assert covered


# -- resilience ------------------------------------------------------------


async def test_corrupt_state_does_not_stop_a_session_starting(tmp_path):
    """Losing memory is recoverable. Refusing to broadcast is not."""
    runtime = build_runtime(tmp_path)
    runtime.load_state({"topics_last_seen": {"x": "not-a-timestamp"}})
    now = utcnow()
    runtime.on_market_event(event(), now)
    assert await runtime.tick(market(), now) is not None


async def test_empty_state_is_harmless(tmp_path):
    runtime = build_runtime(tmp_path)
    runtime.load_state({})
    assert runtime.memory.utterance_count == 0


async def test_state_without_a_planner_is_fine(tmp_path):
    before = build_runtime(tmp_path)
    before.memory.record_utterance("x", "t", utcnow())
    after = build_runtime(tmp_path, with_planner=False)
    after.load_state(before.export_state())
    assert after.memory.utterance_count == 1


# -- through the store -----------------------------------------------------


async def test_state_round_trips_through_the_database(tmp_path):
    store = TraceStore(tmp_path / "test.db")
    await store.start()

    before = build_runtime(tmp_path)
    now = utcnow()
    before.on_market_event(event(), now)
    await before.tick(market(), now)

    store.save_session_state("SESSION_001", before.export_state())
    await store._drain_once()

    loaded = store.load_session_state("SESSION_001")
    assert loaded is not None

    after = build_runtime(tmp_path)
    after.load_state(loaded)
    assert after.memory.utterance_count == before.memory.utterance_count
    await store.stop()


def test_missing_state_returns_none(tmp_path):
    store = TraceStore(tmp_path / "empty.db")
    store.migrate()
    assert store.load_session_state("SESSION_999") is None


def test_corrupt_stored_state_returns_none_rather_than_raising(tmp_path):
    from contextlib import closing

    store = TraceStore(tmp_path / "bad.db")
    store.migrate()
    with closing(store.connect()) as conn:
        conn.execute(
            "INSERT INTO session_state (session_id, updated_at, payload) VALUES (?,?,?)",
            ("SESSION_001", utcnow().isoformat(), "{not json"),
        )
        conn.commit()
    assert store.load_session_state("SESSION_001") is None
