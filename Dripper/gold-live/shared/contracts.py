"""The eight shared contracts.

This module is the interface between the intelligence half and the platform half.
Both sides import from here; neither hand-writes these types.

Pydantic models are the source of truth. JSON Schema is *generated* from them
(`python -m shared.contracts` writes schemas to shared/schemas/) rather than the
other way round -- one definition, no codegen step, and the schema stays exact.

Rules for changing anything in this file:
  - It gets its own PR. Never bundled with implementation.
  - Both owners approve (see CODEOWNERS).
  - Additive change -> bump minor. Removal or type change -> bump major.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

SCHEMA_VERSION = "1.0.0"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Contract(BaseModel):
    """Base for every contract. Frozen so a message can't be mutated in flight.

    `extra="forbid"` is deliberate: a typo in a field name must fail loudly
    rather than be silently dropped, which is exactly the kind of bug that
    survives to production in a system with this many moving parts.

    That interacts badly with computed fields, though. Pydantic WRITES computed
    fields into `model_dump_json()` output but treats them as extras on the way
    back in -- so any contract with a computed field could be published to the
    bus and never read back. The validator below strips them before validation,
    which keeps round-tripping working without giving up typo detection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and cls.model_computed_fields:
            computed = set(cls.model_computed_fields)
            if computed & data.keys():
                return {k: v for k, v in data.items() if k not in computed}
        return data


# ---------------------------------------------------------------------------
# Envelope -- wraps every event on the bus
# ---------------------------------------------------------------------------


class Envelope(Contract):
    event_id: UUID = Field(default_factory=uuid4)
    event_type: str
    schema_version: str = SCHEMA_VERSION
    #: When the fact became true, per the source's clock.
    occurred_at: datetime
    #: When we published it, per our clock. occurred_at - emitted_at = feed lag.
    emitted_at: datetime = Field(default_factory=utcnow)
    #: Joins every step of one utterance: trigger -> context -> generation -> audio.
    trace_id: str
    #: None means a shared-plane event that fans out to all sessions.
    session_id: str | None = None
    payload: dict[str, Any]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def lag_ms(self) -> int:
        return int((self.emitted_at - self.occurred_at).total_seconds() * 1000)


# ---------------------------------------------------------------------------
# 1. MarketState
# ---------------------------------------------------------------------------


class MarketConfidence(str, Enum):
    LIVE = "live"
    DELAYED = "delayed"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class Price(Contract):
    bid: float
    ask: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid(self) -> float:
        return round((self.bid + self.ask) / 2, 3)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread(self) -> float:
        return round(self.ask - self.bid, 3)


class Trend(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"


class Structure(str, Enum):
    HIGHER_HIGH = "higher_high"
    HIGHER_LOW = "higher_low"
    LOWER_HIGH = "lower_high"
    LOWER_LOW = "lower_low"
    CONSOLIDATION = "consolidation"


class TimeframeView(Contract):
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    trend: Trend
    structure: Structure
    swing_high: float | None = None
    swing_low: float | None = None
    atr: float | None = None


class Observation(Contract):
    """Measured. No judgement. 'The 5m range is 4.20.'"""

    key: str
    value: float | str
    unit: str | None = None
    timeframe: str | None = None


class Detection(Contract):
    """A deterministic rule fired. Carries its evidence so traces can explain it."""

    rule_id: str
    label: str
    timeframe: str
    price_level: float | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class MarketContext(Contract):
    """External facts: trading session, calendar, news."""

    kind: Literal["session", "calendar", "news"]
    label: str
    detail: str | None = None
    at: datetime | None = None


class TradingSession(str, Enum):
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OFF_HOURS = "off_hours"


class MarketState(Contract):
    """The shared snapshot. Computed once, fanned out to every session.

    `confidence` is a SAFETY CONTROL, not metadata. The generator must refuse
    to state a numeric price when confidence != LIVE. See intelligence.safety.
    """

    symbol: str = "XAUUSD"
    state_id: UUID = Field(default_factory=uuid4)
    #: Timestamp of the underlying tick.
    as_of: datetime
    computed_at: datetime = Field(default_factory=utcnow)
    confidence: MarketConfidence

    price: Price
    session: TradingSession
    timeframes: dict[str, TimeframeView] = Field(default_factory=dict)

    # Kept strictly separate by epistemic status. There is deliberately no
    # `interpretation` field -- interpretation is the Conversation Engine's job
    # and never travels on the bus.
    observations: list[Observation] = Field(default_factory=list)
    detections: list[Detection] = Field(default_factory=list)
    context: list[MarketContext] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def staleness_ms(self) -> int:
        return int((self.computed_at - self.as_of).total_seconds() * 1000)

    def may_quote_price(self) -> bool:
        """The single gate that stops the AI quoting a nine-minute-old price."""
        return self.confidence is MarketConfidence.LIVE


# ---------------------------------------------------------------------------
# 2. MarketEvent
# ---------------------------------------------------------------------------


class MarketEventKind(str, Enum):
    BOS = "bos"
    CHOCH = "choch"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    VOL_EXPANSION = "vol_expansion"
    SESSION_OPEN = "session_open"
    SESSION_CLOSE = "session_close"
    LEVEL_BREAK = "level_break"
    CALENDAR_RELEASE = "calendar_release"
    NEWS = "news"


class MarketEvent(Contract):
    event_id: UUID = Field(default_factory=uuid4)
    kind: MarketEventKind
    timeframe: str
    occurred_at: datetime
    #: 1-5. Set by deterministic rules, never by a model. Feeds Director priority.
    severity: Annotated[int, Field(ge=1, le=5)]
    price_level: float | None = None
    direction: Literal["up", "down"] | None = None
    #: Why the rule fired. Goes into the trace.
    evidence: dict[str, Any] = Field(default_factory=dict)
    #: Plain-language seed for the generator. Never spoken verbatim.
    narrative_hint: str | None = None
    #: Links the event back to the snapshot it was derived from.
    market_state_id: UUID | None = None


# ---------------------------------------------------------------------------
# 3. CommentEvent
# ---------------------------------------------------------------------------


class CommentIntent(str, Enum):
    MARKET_Q = "market_q"
    TECHNICAL_Q = "technical_q"
    EDUCATION_Q = "education_q"
    GREETING = "greeting"
    JOKE = "joke"
    SPAM = "spam"
    PROVOCATION = "provocation"
    OFF_TOPIC = "off_topic"
    TRADE_ADVICE_REQ = "trade_advice_req"


class CommentClassification(Contract):
    intent: CommentIntent
    #: Routes away from the normal generation path into a constrained one.
    #: Structural branch, not a prompt hint.
    is_risk_sensitive: bool = False
    relevance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    dedupe_cluster_id: str | None = None
    #: OCR confidence passthrough. Low-confidence text must not reach the generator.
    source_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class CommentEvent(Contract):
    #: Stamped at ingest, immutable thereafter. The isolation boundary.
    session_id: str
    platform: Literal["tiktok", "youtube", "twitch", "mock"]
    #: From the source where one exists; for screen capture this is
    #: sha256(author + text), deduped over a sliding window.
    platform_msg_id: str
    #: Salted hash. We do not store viewer display names.
    author_hash: str
    text_raw: str
    text_norm: str
    received_at: datetime = Field(default_factory=utcnow)
    classification: CommentClassification | None = None

    @staticmethod
    def synth_msg_id(author: str, text: str) -> str:
        """Identity for sources with no message ID (screen capture)."""
        return hashlib.sha256(f"{author}\x00{text}".encode()).hexdigest()[:32]

    @staticmethod
    def hash_author(author: str, salt: str) -> str:
        return hashlib.sha256(f"{salt}\x00{author}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 4. SessionState
# ---------------------------------------------------------------------------


class SessionStatus(str, Enum):
    STARTING = "starting"
    LIVE = "live"
    #: The state that matters: still broadcasting, but something is wrong and
    #: behaviour has been constrained (e.g. market stale -> no price quotes).
    DEGRADED = "degraded"
    PAUSED = "paused"
    STOPPING = "stopping"
    DOWN = "down"


class SpeakingState(Contract):
    is_speaking: bool = False
    utterance_id: UUID | None = None
    started_at: datetime | None = None
    est_end_at: datetime | None = None


class PlatformBinding(Contract):
    platform: Literal["tiktok", "youtube", "twitch", "mock"]
    channel_id: str
    adapter_health: Literal["ok", "degraded", "down"] = "ok"


class SessionState(Contract):
    model_config = ConfigDict(frozen=False, extra="forbid")  # mutated in place at runtime

    session_id: str
    status: SessionStatus = SessionStatus.STARTING
    persona_id: str
    platform_binding: PlatformBinding
    device_id: str | None = None
    speaking: SpeakingState = Field(default_factory=SpeakingState)
    topic_cooldowns: dict[str, datetime] = Field(default_factory=dict)
    recent_topics: list[str] = Field(default_factory=list)
    viewers: int = 0
    comments_1m: int = 0
    utterances_1h: int = 0
    last_spoke_at: datetime | None = None


# ---------------------------------------------------------------------------
# 5. AIResponse
# ---------------------------------------------------------------------------


class TriggerType(str, Enum):
    MARKET_EVENT = "market_event"
    COMMENT = "comment"
    SILENCE = "silence"
    SESSION_TRANSITION = "session_transition"
    EDUCATION = "education"


class Priority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TriggerRef(Contract):
    type: TriggerType
    source_event_id: UUID | None = None
    priority: Priority


class Provenance(Contract):
    """How we answer 'why did Session 4 say that?'"""

    market_state_id: UUID | None = None
    market_confidence: MarketConfidence | None = None
    comment_ids: list[str] = Field(default_factory=list)
    memory_refs: list[str] = Field(default_factory=list)
    model: str | None = None
    effort: str | None = None
    prompt_tokens: int | None = None
    cache_read_tokens: int | None = None
    first_token_ms: int | None = None
    generation_ms: int | None = None


class SafetyReport(Contract):
    passed: bool = True
    stated_price: bool = False
    has_disclaimer: bool = False
    violations: list[str] = Field(default_factory=list)


class AIResponse(Contract):
    utterance_id: UUID = Field(default_factory=uuid4)
    session_id: str
    trace_id: str
    text: str
    #: Sentence-split for streaming TTS. Segment 1 goes to audio while the
    #: model is still writing segment 3 -- this is worth ~2s of latency.
    segments: list[str] = Field(default_factory=list)
    trigger: TriggerRef
    provenance: Provenance = Field(default_factory=Provenance)
    safety: SafetyReport = Field(default_factory=SafetyReport)
    interruptible: bool = True
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# 6. AudioRequest
# ---------------------------------------------------------------------------


class AudioRequest(Contract):
    utterance_id: UUID
    session_id: str
    trace_id: str
    segments: list[str]
    voice_id: str
    priority: Priority = Priority.MEDIUM
    #: Utterance to cancel (barge-in).
    preempts: UUID | None = None
    #: Drop rather than speak stale commentary. A reaction to a sweep that
    #: arrives 40s late is worse than silence.
    deadline_ms: int = 30_000
    created_at: datetime = Field(default_factory=utcnow)

    def expired(self, now: datetime | None = None) -> bool:
        now = now or utcnow()
        return (now - self.created_at).total_seconds() * 1000 > self.deadline_ms


# ---------------------------------------------------------------------------
# 7. ServiceHealth
# ---------------------------------------------------------------------------


class HealthState(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    FAILING = "failing"
    DOWN = "down"


class HealthCheck(Contract):
    name: str
    ok: bool
    detail: str | None = None
    measured_at: datetime = Field(default_factory=utcnow)


class ServiceHealth(Contract):
    component: str
    session_id: str | None = None
    state: HealthState = HealthState.OK
    last_heartbeat: datetime = Field(default_factory=utcnow)
    checks: list[HealthCheck] = Field(default_factory=list)
    restart_count_1h: int = 0
    degraded_reason: str | None = None


# ---------------------------------------------------------------------------
# 8. DeviceState
# ---------------------------------------------------------------------------


class AudioDeviceState(Contract):
    output_device: str | None = None
    sample_rate: int = 24_000
    buffer_health: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    underruns_1h: int = 0
    #: A reassigned audio device is a silent failure that takes hours to notice.
    virtual_cable_present: bool = False


class CaptureCalibration(Contract):
    """Where the comment panel sits on this device's screen."""

    monitor: int = 0
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0
    crop_h: int = 0
    row_height_px: int = 28
    fps: int = 5
    min_ocr_confidence: float = 0.55


class DeviceState(Contract):
    device_id: str
    #: Authorisation scope. Enforced server-side -- a device may only ever
    #: operate the session it is bound to.
    bound_session: str
    agent_version: str
    audio: AudioDeviceState = Field(default_factory=AudioDeviceState)
    capture: CaptureCalibration | None = None
    last_seen: datetime = Field(default_factory=utcnow)
    rtt_ms: int | None = None
    platform_client_running: bool = False
    stream_live: bool = False


# ---------------------------------------------------------------------------
# Schema export
# ---------------------------------------------------------------------------

EXPORTED: dict[str, type[BaseModel]] = {
    "envelope": Envelope,
    "market_state": MarketState,
    "market_event": MarketEvent,
    "comment_event": CommentEvent,
    "session_state": SessionState,
    "ai_response": AIResponse,
    "audio_request": AudioRequest,
    "service_health": ServiceHealth,
    "device_state": DeviceState,
}


def export_schemas(dest: str = "shared/schemas") -> list[str]:
    """Write JSON Schema for every contract. Run in CI; diff to catch drift."""
    import json
    from pathlib import Path

    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in EXPORTED.items():
        path = out / f"{name}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
        written.append(str(path))
    return written


if __name__ == "__main__":
    for p in export_schemas():
        print(f"wrote {p}")
