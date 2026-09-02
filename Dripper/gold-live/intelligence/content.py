"""Content planning for continuous operation.

The problem this solves: a 24/7 stream cannot be driven by market events alone.
Spot gold is closed roughly 48 hours a week (Fri ~21:00 UTC to Sun ~22:00 UTC),
the Asian session is frequently flat, and even in an active London session there
are long stretches where nothing worth reacting to happens. Something has to
fill that time without inventing market movement and without repeating.

A flat list of topics does not work. At ~90 utterances an hour, any hand-written
list is exhausted inside a day. The inventory here is COMBINATORIAL instead:

    content item  x  angle  =  a distinct thing to say

45 topics x 8 angles = 360 distinct beats, and the same topic revisited from a
different angle six hours later is a genuinely different segment rather than a
repeat. Add a topic and you add eight beats, not one.

Exhaustion is measured, not assumed -- `coverage_report()` tells you how much of
the inventory is still available and when the system will start repeating. That
number belongs on the dashboard.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from shared.contracts import utcnow


class MarketPhase(str, Enum):
    """What the market is doing, which determines what is worth saying."""

    ACTIVE = "active"      # real movement; market commentary leads
    QUIET = "quiet"        # open but flat; education and audience lead
    CLOSED = "closed"      # weekend/rollover; no price talk at all


class Angle(str, Enum):
    """How to approach a topic. The multiplier on the inventory."""

    DEFINITION = "definition"
    COMMON_MISTAKE = "common_mistake"
    WORKED_EXAMPLE = "worked_example"
    HOW_IT_FAILS = "how_it_fails"
    CONTRAST = "contrast"
    HOW_TO_PRACTISE = "how_to_practise"
    HISTORICAL = "historical"
    AUDIENCE_PROMPT = "audience_prompt"


ANGLE_INSTRUCTION = {
    Angle.DEFINITION: "Explain what this actually means, concretely, without jargon.",
    Angle.COMMON_MISTAKE: "Talk about the mistake people usually make with this.",
    Angle.WORKED_EXAMPLE: "Walk through a specific example, using the chart in front of you if it helps.",
    Angle.HOW_IT_FAILS: "Talk about when this idea does NOT work, and what that looks like.",
    Angle.CONTRAST: "Contrast this with the thing people most often confuse it with.",
    Angle.HOW_TO_PRACTISE: "Give one concrete way someone could practise or verify this themselves.",
    Angle.HISTORICAL: "Reference a period or scenario where this mattered, without inventing specifics.",
    Angle.AUDIENCE_PROMPT: "Open this up to the audience and invite them to answer.",
}


@dataclass(frozen=True, slots=True)
class ContentItem:
    item_id: str
    category: str
    title: str
    seed: str
    #: 1 beginner, 2 intermediate, 3 advanced. Matched against persona audience.
    difficulty: int = 2
    #: Never usable when the market is closed (needs live price action).
    requires_live_market: bool = False


@dataclass(frozen=True, slots=True)
class Beat:
    """One deliverable segment: a topic seen from one angle."""

    item: ContentItem
    angle: Angle

    @property
    def key(self) -> str:
        return f"{self.item.item_id}:{self.angle.value}"

    @property
    def topic(self) -> str:
        return f"edu:{self.item.item_id}"

    def instruction(self) -> str:
        return f"{self.item.seed} {ANGLE_INSTRUCTION[self.angle]}"


@dataclass
class PlannerConfig:
    #: A specific beat may not repeat inside this window.
    beat_cooldown: timedelta = timedelta(hours=12)
    #: A topic may not be revisited from ANY angle inside this window.
    topic_cooldown: timedelta = timedelta(hours=2)
    #: Don't run the same category twice in a row.
    avoid_category_repeat: bool = True
    max_difficulty: int = 3
    #: Minimum wall time between handing out beats. Without this the planner is
    #: asked on every tick and spends the whole inventory in minutes, because a
    #: beat is consumed even when the utterance built from it is later dropped.
    #: Set to roughly the fastest rate a host would plausibly change subject.
    min_offer_interval: timedelta = timedelta(seconds=40)


class ContentPlanner:
    """One per session. Holds no reference to any other session's history.

    State is intentionally serialisable -- `export_state`/`load_state` let a
    restarted session process resume without forgetting what it covered at 3am,
    which is the whole point.
    """

    def __init__(
        self,
        items: list[ContentItem],
        config: PlannerConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.items = items
        self.config = config or PlannerConfig()
        self.rng = random.Random(seed)
        self.beat_last_used: dict[str, datetime] = {}
        self.topic_last_used: dict[str, datetime] = {}
        self.last_category: str | None = None
        self.served = 0
        self.last_offer_at: datetime | None = None
        #: 0 healthy, 3 fully out of fresh material. Belongs on the dashboard.
        self.degraded_level = 0

    # -- inventory --------------------------------------------------------

    def all_beats(self, phase: MarketPhase) -> list[Beat]:
        out = []
        for item in self.items:
            if item.difficulty > self.config.max_difficulty:
                continue
            if phase is MarketPhase.CLOSED and item.requires_live_market:
                continue
            for angle in Angle:
                if phase is MarketPhase.CLOSED and angle is Angle.WORKED_EXAMPLE:
                    continue  # no live chart to work through
                out.append(Beat(item=item, angle=angle))
        return out

    @property
    def inventory_size(self) -> int:
        return len(self.all_beats(MarketPhase.QUIET))

    # -- selection --------------------------------------------------------

    def _eligible(self, phase: MarketPhase, now: datetime) -> list[Beat]:
        beats = []
        for beat in self.all_beats(phase):
            last_beat = self.beat_last_used.get(beat.key)
            if last_beat and now - last_beat < self.config.beat_cooldown:
                continue
            last_topic = self.topic_last_used.get(beat.item.item_id)
            if last_topic and now - last_topic < self.config.topic_cooldown:
                continue
            if (
                self.config.avoid_category_repeat
                and beat.item.category == self.last_category
            ):
                continue
            beats.append(beat)
        return beats

    def next_beat(self, phase: MarketPhase, now: datetime | None = None) -> Beat | None:
        """Pick the next thing to talk about.

        Returns None both when rate-limited and when genuinely exhausted --
        callers that need to distinguish should check `is_exhausted`.
        """
        now = now or utcnow()

        if (
            self.last_offer_at is not None
            and now - self.last_offer_at < self.config.min_offer_interval
        ):
            return None

        beats = self._degrading_eligible(phase, now)
        if not beats:
            return None

        # Prefer the least recently used topic so coverage stays even rather
        # than clustering on whatever the RNG likes.
        beats.sort(
            key=lambda b: (
                self.topic_last_used.get(b.item.item_id, datetime.min.replace(tzinfo=now.tzinfo)),
                self.rng.random(),
            )
        )
        return beats[0]

    def _degrading_eligible(self, phase: MarketPhase, now: datetime) -> list[Beat]:
        """Eligible beats, relaxing constraints rather than returning nothing.

        A 48-hour weekend close will exhaust any hand-written inventory. When
        that happens the correct behaviour is to revisit the oldest material,
        NOT to fall silent -- 44 hours of dead air is a far worse outcome than
        repeating a topic the audience heard two days ago, and the audience
        during a weekend is largely not the same people anyway.

        Constraints are dropped in order of how much they cost to lose:
          1. everything enforced (the normal case)
          2. category variety
          3. per-beat cooldown  -- repeat an angle, oldest first
          4. topic cooldown     -- repeat a topic, oldest first
        `degraded_level` reports which rung we are on; anything above 0
        sustained for long is the signal to write more content.
        """
        beats = self._eligible(phase, now)
        if beats:
            self.degraded_level = 0
            return beats

        saved, self.last_category = self.last_category, None
        beats = self._eligible(phase, now)
        self.last_category = saved
        if beats:
            self.degraded_level = 1
            return beats

        pool = self.all_beats(phase)
        beats = [
            b for b in pool
            if not (
                (last := self.topic_last_used.get(b.item.item_id))
                and now - last < self.config.topic_cooldown
            )
        ]
        if beats:
            self.degraded_level = 2
            return beats

        self.degraded_level = 3
        return pool

    def is_exhausted(self, phase: MarketPhase, now: datetime | None = None) -> bool:
        """Genuinely out of material, as opposed to merely rate-limited."""
        return not self._eligible(phase, now or utcnow())

    def mark_used(self, beat: Beat, now: datetime | None = None) -> None:
        now = now or utcnow()
        self.beat_last_used[beat.key] = now
        self.topic_last_used[beat.item.item_id] = now
        self.last_category = beat.item.category
        self.last_offer_at = now
        self.served += 1

    # -- observability ----------------------------------------------------

    def coverage_report(self, phase: MarketPhase, now: datetime | None = None) -> dict[str, Any]:
        now = now or utcnow()
        total = len(self.all_beats(phase))
        available = len(self._eligible(phase, now))
        return {
            "phase": phase.value,
            "total_beats": total,
            "available_now": available,
            "served": self.served,
            "utilisation": round(1 - (available / total), 3) if total else 1.0,
            "distinct_topics": len(self.topic_last_used),
        }

    # -- persistence ------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        return {
            "beat_last_used": {k: v.isoformat() for k, v in self.beat_last_used.items()},
            "topic_last_used": {k: v.isoformat() for k, v in self.topic_last_used.items()},
            "last_category": self.last_category,
            "served": self.served,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.beat_last_used = {
            k: datetime.fromisoformat(v) for k, v in state.get("beat_last_used", {}).items()
        }
        self.topic_last_used = {
            k: datetime.fromisoformat(v) for k, v in state.get("topic_last_used", {}).items()
        }
        self.last_category = state.get("last_category")
        self.served = int(state.get("served", 0))


# ---------------------------------------------------------------------------
# Market phase
# ---------------------------------------------------------------------------

# Spot gold: closed Friday ~21:00 UTC to Sunday ~22:00 UTC.
FRIDAY, SATURDAY, SUNDAY = 4, 5, 6
CLOSE_HOUR_FRI = 21
OPEN_HOUR_SUN = 22


def market_is_closed(now: datetime) -> bool:
    wd = now.weekday()
    if wd == SATURDAY:
        return True
    if wd == FRIDAY and now.hour >= CLOSE_HOUR_FRI:
        return True
    if wd == SUNDAY and now.hour < OPEN_HOUR_SUN:
        return True
    return False


def classify_phase(now: datetime, *, atr: float | None, atr_baseline: float = 2.0) -> MarketPhase:
    """Closed beats quiet beats active. Volatility decides the rest."""
    if market_is_closed(now):
        return MarketPhase.CLOSED
    if atr is None:
        return MarketPhase.QUIET
    return MarketPhase.ACTIVE if atr >= atr_baseline else MarketPhase.QUIET


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_content(path: str | Path) -> list[ContentItem]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [
        ContentItem(
            item_id=d["item_id"],
            category=d["category"],
            title=d["title"],
            seed=d["seed"],
            difficulty=int(d.get("difficulty", 2)),
            requires_live_market=bool(d.get("requires_live_market", False)),
        )
        for d in data["items"]
    ]
