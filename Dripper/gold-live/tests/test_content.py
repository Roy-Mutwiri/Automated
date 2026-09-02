"""Content planning: the layer that makes 24/7 operation possible.

These cover the failures found by the soak run -- inventory burned faster than
it could be spoken, and the planner falling silent instead of degrading.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from intelligence.content import (
    Angle,
    ContentItem,
    ContentPlanner,
    MarketPhase,
    PlannerConfig,
    classify_phase,
    load_content,
    market_is_closed,
)

ROOT = Path(__file__).resolve().parent.parent

TUESDAY = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
FRIDAY_LATE = datetime(2026, 9, 4, 22, 0, tzinfo=timezone.utc)
SATURDAY = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
SUNDAY_EARLY = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
SUNDAY_LATE = datetime(2026, 9, 6, 23, 0, tzinfo=timezone.utc)


def items(n: int = 4) -> list[ContentItem]:
    return [
        ContentItem(
            item_id=f"item_{i}", category=f"cat_{i % 2}", title=f"Topic {i}",
            seed=f"Seed text for topic {i}.",
        )
        for i in range(n)
    ]


def planner(n: int = 4, **cfg) -> ContentPlanner:
    return ContentPlanner(items(n), PlannerConfig(**cfg), seed=1)


# -- market hours ---------------------------------------------------------


@pytest.mark.parametrize(
    "when,closed",
    [
        (TUESDAY, False),
        (FRIDAY_LATE, True),
        (SATURDAY, True),
        (SUNDAY_EARLY, True),
        (SUNDAY_LATE, False),
    ],
)
def test_market_hours(when, closed):
    """Spot gold is shut ~48h a week. A 24/7 stream has to know that."""
    assert market_is_closed(when) is closed


def test_phase_classification():
    assert classify_phase(SATURDAY, atr=5.0) is MarketPhase.CLOSED
    assert classify_phase(TUESDAY, atr=5.0) is MarketPhase.ACTIVE
    assert classify_phase(TUESDAY, atr=0.5) is MarketPhase.QUIET
    assert classify_phase(TUESDAY, atr=None) is MarketPhase.QUIET


# -- inventory ------------------------------------------------------------


def test_inventory_is_combinatorial():
    """Adding one item adds eight beats, not one. That multiplier is the point."""
    p = planner(10)
    assert p.inventory_size == 10 * len(Angle)


def test_live_only_items_excluded_when_closed():
    p = ContentPlanner(
        [
            ContentItem(item_id="a", category="x", title="A", seed="s"),
            ContentItem(
                item_id="b", category="y", title="B", seed="s", requires_live_market=True
            ),
        ]
    )
    closed_ids = {b.item.item_id for b in p.all_beats(MarketPhase.CLOSED)}
    assert "b" not in closed_ids
    assert "a" in closed_ids


# -- rate limiting --------------------------------------------------------


def test_offers_are_rate_limited():
    """Without this the whole inventory is spent in minutes, then silence."""
    p = planner(20, min_offer_interval=timedelta(seconds=90))
    first = p.next_beat(MarketPhase.QUIET, TUESDAY)
    assert first is not None
    p.mark_used(first, TUESDAY)

    assert p.next_beat(MarketPhase.QUIET, TUESDAY + timedelta(seconds=30)) is None
    assert p.next_beat(MarketPhase.QUIET, TUESDAY + timedelta(seconds=120)) is not None


def test_rate_limited_is_not_exhausted():
    """None from next_beat must not be mistaken for running out of material."""
    p = planner(20)
    beat = p.next_beat(MarketPhase.QUIET, TUESDAY)
    assert beat is not None
    p.mark_used(beat, TUESDAY)
    assert p.next_beat(MarketPhase.QUIET, TUESDAY) is None
    assert not p.is_exhausted(MarketPhase.QUIET, TUESDAY)


# -- cooldowns and degradation -------------------------------------------


def test_topic_not_revisited_within_cooldown():
    p = planner(6, topic_cooldown=timedelta(hours=2), min_offer_interval=timedelta(0))
    beat = p.next_beat(MarketPhase.QUIET, TUESDAY)
    assert beat is not None
    p.mark_used(beat, TUESDAY)

    soon = TUESDAY + timedelta(minutes=30)
    for _ in range(5):
        nxt = p.next_beat(MarketPhase.QUIET, soon)
        assert nxt is None or nxt.item.item_id != beat.item.item_id


def test_planner_degrades_rather_than_falling_silent():
    """A 48h close exhausts any inventory. Repeating beats dead air."""
    p = planner(2, min_offer_interval=timedelta(0), topic_cooldown=timedelta(hours=6))
    now = TUESDAY
    served = 0
    for _ in range(60):
        beat = p.next_beat(MarketPhase.QUIET, now)
        assert beat is not None, "planner must never return nothing to say"
        p.mark_used(beat, now)
        served += 1
        now += timedelta(minutes=1)
    assert served == 60
    assert p.degraded_level > 0, "should report that it is reusing material"


def test_degraded_level_reports_healthy_when_fresh():
    p = planner(40, min_offer_interval=timedelta(0))
    beat = p.next_beat(MarketPhase.QUIET, TUESDAY)
    assert beat is not None
    assert p.degraded_level == 0


# -- persistence ----------------------------------------------------------


def test_state_survives_a_restart():
    """A restarted session must not forget what it covered at 3am."""
    a = planner(8, min_offer_interval=timedelta(0))
    for i in range(5):
        beat = a.next_beat(MarketPhase.QUIET, TUESDAY + timedelta(minutes=i))
        assert beat is not None
        a.mark_used(beat, TUESDAY + timedelta(minutes=i))

    b = planner(8, min_offer_interval=timedelta(0))
    b.load_state(a.export_state())
    assert b.served == a.served
    assert b.beat_last_used == a.beat_last_used
    assert b.topic_last_used == a.topic_last_used


# -- real inventory -------------------------------------------------------


def test_shipped_inventory_loads_and_is_substantial():
    loaded = load_content(ROOT / "configs" / "content.yaml")
    assert len(loaded) >= 40
    assert len({i.item_id for i in loaded}) == len(loaded), "duplicate item_id"
    p = ContentPlanner(loaded)
    assert p.inventory_size >= 300
    # Live-only items must be a small minority or weekends have nothing to say.
    live_only = [i for i in loaded if i.requires_live_market]
    assert len(live_only) < len(loaded) * 0.2
