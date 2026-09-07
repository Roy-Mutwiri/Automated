"""The host's state, and the rule that it is never invented.

Every assertion here exists to defend one property: the host's expression
corresponds to something that actually happened. A state model that drifts on
its own, or that reacts identically to the fourth break of structure as to the
first, produces exactly the fake-human effect it was built to avoid.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intelligence.host_state import (
    AROUSAL_BASELINE,
    SURPRISE_MIN_SEVERITY,
    HostState,
)
from shared.contracts import MarketConfidence, MarketEvent, MarketEventKind

T0 = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


def event(kind=MarketEventKind.BOS, severity=5, at=T0) -> MarketEvent:
    return MarketEvent(kind=kind, timeframe="5m", occurred_at=at, severity=severity)


# -- nothing happens --------------------------------------------------------


def test_a_quiet_host_stays_at_baseline():
    """The default case, and the one that matters most: with no events the
    host must not drift into excitement on its own."""
    state = HostState()
    state.decay_to(T0)
    state.decay_to(T0 + timedelta(minutes=30))
    assert state.arousal == pytest.approx(AROUSAL_BASELINE)
    assert state.surprise == 0.0
    assert state.uncertainty == 0.0
    assert state.surprises == 0


def test_decay_is_frame_rate_independent():
    """One tick of 60s and 60 ticks of 1s must land in the same place, or the
    host would sound different depending on how busy the loop was."""
    coarse, fine = HostState(), HostState()
    coarse.observe_event(event(), T0)
    fine.observe_event(event(), T0)

    coarse.decay_to(T0 + timedelta(seconds=60))
    for i in range(1, 61):
        fine.decay_to(T0 + timedelta(seconds=i))

    assert coarse.arousal == pytest.approx(fine.arousal, abs=1e-9)
    assert coarse.surprise == pytest.approx(fine.surprise, abs=1e-9)


def test_decay_never_runs_backwards():
    state = HostState()
    state.observe_event(event(), T0)
    state.decay_to(T0 + timedelta(seconds=30))
    before = state.arousal
    state.decay_to(T0 + timedelta(seconds=10))  # a clock that went backwards
    assert state.arousal == before


# -- events drive it --------------------------------------------------------


def test_a_big_event_raises_arousal_and_surprise():
    state = HostState()
    assert state.observe_event(event(severity=5), T0) is True
    assert state.arousal > AROUSAL_BASELINE
    assert state.surprise > 0.0


def test_severity_scales_the_response():
    small, large = HostState(), HostState()
    small.observe_event(event(severity=3), T0)
    large.observe_event(event(severity=5), T0)
    assert large.arousal > small.arousal
    assert large.surprise > small.surprise


def test_a_low_severity_event_is_never_startling():
    state = HostState()
    for severity in range(1, SURPRISE_MIN_SEVERITY):
        assert state.observe_event(event(severity=severity), T0) is False
    assert state.surprises == 0


def test_a_scheduled_event_is_never_startling():
    """A session opening at the time it opens every day is information, not a
    shock, however important it is."""
    state = HostState()
    assert (
        state.observe_event(
            event(kind=MarketEventKind.SESSION_OPEN, severity=5), T0
        )
        is False
    )


# -- novelty ----------------------------------------------------------------


def test_the_same_event_repeating_stops_being_startling():
    """The rule that stops reaction sounds firing on every tick. A host who
    gasps at the fourth identical break of structure is a host nobody
    believes."""
    state = HostState()
    assert state.observe_event(event(severity=4), T0) is True
    for i in range(1, 5):
        again = event(severity=4, at=T0 + timedelta(seconds=i))
        assert state.observe_event(again, T0 + timedelta(seconds=i)) is False
    assert state.surprises == 1


def test_an_escalation_is_startling_again():
    state = HostState()
    state.observe_event(event(severity=3), T0)
    escalated = event(severity=5, at=T0 + timedelta(seconds=5))
    assert state.observe_event(escalated, T0 + timedelta(seconds=5)) is True


def test_a_different_kind_of_event_is_startling_again():
    state = HostState()
    state.observe_event(event(kind=MarketEventKind.BOS, severity=4), T0)
    other = event(kind=MarketEventKind.LIQUIDITY_SWEEP, severity=4)
    assert state.observe_event(other, T0 + timedelta(seconds=3)) is True


def test_surprise_fades_much_faster_than_arousal():
    """A host still audibly startled thirty seconds later sounds unhinged;
    one still slightly lively sounds normal."""
    state = HostState()
    state.observe_event(event(severity=5), T0)
    state.decay_to(T0 + timedelta(seconds=30))
    assert state.surprise < 0.05
    assert state.arousal > AROUSAL_BASELINE


def test_repeated_events_cannot_stack_into_permanent_shock():
    state = HostState()
    kinds = [
        MarketEventKind.BOS,
        MarketEventKind.CHOCH,
        MarketEventKind.LIQUIDITY_SWEEP,
        MarketEventKind.VOL_EXPANSION,
    ]
    now = T0
    for i in range(40):
        now = T0 + timedelta(seconds=i)
        state.observe_event(event(kind=kinds[i % len(kinds)], severity=5, at=now), now)
    assert state.arousal <= 1.0
    assert state.surprise <= 1.0
    assert 0.0 <= state.energy <= 1.0


# -- data quality -----------------------------------------------------------


def test_a_stale_feed_raises_uncertainty():
    state = HostState()
    state.observe_market(MarketConfidence.LIVE, T0)
    assert state.uncertainty == 0.0
    state.observe_market(MarketConfidence.STALE, T0)
    assert state.uncertainty > 0.0


def test_regaining_a_live_feed_does_not_instantly_erase_doubt():
    """Discovering the feed is stale is a step change; trusting it again is
    gradual, which is also how a person behaves."""
    state = HostState()
    state.observe_market(MarketConfidence.STALE, T0)
    high = state.uncertainty
    state.observe_market(MarketConfidence.LIVE, T0)
    assert state.uncertainty == high  # unchanged; only decay lowers it
    state.decay_to(T0 + timedelta(seconds=60))
    assert state.uncertainty < high


# -- determinism ------------------------------------------------------------


def test_the_same_events_produce_the_same_state():
    """No randomness anywhere, so a recording can be reproduced."""

    def run() -> dict:
        state = HostState()
        for i, kind in enumerate(
            [MarketEventKind.BOS, MarketEventKind.VOL_EXPANSION, MarketEventKind.CHOCH]
        ):
            at = T0 + timedelta(seconds=i * 7)
            state.observe_event(event(kind=kind, severity=4, at=at), at)
        state.observe_market(MarketConfidence.DELAYED, T0 + timedelta(seconds=30))
        state.decay_to(T0 + timedelta(seconds=45))
        return state.snapshot()

    assert run() == run()
