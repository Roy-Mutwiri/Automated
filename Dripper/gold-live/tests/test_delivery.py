"""Delivery planning, and the bounds that keep it from producing bad audio.

Two kinds of test here. The first kind checks that delivery corresponds to
events -- that the host speeds up because something happened, not otherwise.
The second kind is the safety net: whatever produced a plan, only bounded
values reach the synthesiser. That matters more than it looks, because the
plan layer is explicitly designed so that an LLM could author plans later, and
a model that emits `pace: 4.0` must not be able to render twenty seconds of
smeared audio on a live stream.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intelligence.delivery import (
    BIG_SURPRISE_REACTION,
    REACTION_COOLDOWN_S,
    REACTIONS,
    DeliveryPlanner,
)
from intelligence.host_state import HostState
from shared.contracts import MarketConfidence, MarketEvent, MarketEventKind, Priority
from shared.delivery import (
    GAP_DEFAULT_MS,
    GAP_MAX_MS,
    GAP_MIN_MS,
    NULL_CAPABILITIES,
    PACE_MAX,
    PACE_MIN,
    PAUSE_MAX_MS,
    PAUSE_MIN_AUDIBLE_MS,
    PIPER_CAPABILITIES,
    REACTION_MAX_CHARS,
    DeliveryCapabilities,
    DeliveryPlan,
)

T0 = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


def event(kind=MarketEventKind.BOS, severity=5, at=T0) -> MarketEvent:
    return MarketEvent(kind=kind, timeframe="5m", occurred_at=at, severity=severity)


def startled(kind=MarketEventKind.BOS, severity=5) -> tuple[HostState, MarketEventKind]:
    state = HostState()
    assert state.observe_event(event(kind, severity), T0) is True
    return state, kind


# -- the default case -------------------------------------------------------


def test_normal_speech_is_the_default_when_nothing_happened():
    """The most important test in this file. Most utterances follow no event
    at all, and those must sound ordinary -- a host who is permanently
    animated is as obviously synthetic as one who is permanently flat."""
    plan = DeliveryPlanner().plan(HostState(), T0).renderable_with(PIPER_CAPABILITIES)
    assert plan.reaction is None
    assert plan.pace == pytest.approx(1.0, abs=0.02)
    assert plan.pause_before_ms == 0
    assert plan.pause_after_ms == 0


def test_a_neutral_plan_is_genuinely_neutral():
    plan = DeliveryPlan.neutral().renderable_with(PIPER_CAPABILITIES)
    assert (plan.pace, plan.pause_before_ms, plan.pause_after_ms) == (1.0, 0, 0)
    assert plan.reaction is None


def test_no_event_means_no_reaction_however_often_it_is_asked():
    planner = DeliveryPlanner()
    for i in range(50):
        plan = planner.plan(HostState(), T0 + timedelta(seconds=i))
        assert plan.reaction is None


# -- events change delivery -------------------------------------------------


def test_a_startling_event_speeds_the_host_up_and_adds_a_beat_before():
    state, kind = startled()
    plan = DeliveryPlanner().plan(state, T0, startling_event=kind)
    quiet = DeliveryPlanner().plan(HostState(), T0)
    assert plan.pace > quiet.pace
    assert plan.pause_before_ms > 0
    assert plan.gap_ms < quiet.gap_ms, "an energised host leaves less air"


def test_uncertainty_slows_the_host_down():
    unsure = HostState()
    unsure.observe_market(MarketConfidence.STALE, T0)
    plan = DeliveryPlanner().plan(unsure, T0)
    assert plan.pace < DeliveryPlanner().plan(HostState(), T0).pace
    assert plan.pause_before_ms > 0


def test_delivery_returns_to_normal_once_the_moment_passes():
    state, kind = startled()
    planner = DeliveryPlanner()
    excited = planner.plan(state, T0, startling_event=kind)
    state.decay_to(T0 + timedelta(minutes=3))
    later = planner.plan(state, T0 + timedelta(minutes=3))
    assert later.pause_before_ms == 0
    assert later.pace < excited.pace


# -- reactions --------------------------------------------------------------


def test_the_reaction_matches_the_event_that_caused_it():
    state, kind = startled(MarketEventKind.LIQUIDITY_SWEEP, severity=4)
    plan = DeliveryPlanner().plan(state, T0, startling_event=kind)
    assert plan.reaction in (REACTIONS[kind], BIG_SURPRISE_REACTION)


def test_a_merely_notable_event_moves_the_state_without_a_spoken_reaction():
    """Severity 3 is enough to change how the host sounds and not enough to
    make a noise about. Reacting out loud to every moderate event is the
    fake-human failure in its most obvious form."""
    state, kind = startled(MarketEventKind.LIQUIDITY_SWEEP, severity=3)
    plan = DeliveryPlanner().plan(state, T0, startling_event=kind)
    assert plan.reaction is None
    assert plan.pace > DeliveryPlanner().plan(HostState(), T0).pace


def test_a_very_big_surprise_overrides_the_per_kind_word():
    state = HostState()
    state.observe_event(event(MarketEventKind.BOS, 5), T0)
    state.observe_event(event(MarketEventKind.VOL_EXPANSION, 5), T0)
    assert state.surprise >= 0.8
    plan = DeliveryPlanner().plan(
        state, T0, startling_event=MarketEventKind.VOL_EXPANSION
    )
    assert plan.reaction == BIG_SURPRISE_REACTION


def test_reactions_are_rate_limited_even_under_a_storm_of_events():
    """The backstop behind the novelty rule. Reaction units must not be able
    to run away, whatever the market does."""
    planner = DeliveryPlanner()
    kinds = [
        MarketEventKind.BOS,
        MarketEventKind.CHOCH,
        MarketEventKind.LIQUIDITY_SWEEP,
        MarketEventKind.VOL_EXPANSION,
    ]
    state = HostState()
    reacted = 0
    for i in range(120):  # two minutes of an event every second
        now = T0 + timedelta(seconds=i)
        kind = kinds[i % len(kinds)]
        was_startling = state.observe_event(event(kind, 5, now), now)
        plan = planner.plan(
            state, now, startling_event=kind if was_startling else None
        )
        if plan.reaction:
            reacted += 1
    assert reacted <= 120 / REACTION_COOLDOWN_S + 1, f"reacted {reacted} times"


def test_a_reaction_is_never_longer_than_a_word():
    for text in REACTIONS.values():
        assert len(text) <= REACTION_MAX_CHARS


def test_an_over_long_reaction_is_discarded_rather_than_spoken():
    """The guard that stops a reaction unit becoming runaway audio, whatever
    produced the string."""
    plan = DeliveryPlan(reaction="Well now that is a genuinely enormous move isn't it")
    assert plan.renderable_with(PIPER_CAPABILITIES).reaction is None


@pytest.mark.parametrize("text", ["", "   ", None])
def test_an_empty_reaction_is_not_a_reaction(text):
    assert DeliveryPlan(reaction=text).renderable_with(PIPER_CAPABILITIES).reaction is None


# -- bounds: expressive metadata cannot produce pathological audio ----------


@pytest.mark.parametrize(
    "plan",
    [
        DeliveryPlan(pace=9.0),
        DeliveryPlan(pace=-4.0),
        DeliveryPlan(pace=0.0),
        DeliveryPlan(pause_before_ms=90_000),
        DeliveryPlan(pause_after_ms=-500),
        DeliveryPlan(gap_ms=30_000),
        DeliveryPlan(gap_ms=-1),
    ],
)
def test_no_plan_can_drive_the_renderer_outside_its_measured_range(plan):
    """Whatever built the plan -- today the planner, tomorrow possibly a model
    -- only bounded values reach the synthesiser."""
    r = plan.renderable_with(PIPER_CAPABILITIES)
    assert PACE_MIN <= r.pace <= PACE_MAX
    assert 0 <= r.pause_before_ms <= PAUSE_MAX_MS
    assert 0 <= r.pause_after_ms <= PAUSE_MAX_MS
    assert GAP_MIN_MS <= r.gap_ms <= GAP_MAX_MS


def test_planned_output_is_always_inside_the_bounds():
    """Not just adversarial input -- the planner's own output, across the
    whole reachable state space."""
    planner = DeliveryPlanner()
    state = HostState()
    for i in range(200):
        now = T0 + timedelta(seconds=i * 3)
        if i % 3 == 0:
            state.observe_event(event(severity=(i % 5) + 1, at=now), now)
        if i % 7 == 0:
            state.observe_market(MarketConfidence.STALE, now)
        r = planner.plan(state, now).renderable_with(PIPER_CAPABILITIES)
        assert PACE_MIN <= r.pace <= PACE_MAX
        assert 0 <= r.pause_before_ms <= PAUSE_MAX_MS
        assert GAP_MIN_MS <= r.gap_ms <= GAP_MAX_MS


# -- capability degradation -------------------------------------------------


def test_a_backend_that_can_do_nothing_receives_a_neutral_plan():
    state, kind = startled()
    plan = DeliveryPlanner().plan(state, T0, startling_event=kind)
    assert plan.reaction is not None

    degraded = plan.renderable_with(NULL_CAPABILITIES)
    assert degraded.reaction is None
    assert degraded.pace == 1.0
    assert degraded.pause_before_ms == 0
    assert degraded.pause_after_ms == 0


def test_unsupported_expressive_fields_are_stripped_not_silently_assumed():
    """Piper cannot emphasise a word, hit a pitch target or laugh. Carrying
    those fields is fine; pretending they were honoured is not."""
    plan = DeliveryPlan(emphasis=("actually", "big"), pitch_bias=0.8, laugh=True)
    r = plan.renderable_with(PIPER_CAPABILITIES)
    assert r.emphasis == ()
    assert r.pitch_bias == 0.0
    assert r.laugh is False


def test_a_richer_backend_would_keep_them_without_any_code_change():
    """The point of the capability layer: replacing Piper turns fields on."""
    rich = DeliveryCapabilities(
        pace=True, pauses=True, reactions=True, emphasis=True, pitch=True, laughter=True
    )
    plan = DeliveryPlan(emphasis=("big",), pitch_bias=0.4, laugh=True)
    r = plan.renderable_with(rich)
    assert r.emphasis == ("big",)
    assert r.pitch_bias == 0.4
    assert r.laugh is True


# -- determinism ------------------------------------------------------------


def test_the_same_state_plans_the_same_delivery():
    state, kind = startled()
    a = DeliveryPlanner().plan(state, T0, startling_event=kind)
    b = DeliveryPlanner().plan(state, T0, startling_event=kind)
    assert a == b


def test_priority_does_not_secretly_randomise_anything():
    state = HostState()
    plans = {
        DeliveryPlanner().plan(state, T0, priority=p)
        for p in (Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL)
    }
    assert len(plans) == 1

# -- what the renderer can actually express ---------------------------------


def test_a_resting_host_speaks_at_the_voices_natural_rate():
    """"Normal" has to be exactly normal. The earlier formula left an idle
    host permanently 1.8% slow for no reason at all."""
    plan = DeliveryPlanner().plan(HostState(), T0)
    assert plan.pace == pytest.approx(1.0, abs=1e-9)


def test_a_resting_host_gets_the_default_gap():
    plan = DeliveryPlanner().plan(HostState(), T0).renderable_with(PIPER_CAPABILITIES)
    assert plan.gap_ms == GAP_DEFAULT_MS


@pytest.mark.parametrize("ms", [1, 10, 45, 59])
def test_a_pause_too_short_to_hear_is_not_rendered(ms):
    """Below roughly 50-80ms a gap reads as a stop consonant, not silence. A
    real run was producing `before=10ms`, which cost a code path to render
    something no listener could perceive."""
    assert (
        DeliveryPlan(pause_before_ms=ms).renderable_with(PIPER_CAPABILITIES).pause_before_ms
        == 0
    )


def test_a_pause_at_the_audibility_floor_survives():
    plan = DeliveryPlan(pause_before_ms=PAUSE_MIN_AUDIBLE_MS)
    assert plan.renderable_with(PIPER_CAPABILITIES).pause_before_ms == PAUSE_MIN_AUDIBLE_MS


def test_arousal_moves_the_one_channel_piper_renders_exactly():
    """Rate cannot carry arousal on Piper: across the whole reachable range the
    rendered duration of a line moves 2-5%, and Piper's own run-to-run spread
    on identical input is ~3.5%, so the signal sits inside the engine's noise.
    Silence is exact, so arousal spends its budget there instead. If this
    range narrows, arousal stops being audible at all."""
    quiet = DeliveryPlanner().plan(HostState(), T0).renderable_with(PIPER_CAPABILITIES)

    excited = HostState()
    for i in range(3):
        at = T0 + timedelta(seconds=i)
        excited.observe_event(event(severity=5, at=at), at)
    loud = DeliveryPlanner().plan(excited, T0 + timedelta(seconds=3))
    loud = loud.renderable_with(PIPER_CAPABILITIES)

    assert quiet.gap_ms - loud.gap_ms >= 40, (
        f"arousal only moves the gap {quiet.gap_ms - loud.gap_ms}ms, "
        "which is not enough to hear"
    )
