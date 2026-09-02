"""Tests for attention, state scheduling, constraints and memory.

Most of these are regression tests for bugs found while building the layer, and
each names the failure it locks down. A test that only asserts the code does
what the code does is worth very little; a test that asserts the head comes back
to neutral is worth a lot, because it did not.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from presenter.behavior.attention import GAZE_UNITS_PER_DEG, AttentionSystem
from presenter.behavior.constraints import SEATED_PRESENTER, apply
from presenter.behavior.engine import BehaviorEngine
from presenter.behavior.scheduler import IDLE_STATE_GRAPH, StateScheduler
from presenter.behavior.state import PROFILES, BehaviorState
from presenter.types import AvatarPose


def run(minutes=5.0, seed=0, fps=30.0, profile="PRESENTER_CALM"):
    engine = BehaviorEngine(profile=profile, seed=seed)
    for _ in range(int(minutes * 60 * fps)):
        engine.update(1.0 / fps)
    return engine


# --- the head must come back -------------------------------------------------

def test_head_returns_to_neutral_when_gaze_returns_to_lens():
    """Regression: the head used to ratchet.

    The first version gave the head a share of each *shift* and then never
    corrected it: once the gaze was on target no further correction was
    computed, so the head stayed wherever the last turn left it and yaw drifted
    to 11.5 degrees over a minute of ordinary glancing. Anchoring the head to
    the *target's* eccentricity instead makes the lens a zero-share target, so
    looking back at the audience brings the head home on its own.
    """
    engine = run(minutes=6.0, seed=3)
    hist = []
    for _ in range(30 * 60 * 4):
        engine.update(1.0 / 30.0)
        if engine.attention.current == "LENS" and not engine.attention.is_shifting:
            hist.append(abs(engine.attention.head_yaw))
    assert hist, "never settled on the lens in four minutes"
    # Settled on the lens, the head should be essentially straight.
    assert statistics.median(hist) < 0.6


def test_head_yaw_does_not_drift_over_time():
    """The same failure seen from outside: no secular trend in head yaw."""
    engine = BehaviorEngine(seed=11)
    first, last = [], []
    n = 30 * 60 * 10
    for i in range(n):
        p = engine.update(1.0 / 30.0)
        if i < n // 5:
            first.append(p.yaw)
        elif i > n * 4 // 5:
            last.append(p.yaw)
    assert abs(statistics.fmean(last) - statistics.fmean(first)) < 1.0


# --- world space -------------------------------------------------------------

def test_attention_has_no_camera_input():
    """The hard requirement, enforced structurally.

    A camera cut cannot change the gaze if no behaviour module can see a
    camera. Rather than simulate a cut, assert the absence of the coupling:
    nothing in the attention module's surface mentions a camera, so there is no
    path by which one could be consulted.
    """
    import inspect

    from presenter.behavior import attention

    src = inspect.getsource(attention)
    for banned in ("active_camera", "camera_index", "current_camera", "cam_id"):
        assert banned not in src

    sys_ = AttentionSystem(PROFILES["PRESENTER_CALM"])
    public = {n for n in dir(sys_) if not n.startswith("_")}
    assert not {n for n in public if "camera" in n.lower()}


def test_gaze_is_deterministic_for_a_seed():
    """Same seed, same performance - so seven cameras can render one take."""
    a = [round(p.gaze_x, 9) for p in _poses(seed=5)]
    b = [round(p.gaze_x, 9) for p in _poses(seed=5)]
    assert a == b
    c = [round(p.gaze_x, 9) for p in _poses(seed=6)]
    assert a != c


def _poses(seed, frames=900):
    e = BehaviorEngine(seed=seed)
    return [e.update(1.0 / 30.0) for _ in range(frames)]


# --- eye/head division -------------------------------------------------------

def test_small_shifts_are_eyes_only():
    """Below the ocular threshold the neck should not move at all."""
    profile = PROFILES["PRESENTER_CALM"]
    a = AttentionSystem(profile)
    assert a._hold_share(0.0) == 0.0
    assert a._hold_share(8.0) == 0.0
    assert a._hold_share(10.9) == 0.0
    assert a._hold_share(21.0) > 0.0
    # The head never takes the whole of it; the eyes keep some eccentricity.
    assert a._hold_share(90.0) <= 0.63


def test_head_share_grows_with_eccentricity():
    a = AttentionSystem(PROFILES["PRESENTER_CALM"])
    shares = [a._hold_share(d) for d in (12, 18, 25, 34, 42)]
    assert shares == sorted(shares)


# --- state scheduling --------------------------------------------------------

def test_state_durations_respect_min_and_max():
    """Semi-Markov, not geometric: a state cannot end early by bad luck."""
    from presenter.behavior.randomness import Rng

    rng = Rng(4)
    sched = StateScheduler()
    sched.start(0.0, rng)
    now, seen = 0.0, []
    while now < 4000.0:
        now += 0.5
        prev, entered = sched.state, sched.entered_at
        if sched.update(now, rng) is not None:
            seen.append((prev, now - entered))
    assert len(seen) > 25
    for name, dur in seen:
        spec = IDLE_STATE_GRAPH[name]
        assert dur >= spec.min_duration - 0.6, f"{name} ended after {dur:.1f}s"
        assert dur <= spec.max_duration + 0.6


def test_state_cooldowns_prevent_flicker():
    """No NEUTRAL -> HAPPY -> NEUTRAL -> HAPPY. Not damped: unavailable."""
    from presenter.behavior.randomness import Rng

    rng = Rng(9)
    sched = StateScheduler()
    sched.start(0.0, rng)
    now = 0.0
    entries: dict[str, list[float]] = {}
    while now < 6000.0:
        now += 0.5
        new = sched.update(now, rng)
        if new:
            entries.setdefault(new, []).append(now)
    for name, times in entries.items():
        cd = IDLE_STATE_GRAPH[name].cooldown
        if cd <= 0 or len(times) < 2:
            continue
        assert min(b - a for a, b in zip(times, times[1:])) >= cd


def test_external_state_suspends_the_scheduler():
    """Regression: KeyError the moment anything set a non-idle state.

    When the content pipeline drives him into SPEAKING, that caller owns the
    state; the scheduler must stand down rather than look for successors of a
    state its graph has never heard of.
    """
    engine = BehaviorEngine(seed=2)
    engine.set_state(BehaviorState.SPEAKING)
    for _ in range(30 * 120):
        engine.update(1.0 / 30.0)
    assert engine.state == BehaviorState.SPEAKING


# --- blinking ----------------------------------------------------------------

def test_visual_demand_suppresses_blinking():
    """Reading suppresses blinks; the middle distance releases them.

    Measured ranges are 1.4-14.4/min reading against 10.5-32.5/min in
    conversation, so the ordering here is not a stylistic preference.
    """
    from presenter.behavior.blinking import BLINK_DEMAND_BASE, BLINK_DEMAND_SPAN
    from presenter.behavior.context import Drives
    from presenter.behavior.randomness import Rng
    from presenter.behavior.state import STATE_MODULATION, StateModulation

    def rate(demand):
        engine = BehaviorEngine(seed=8, autonomous=False)
        blinks = 0
        for _ in range(30 * 60 * 12):
            engine.attention.current = demand
            engine.update(1.0 / 30.0)
            blinks = engine.stats.blinks
        return blinks / 12.0

    reading = rate("MAIN_DISPLAY")
    lens = rate("LENS")
    away = rate("MIDDLE_DISTANCE")
    assert reading < lens < away
    assert 1.4 <= reading <= 14.4      # measured reading range
    assert 8.0 <= lens <= 21.0         # measured primary-gaze range


# --- constraints -------------------------------------------------------------

def test_constraints_bound_everything_and_are_smooth():
    """Anatomy wins, and it wins without a hard stop."""
    for raw in (5.0, 20.0, 60.0, 400.0, -400.0):
        p = AvatarPose(yaw=raw, pitch=raw, roll=raw,
                       eye_open_l=raw, eye_open_r=-raw,
                       gaze_x=raw, gaze_y=raw)
        apply(p)
        assert abs(p.yaw) <= SEATED_PRESENTER.yaw
        assert -SEATED_PRESENTER.pitch_down <= p.pitch <= SEATED_PRESENTER.pitch_up
        assert abs(p.roll) <= SEATED_PRESENTER.roll
        assert 0.0 <= p.eye_open_l <= 1.0
        assert 0.0 <= p.eye_open_r <= 1.0
        assert abs(p.gaze_x) <= 0.55

    # Small values pass through essentially untouched - the limit must not tax
    # ordinary motion.
    p = AvatarPose(yaw=2.0)
    apply(p)
    assert p.yaw == pytest.approx(2.0, abs=0.02)


# --- no loops ----------------------------------------------------------------

def test_no_repeated_voluntary_ngrams_over_thirty_minutes():
    engine = run(minutes=30.0, seed=17)
    repeats = engine.memory.repeated_ngrams(n=3, min_repeats=3)
    assert not repeats, f"repeating behaviour sequences: {repeats[:4]}"


def test_attention_does_not_alternate_between_two_targets():
    """A LENS, CHAT, LENS, CHAT cycle is the most obvious idle loop available."""
    engine = BehaviorEngine(seed=13)
    seq = []
    last = None
    for _ in range(30 * 60 * 20):
        engine.update(1.0 / 30.0)
        if engine.attention.current != last:
            last = engine.attention.current
            seq.append(last)
    pairs = [(a, b) for a, b in zip(seq, seq[1:])]
    assert len(seq) > 60
    # No single A->B transition may dominate the whole sequence.
    from collections import Counter
    top, count = Counter(pairs).most_common(1)[0]
    assert count / len(pairs) < 0.30, f"{top} is {count}/{len(pairs)} of shifts"


def test_stillness_periods_actually_occur():
    """The presenter must be capable of doing nothing for several seconds."""
    engine = run(minutes=10.0, seed=21)
    events = [e for e in engine.peek_events()
              if e.kind in ("attention", "head_yaw", "head_pitch", "head_roll",
                            "posture_shift", "expression")]
    gaps = [b.time - a.time for a, b in zip(events, events[1:])]
    assert max(gaps) > 6.0, "never still for more than six seconds"
    assert sum(1 for g in gaps if g > 3.0) / len(gaps) > 0.15
