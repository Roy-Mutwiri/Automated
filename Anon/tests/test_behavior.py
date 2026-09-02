"""Regression tests for the properties that make the avatar look alive.

These lock in the specific bugs already found, so they cannot come back
silently. They are fast - the engine has no I/O and simulates faster than real
time - so they run on every change.
"""

from __future__ import annotations

import statistics

import pytest

from presenter.behavior import PROFILES, BehaviorEngine, BehaviorState
from presenter.behavior.curves import blink_profile, min_jerk

VOLUNTARY = {
    "gaze_shift", "gaze_left", "gaze_right", "gaze_down", "gaze_return",
    "head_yaw", "head_pitch", "head_roll", "expression", "posture_shift",
}


def run(minutes=10.0, profile="PRESENTER_CALM", seed=0, fps=30.0,
        state=BehaviorState.IDLE_ATTENTIVE):
    engine = BehaviorEngine(profile=profile, state=state, seed=seed)
    dt = 1.0 / fps
    for _ in range(int(minutes * 60.0 * fps)):
        engine.update(dt)
    return engine


def intervals(events, kinds):
    times = [e.time for e in events if e.kind in kinds]
    return [b - a for a, b in zip(times, times[1:])]


def cv(values):
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


# -- curves -----------------------------------------------------------------
def test_min_jerk_endpoints_are_stationary():
    assert min_jerk(0.0) == pytest.approx(0.0)
    assert min_jerk(1.0) == pytest.approx(1.0)
    # Near-zero velocity at both ends is what removes the mechanical snap.
    assert min_jerk(0.02) < 0.001
    assert min_jerk(0.98) > 0.999


def test_blink_closes_faster_than_it_opens():
    """The asymmetry is the single most recognisable feature of a real blink."""
    closure = [blink_profile(i / 200.0) for i in range(201)]
    peak = max(range(len(closure)), key=lambda i: closure[i])
    # Peak closure well before the midpoint means the closing phase is shorter.
    assert peak / 200.0 < 0.45
    assert closure[0] == pytest.approx(0.0, abs=1e-6)
    assert closure[-1] == pytest.approx(0.0, abs=1e-6)


# -- timing distributions ---------------------------------------------------
def test_blink_intervals_are_not_fixed():
    """Regression: the brief forbids fixed-interval blinking outright."""
    engine = run()
    iv = intervals(engine.peek_events(),
                   {"blink", "blink_partial", "double_blink_second"})
    assert len(iv) > 50
    assert cv(iv) > 0.35, "blink intervals look metronomic"
    assert max(iv) / min(iv) > 5.0, "blink interval range is too narrow"


def test_head_move_rate_matches_profile():
    """Regression for the suppression-baked-into-intervals bug.

    The motion budget was folded into the sampled interval, which stretched the
    observed median to ~3x the profile base and dropped the rate to 1.87/min.
    """
    engine = run(minutes=20.0)
    rate = engine.stats.head_moves / 20.0
    assert 2.0 <= rate <= 20.0, f"head move rate {rate:.2f}/min off target"


def test_frame_rate_invariance():
    """Behaviour must be identical at 25 and 60 FPS - it is time-driven."""
    slow = run(minutes=15.0, fps=25.0, seed=11)
    fast = run(minutes=15.0, fps=60.0, seed=11)
    for attr in ("blinks", "breaths"):
        a = getattr(slow.stats, attr) / 15.0
        b = getattr(fast.stats, attr) / 15.0
        assert abs(a - b) / max(a, b) < 0.15, f"{attr} varies with frame rate"


# -- the core requirement ---------------------------------------------------
def test_genuine_stillness_occurs():
    """Regression: an earlier tuning passed a weak bar while visibly fidgeting."""
    engine = run(minutes=20.0)
    times = [e.time for e in engine.peek_events() if e.kind in VOLUNTARY]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert statistics.median(gaps) >= 3.0, "avatar fidgets - median gap too short"
    assert sum(1 for g in gaps if g >= 3.0) / len(gaps) >= 0.50
    assert max(gaps) >= 10.0, "no genuinely still stretch occurs"
    assert len(times) / 20.0 <= 17.0, "too many voluntary movements per minute"


def test_involuntary_motion_never_stops():
    """Still must not mean frozen: eyes and breath run even during quiet."""
    engine = BehaviorEngine(profile="PRESENTER_FOCUSED", seed=3)
    gaze_positions, scales = [], []
    for _ in range(600):  # 20 s
        pose = engine.update(1.0 / 30.0)
        gaze_positions.append((pose.gaze_x, pose.gaze_y))
        scales.append(pose.scale)
    assert len(set(gaze_positions)) > 500, "gaze is frozen"
    assert max(scales) - min(scales) > 1e-4, "breathing is not moving the frame"


def test_face_is_never_perfectly_symmetric_during_a_blink():
    """Regression: perfect symmetry is listed as a bug in the brief."""
    engine = BehaviorEngine(seed=17)
    asymmetric = False
    for _ in range(30 * 60 * 2):  # 2 minutes
        pose = engine.update(1.0 / 30.0)
        if 0.05 < pose.eye_open_l < 0.95 and pose.eye_open_l != pose.eye_open_r:
            asymmetric = True
            break
    assert asymmetric, "both eyelids move identically"


# -- states and profiles ----------------------------------------------------
def test_focus_suppresses_and_speech_raises_blink_rate():
    """Matches the reported halving under focus / doubling in conversation."""
    idle = run(minutes=15.0, seed=5).stats.blinks / 15.0
    focused = run(minutes=15.0, seed=5, state=BehaviorState.FOCUSED).stats.blinks / 15.0
    speaking = run(minutes=15.0, seed=5, state=BehaviorState.SPEAKING).stats.blinks / 15.0
    assert focused < idle < speaking


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_every_profile_stays_within_anatomical_limits(profile):
    engine = BehaviorEngine(profile=profile, seed=2)
    p = PROFILES[profile]
    for _ in range(30 * 60 * 5):  # 5 minutes
        pose = engine.update(1.0 / 30.0)
        assert abs(pose.yaw) <= p.head_max_yaw * 1.35
        assert abs(pose.pitch) <= p.head_max_pitch * 1.35
        assert abs(pose.roll) <= p.head_max_roll * 1.35
        assert 0.0 <= pose.eye_open_l <= 1.0
        assert 0.0 <= pose.eye_open_r <= 1.0
        assert abs(pose.gaze_x) <= 0.56
        assert abs(pose.gaze_y) <= 0.46


def test_seeds_diverge():
    a = run(minutes=10.0, seed=1)
    b = run(minutes=10.0, seed=2)
    ka = [e.kind for e in a.peek_events()]
    kb = [e.kind for e in b.peek_events()]
    assert ka[:40] != kb[:40], "different seeds produce identical behaviour"


def test_long_stall_does_not_teleport_the_avatar():
    """A GPU hiccup must not integrate as one enormous step."""
    engine = BehaviorEngine(seed=9)
    for _ in range(300):
        engine.update(1.0 / 30.0)
    before = engine.update(1.0 / 30.0)
    after = engine.update(5.0)  # a 5-second stall
    assert abs(after.yaw - before.yaw) < 3.0
    assert abs(after.gaze_x - before.gaze_x) < 0.5


# -- blink visibility -------------------------------------------------------
def sample_blink(fps: float, seed: int = 4) -> list[float]:
    """Return eyelid closure sampled at one blink's rendered frames."""
    engine = BehaviorEngine(seed=seed)
    dt = 1.0 / fps
    for _ in range(int(3 * fps)):      # let the frame-interval estimate settle
        engine.update(dt)
    frames: list[float] = []
    capturing = False
    for _ in range(int(120 * fps)):
        pose = engine.update(dt)
        closed = 1.0 - min(pose.eye_open_l, pose.eye_open_r)
        if closed > 0.01:
            capturing = True
            frames.append(closed)
        elif capturing:
            if len(frames) >= 2:
                return frames
            frames, capturing = [], False
    return frames


@pytest.mark.parametrize("fps", [12.0, 15.0, 24.0, 30.0, 60.0])
def test_blink_spans_enough_frames_to_read_as_motion(fps):
    """Regression: at low frame rates a blink was a single-frame flash.

    Frame-rate-independent *timing* is not sufficient. A 145 ms blink sampled
    at 13 FPS produced closure 0.00 -> 0.88 -> 0.00: the lid appeared to
    teleport shut, which is precisely what reads as synthetic. The blink now
    stretches so it is sampled across several frames at any realistic rate.
    """
    frames = sample_blink(fps)
    assert len(frames) >= 4, (
        f"blink at {fps} FPS spans only {len(frames)} frames ({frames}) - "
        "it will read as a flash, not a movement"
    )
    # And it must actually pass through intermediate closure, not jump.
    assert any(0.15 < f < 0.85 for f in frames), (
        f"blink at {fps} FPS has no partially-closed frame: {frames}"
    )


def test_blink_stays_physiological_at_every_frame_rate():
    """The low-frame-rate stretch must not push blinks outside human range."""
    for fps in (10.0, 13.0, 30.0, 60.0):
        engine = BehaviorEngine(seed=8)
        dt = 1.0 / fps
        for _ in range(int(3 * fps)):
            engine.update(dt)
        durations = []
        start = None
        for _ in range(int(180 * fps)):
            pose = engine.update(dt)
            closed = 1.0 - min(pose.eye_open_l, pose.eye_open_r)
            if closed > 0.01 and start is None:
                start = engine.now
            elif closed <= 0.01 and start is not None:
                durations.append(engine.now - start)
                start = None
        assert durations, f"no blinks at {fps} FPS"
        longest = max(durations)
        assert longest <= 0.45, (
            f"blink of {longest * 1000:.0f} ms at {fps} FPS exceeds the "
            "100-400 ms physiological range"
        )


def test_blink_moves_the_brow():
    """A blink that moves only the eyelid reads as a shutter, not a person."""
    engine = BehaviorEngine(seed=21)
    coupled = False
    for _ in range(30 * 90):
        pose = engine.update(1.0 / 30.0)
        closure = 1.0 - min(pose.eye_open_l, pose.eye_open_r)
        if closure > 0.5:
            if pose.brow_l < -0.01 and pose.brow_r < -0.01:
                coupled = True
                break
    assert coupled, "brow does not follow the eyelid during a blink"
