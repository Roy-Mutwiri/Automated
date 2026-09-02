"""Tests for the canonical motion state, breathing, body and adapters.

The first test is the important one. Everything else here checks that a
subsystem does what it says; that one checks that the *architecture* holds, and
it is the only thing standing between this design and a slow slide back into a
renderer-coupled behaviour engine.
"""

from __future__ import annotations

import ast
import math
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from presenter.behavior.engine import BehaviorEngine
from presenter.motion.adapters.face2d import to_avatar_pose
from presenter.motion.body import SEATED_NEUTRAL, neutral_head_baseline
from presenter.motion.breathing import BREATH_COUPLING
from presenter.motion.state import HumanMotionState, JointRotation


# --- the architectural rule --------------------------------------------------

FORBIDDEN = {"bpy", "cv2", "torch", "torchvision", "mathutils", "PIL",
             "diffusers", "onnxruntime"}


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_behaviour_and_motion_never_import_a_renderer():
    """No `MPFBBehaviorEngine`, and no way to grow one.

    The rule is that behaviour is renderer- and character-independent. A rule
    that is only written down decays; this makes it fail a test. Adapters are
    the one place allowed to know about a renderer, so they are excluded - that
    is their entire job.
    """
    roots = [ROOT / "src" / "presenter" / "behavior",
             ROOT / "src" / "presenter" / "motion"]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            if "adapters" in path.parts:
                continue
            bad = _imports_of(path) & FORBIDDEN
            if bad:
                offenders.append(f"{path.relative_to(ROOT)} imports {sorted(bad)}")
    assert not offenders, "behaviour must not import a renderer:\n" + "\n".join(offenders)


def test_motion_state_does_not_know_about_avatar_pose():
    """The canonical state must not be shaped by one renderer's vocabulary."""
    src = (ROOT / "src" / "presenter" / "motion" / "state.py").read_text(encoding="utf-8")
    assert "AvatarPose" not in src.split('"""')[2]  # body, not the docstring


# --- breathing ---------------------------------------------------------------

def test_breath_coupling_decreases_up_the_body():
    """Rib cage primary, head near zero. The ordering is the anatomy."""
    mag = {k: abs(v[0]) for k, v in BREATH_COUPLING.items()}
    assert mag["chest"] > mag["spine_mid"] > mag["shoulder_l"] > mag["head"]
    assert mag["head"] < 0.1 * mag["chest"], "the head must barely move"
    assert mag["clavicle_l"] < 0.4 * mag["chest"]


def test_breath_neck_opposes_chest():
    """The neck compensates so the gaze stays level while the chest opens.

    Without this the head nods in time with the breath, which is the clearest
    way a breathing rig announces itself.
    """
    assert BREATH_COUPLING["chest"][0] < 0 < BREATH_COUPLING["neck"][0]


def test_breathing_never_scales_the_head():
    engine = BehaviorEngine(seed=4)
    for _ in range(30 * 60):
        pose = engine.update(1.0 / 30.0)
        assert pose.scale == 1.0
        assert pose.tx == 0.0 and pose.ty == 0.0


def test_breath_waveform_is_asymmetric_with_a_pause():
    """Inhale shorter than exhale, and a rest at the bottom. Not a sine."""
    engine = BehaviorEngine(seed=6)
    drive = []
    for _ in range(30 * 40):
        engine.update(1.0 / 30.0)
        drive.append(engine.motion.breathing.drive)

    rising = sum(1 for a, b in zip(drive, drive[1:]) if b > a + 1e-6)
    falling = sum(1 for a, b in zip(drive, drive[1:]) if b < a - 1e-6)
    flat = sum(1 for a, b in zip(drive, drive[1:]) if abs(b - a) <= 1e-6)
    assert rising < falling, "inhale should be shorter than exhale"
    assert flat > 0.05 * len(drive), "there is no pause between breaths"
    assert max(drive) > 0.95 and min(drive) < 0.05


def test_breath_rate_varies_slowly_not_per_cycle():
    """Physiology drifts; it does not re-roll every breath."""
    engine = BehaviorEngine(seed=8)
    rates = []
    for i in range(30 * 240):
        engine.update(1.0 / 30.0)
        if i % 30 == 0:
            rates.append(engine.motion.breathing.rate)
    diffs = [abs(b - a) for a, b in zip(rates, rates[1:])]
    # Second-to-second change must be far smaller than the overall spread.
    assert statistics.fmean(diffs) < 0.15 * (max(rates) - min(rates) + 1e-6)
    assert 8.0 < statistics.fmean(rates) < 24.0


# --- posture -----------------------------------------------------------------

def test_seated_neutral_is_asymmetric():
    """Nobody sits square. A symmetric neutral is a T-pose in a chair."""
    assert SEATED_NEUTRAL["clavicle_l"][2] != SEATED_NEUTRAL["clavicle_r"][2]
    assert SEATED_NEUTRAL["shoulder_l"] != SEATED_NEUTRAL["shoulder_r"]
    assert SEATED_NEUTRAL["wrist_l"] != SEATED_NEUTRAL["wrist_r"]
    assert any(v != 0.0 for v in SEATED_NEUTRAL["head"])


def test_comfort_shifts_are_rare():
    engine = BehaviorEngine(seed=12)
    for _ in range(30 * 60 * 10):
        engine.update(1.0 / 30.0)
    per_min = engine.body.shift_count / 10.0
    assert 0.1 <= per_min <= 2.0, f"{per_min:.2f} posture shifts/min"


def test_hands_are_always_in_contact():
    """No floating wrists. A hand resting on nothing is not resting."""
    engine = BehaviorEngine(seed=2)
    for _ in range(30 * 60 * 3):
        engine.update(1.0 / 30.0)
        assert engine.motion.hand_r.contact is not None
        assert engine.motion.hand_l.contact is not None


def test_resting_fingers_are_never_straight_or_equal():
    m = HumanMotionState()
    for hand in (m.hand_l, m.hand_r):
        assert all(c > 0.05 for c in hand.curl), "straight fingers are a mannequin"
        assert len(set(hand.curl)) > 3, "every finger at the same curl is a glove"


# --- adapters ----------------------------------------------------------------

def test_face2d_subtracts_the_rest_pose():
    """The 2D renderer wants deltas; the plate already contains the posture.

    Regression: feeding it the absolute chain counted the seated neutral twice
    and tipped the head down 7.8 degrees.
    """
    m = HumanMotionState()
    for name, (rx, ry, rz) in SEATED_NEUTRAL.items():
        j = m.joints().get(name)
        if j is not None:
            j.rx, j.ry, j.rz = rx, ry, rz
    pose = to_avatar_pose(m)
    assert abs(pose.pitch) < 1e-6
    assert abs(pose.yaw) < 1e-6
    assert abs(pose.roll) < 1e-6


def test_face2d_sums_the_whole_chain():
    """A turned chest points the face somewhere, even to a face-only renderer."""
    m = HumanMotionState()
    m.chest.ry = 4.0
    m.neck.ry = 3.0
    m.head.ry = 2.0
    b_rx, b_ry, b_rz = neutral_head_baseline()
    assert to_avatar_pose(m).yaw == pytest.approx(9.0 - b_ry, abs=1e-6)


# --- numerical health --------------------------------------------------------

def test_no_nans_and_no_frame_jumps_anywhere():
    """Every joint, every frame: finite, and no teleports."""
    engine = BehaviorEngine(seed=15)
    prev = None
    worst = {"eye": 0.0, "body": 0.0}
    for _ in range(30 * 60 * 5):
        engine.update(1.0 / 30.0)
        cur = {}
        for name, j in engine.motion.joints().items():
            for axis, v in (("rx", j.rx), ("ry", j.ry), ("rz", j.rz)):
                assert math.isfinite(v), f"{name}.{axis} is {v}"
                cur[f"{name}.{axis}"] = v
        if prev is not None:
            for k, v in cur.items():
                d = abs(v - prev[k])
                key = "eye" if k.startswith("eye_") else "body"
                worst[key] = max(worst[key], d)
        prev = cur

    # Two limits, because eyes and bodies move at completely different speeds
    # and a single threshold tests neither. A saccade is ballistic: the main
    # sequence puts peak velocity at 400-600 deg/s, so 17 deg in a 30 fps frame
    # is not a pop, it is a correctly fast eye. A neck that moved that fast
    # would be a whiplash injury.
    fps = 30.0
    assert worst["eye"] * fps < 650.0, (
        f"eye peak {worst['eye'] * fps:.0f} deg/s exceeds saccadic velocity")
    assert worst["eye"] * fps > 150.0, (
        f"eye peak {worst['eye'] * fps:.0f} deg/s is too slow to be a saccade")
    assert worst["body"] * fps < 120.0, (
        f"body joint peak {worst['body'] * fps:.0f} deg/s is a pop")


def test_smile_is_coordinated_not_mouth_only():
    """The hard fail: corners move, cheeks dead, eyes identical."""
    from presenter.motion.expression import FacialExpressionSystem
    from presenter.behavior.context import Drives
    from presenter.behavior.randomness import Rng
    from presenter.behavior.state import PROFILES, BehaviorState, StateModulation

    sysx = FacialExpressionSystem(PROFILES["PRESENTER_CALM"])
    rng = Rng(3)
    drives = Drives(rng=rng, profile=PROFILES["PRESENTER_CALM"],
                    state=BehaviorState.MILD_POSITIVE, mod=StateModulation())
    sysx.trigger(drives, "SMALL_SMILE", 1.0)

    best = None
    for i in range(240):
        drives.now = i / 30.0
        drives.dt = 1.0 / 30.0
        m = HumanMotionState()
        sysx.update(drives, m)
        if best is None or m.face.mouth_corner_l > best.face.mouth_corner_l:
            best = m
    f = best.face
    assert f.mouth_corner_l > 0.1
    assert f.cheek_l > 0.4 * abs(f.mouth_corner_l), "cheeks are dead"
    assert f.eye_squint_l > 0.2 * abs(f.mouth_corner_l), "eyes are unchanged"
    assert f.mouth_corner_l != f.mouth_corner_r, "a perfectly symmetric smile"


def test_expression_has_a_reaction_latency():
    """Nothing reacts at frame zero."""
    from presenter.motion.expression import FacialExpressionSystem
    from presenter.behavior.context import Drives
    from presenter.behavior.randomness import Rng
    from presenter.behavior.state import PROFILES, BehaviorState, StateModulation

    sysx = FacialExpressionSystem(PROFILES["PRESENTER_CALM"])
    drives = Drives(rng=Rng(1), profile=PROFILES["PRESENTER_CALM"],
                    state=BehaviorState.MILD_POSITIVE, mod=StateModulation())
    sysx.trigger(drives, "SMALL_SMILE", 1.0)
    m = HumanMotionState()
    drives.now = 0.0
    sysx.update(drives, m)
    assert m.face.mouth_corner_l == 0.0, "reacted on the trigger frame"
    latency = sysx._active.latency
    assert 0.09 <= latency <= 1.4
