"""Seated posture: the neutral pose, the posture continuum, and rare comfort shifts.

## Neutral is not zero

A rig at zero rotation is a T-pose with the arms lowered. It is not a person
sitting, and the difference is almost entirely asymmetry: nobody sits with
their shoulders level, their elbows at matched angles and their head square to
the room. `SEATED_NEUTRAL` therefore has a deliberate, *fixed* imbalance - the
left shoulder a little lower, the elbows unequal, the head a degree off centre.

Fixed is the operative word. The asymmetry belongs to the person and never
re-randomises; a body that is asymmetric differently every second reads as
noise rather than as somebody's habitual way of sitting.

## Posture is a continuum, not four clips

`engagement` runs -1 (settled back into the chair) to +1 (forward, focused).
Every anchor named in the brief is a *region* of that axis, and the body is
interpolated to wherever it currently sits. Transitions run over seconds on a
critically-damped approach, because a torso has mass.

## Comfort shifts are rare

Over five minutes there should be a handful, not a rhythm. Each is a discrete
event drawn from a long-tailed interval and subject to the same
recency-suppression the face events use, so the same adjustment does not recur.
"""

from __future__ import annotations

import math

from ..behavior.randomness import OrnsteinUhlenbeck
from ..types import BehaviorEvent
from .state import HumanMotionState, JointRotation, PostureState

__all__ = ["BodySystem", "SEATED_NEUTRAL"]


# Degrees. The pose a person actually holds in a task chair at a desk:
# pelvis tilted back into the seat, a natural lumbar curve, the chest carried
# slightly open, and the head a shade forward as it always is at a monitor.
SEATED_NEUTRAL: dict[str, tuple[float, float, float]] = {
    "pelvis":      (+4.2, +0.0, +0.5),
    "spine_lower": (+3.4, -0.4, +0.3),
    "spine_mid":   (+1.6, +0.3, -0.4),
    "chest":       (-1.1, -0.2, +0.2),
    # Left clavicle lower than right: the persona's stable shoulder drop.
    "clavicle_l":  (+0.6, +0.0, +1.5),
    "clavicle_r":  (+0.2, +0.0, -0.7),
    "shoulder_l":  (+2.6, -2.2, +1.0),
    "shoulder_r":  (+2.1, +3.1, -0.6),
    "elbow_l":     (+2.0, +0.0, +0.0),
    "elbow_r":     (+2.0, +0.0, +0.0),
    "wrist_l":     (-2.0, +1.0, +0.0),
    "wrist_r":     (-1.5, -1.5, +0.0),
    # Head slightly forward of vertical, a degree off square, barely rolled.
    "neck":        (-2.4, +0.4, -0.3),
    "head":        (+1.4, +0.7, -0.5),
}

# How the body changes from settled-back (-1) to forward-focus (+1), in degrees
# per unit of engagement. Leaning forward is mostly hips and lumbar, not neck:
# a lean driven from the neck is a person craning, which reads as strain.
ENGAGEMENT_COUPLING: dict[str, tuple[float, float, float]] = {
    "pelvis":      (-3.6, 0.0, 0.0),
    "spine_lower": (-2.8, 0.0, 0.0),
    "spine_mid":   (-1.9, 0.0, 0.0),
    "chest":       (-1.2, 0.0, 0.0),
    "clavicle_l":  (-0.5, 0.0, 0.0),
    "clavicle_r":  (-0.5, 0.0, 0.0),
    "neck":        (+1.4, 0.0, 0.0),
    "head":        (+0.6, 0.0, 0.0),
}


class _Shift:
    """One comfort adjustment in flight."""

    __slots__ = ("kind", "start", "duration", "targets", "applied")

    def __init__(self, kind, start, duration, targets):
        self.kind = kind
        self.start = start
        self.duration = duration
        self.targets = targets      # joint -> (rx, ry, rz) peak offset
        self.applied = {}


# Rare adjustments. Amplitudes are small; what makes them read is that they
# persist rather than that they are large.
COMFORT_SHIFTS = {
    "SHOULDER_SETTLE": dict(
        weight=1.4, duration=(1.1, 2.2),
        targets={"clavicle_l": (0.0, 0.0, -1.6), "clavicle_r": (0.0, 0.0, +0.9),
                 "shoulder_l": (+0.8, 0.0, -0.7), "shoulder_r": (+0.5, 0.0, +0.4)},
    ),
    "PELVIS_COMFORT_SHIFT": dict(
        weight=1.0, duration=(1.6, 3.0),
        targets={"pelvis": (-1.2, +1.4, +0.9), "spine_lower": (+0.6, -0.8, -0.5)},
    ),
    "SMALL_LEAN_FORWARD": dict(
        weight=0.9, duration=(2.2, 4.0),
        targets={"pelvis": (-2.4, 0.0, 0.0), "spine_lower": (-1.8, 0.0, 0.0),
                 "spine_mid": (-1.2, 0.0, 0.0), "neck": (+0.9, 0.0, 0.0)},
    ),
    "SETTLE_BACK": dict(
        weight=1.1, duration=(2.6, 4.6),
        targets={"pelvis": (+2.2, 0.0, 0.0), "spine_lower": (+1.7, 0.0, 0.0),
                 "chest": (+0.8, 0.0, 0.0)},
    ),
    "TORSO_ROTATE": dict(
        weight=0.6, duration=(1.8, 3.2),
        targets={"spine_lower": (0.0, +1.5, 0.0), "spine_mid": (0.0, +1.2, 0.0),
                 "chest": (0.0, +0.9, 0.0)},
    ),
}


def _min_jerk(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


class BodySystem:
    """Neutral seated pose, posture continuum, and rare comfort adjustments."""

    def __init__(self, profile, persona=None) -> None:
        self.profile = profile
        activity = float((persona or {}).get("posture_activity", 0.4))
        self.activity = max(activity, 0.05)

        # Engagement drifts slowly on its own, and is nudged by state.
        self._engagement = 0.0
        self._engagement_target = 0.0
        self._drift = OrnsteinUhlenbeck.from_amplitude(0.22, 70.0)

        self._active: list[_Shift] = []
        self._held: dict[str, list[float]] = {}
        self._next_at = None
        self._recent: list[str] = []
        self.shift_count = 0

    # -- scheduling ----------------------------------------------------------
    def _schedule(self, drives) -> None:
        median = 74.0 / self.activity
        self._next_at = drives.now + drives.rng.lognormal_interval(
            median=median, shape=0.62, low=18.0, high=median * 5.0)

    def _choose(self, drives) -> str:
        weights = {}
        for name, spec in COMFORT_SHIFTS.items():
            w = spec["weight"]
            w *= 0.35 ** self._recent.count(name)
            # A lean and a settle are opposites; do not settle back when
            # already back, or lean forward when already forward.
            if name == "SMALL_LEAN_FORWARD" and self._engagement > 0.35:
                w *= 0.15
            if name == "SETTLE_BACK" and self._engagement < -0.35:
                w *= 0.15
            weights[name] = max(w, 1e-4)
        total = sum(weights.values())
        r = drives.rng.uniform(0.0, total)
        acc = 0.0
        for name, w in weights.items():
            acc += w
            if r <= acc:
                return name
        return "SHOULDER_SETTLE"

    def _begin(self, drives, kind: str) -> None:
        spec = COMFORT_SHIFTS[kind]
        lo, hi = spec["duration"]
        dur = drives.rng.uniform(lo, hi)

        # Sign and scale vary per instance, so the same adjustment is never
        # literally the same movement twice.
        sign = 1.0 if drives.rng.chance(0.5) else -1.0
        scale = drives.rng.uniform(0.65, 1.35)
        targets = {}
        for joint, (rx, ry, rz) in spec["targets"].items():
            s = scale * (sign if joint in ("pelvis", "spine_lower", "spine_mid",
                                           "chest") and kind == "TORSO_ROTATE"
                         else 1.0)
            targets[joint] = (rx * scale, ry * s, rz * scale)

        self._active.append(_Shift(kind, drives.now, dur, targets))
        self.shift_count += 1
        self._recent.append(kind)
        if len(self._recent) > 4:
            self._recent.pop(0)

        if kind == "SMALL_LEAN_FORWARD":
            self._engagement_target = min(self._engagement_target + 0.55, 1.0)
        elif kind == "SETTLE_BACK":
            self._engagement_target = max(self._engagement_target - 0.55, -1.0)

        drives.emit(BehaviorEvent(
            time=drives.now, kind="posture_shift", detail=kind,
            magnitude=dur, metadata={"kind": kind, "duration": dur}))

    # -- per frame -----------------------------------------------------------
    def update(self, drives, motion: HumanMotionState) -> None:
        if self._next_at is None:
            self._schedule(drives)
        elif drives.now >= self._next_at:
            if drives.allow_voluntary():
                self._begin(drives, self._choose(drives))
                self._schedule(drives)
            else:
                self._next_at = drives.now + 3.0

        # Engagement: state pulls it, a slow process wanders it, and it
        # approaches its target over seconds because a torso has mass.
        pull = getattr(drives.mod, "engagement_bias", None)
        if pull is None:
            pull = {"FOCUSED": 0.55, "READING": 0.35, "THINKING": -0.25,
                    "IDLE_RELAXED": -0.45, "IDLE_ATTENTIVE": 0.05}.get(
                        drives.state.value, 0.0)
        self._engagement_target += (pull - self._engagement_target) * min(
            drives.dt / 6.0, 1.0)
        wander = self._drift.step(drives.dt, drives.rng)
        want = max(min(self._engagement_target + wander, 1.0), -1.0)
        self._engagement += (want - self._engagement) * (
            1.0 - math.exp(-drives.dt / 3.2))

        joints = motion.joints()

        # 1. Neutral seated pose.
        for name, (rx, ry, rz) in SEATED_NEUTRAL.items():
            j = joints.get(name)
            if j is not None:
                j.rx += rx
                j.ry += ry
                j.rz += rz

        # 2. Posture continuum.
        for name, (rx, ry, rz) in ENGAGEMENT_COUPLING.items():
            j = joints.get(name)
            if j is not None:
                j.rx += rx * self._engagement
                j.ry += ry * self._engagement
                j.rz += rz * self._engagement

        # 3. Comfort shifts in flight. Each eases in over its duration and
        #    then *stays* - a settle that springs back was not a settle.
        still_active = []
        for sh in self._active:
            t = (drives.now - sh.start) / max(sh.duration, 1e-4)
            k = _min_jerk(t)
            for joint, (rx, ry, rz) in sh.targets.items():
                prev = sh.applied.get(joint, (0.0, 0.0, 0.0))
                now_v = (rx * k, ry * k, rz * k)
                held = self._held.setdefault(joint, [0.0, 0.0, 0.0])
                for i in range(3):
                    held[i] += now_v[i] - prev[i]
                sh.applied[joint] = now_v
            if t < 1.0:
                still_active.append(sh)
        self._active = still_active

        # 4. Held offsets decay very slowly back toward the neutral pose, so
        #    adjustments accumulate for minutes rather than forever.
        decay = math.exp(-drives.dt / 240.0)
        for joint, held in self._held.items():
            j = joints.get(joint)
            for i in range(3):
                held[i] *= decay
            if j is not None:
                j.rx += held[0]
                j.ry += held[1]
                j.rz += held[2]

        motion.posture = PostureState(
            engagement=self._engagement,
            lean=-ENGAGEMENT_COUPLING["pelvis"][0] * self._engagement,
            settle=max(-self._engagement, 0.0),
            shoulder_drop_l=SEATED_NEUTRAL["clavicle_l"][2],
            shoulder_drop_r=SEATED_NEUTRAL["clavicle_r"][2],
        )

    @property
    def engagement(self) -> float:
        return self._engagement
