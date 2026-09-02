"""Where the presenter is looking, and why.

The subsystem this replaces chose gaze targets as *displacements*: pick a random
angle, pick an amplitude, move the eyes there. That produces statistically
plausible eye movement attached to nothing. Asked "why did he look left?", the
only available answer was "the sampler said so", which is exactly the failure
the brief names - movement without a reason.

Here the presenter looks at **things**, and the things are fixed in the room.

## World space, and why it is not optional

Every target is stored as a direction in the *room*: an azimuth and elevation
relative to the subject's neutral forward. Not as a screen offset, not relative
to any camera.

This is what makes the hard requirement hold structurally rather than by
convention. The director may cut from camera 1 to camera 3 mid-fixation; the
presenter does not know which camera is live and nothing in this file can see
one. He goes on looking at the same physical object, and the new camera simply
observes that from a different angle. There is no code path by which a camera
switch could alter the gaze, because the camera is not an input.

## The geometry is a streamer's, not a conversation's

Dyadic social gaze models assume a face to look at. A streamer's attention is
divided between a lens, one or more displays, a desk, and the middle distance -
and crucially the lens and the main display are only a few degrees apart,
because the camera sits on top of the monitor. That near-coincidence is why a
streamer can appear to hold eye contact while actually reading: the two targets
are close enough that the difference is a few degrees of iris, which is
precisely the detail that makes the behaviour read as real.

## Eye-head coordination

A gaze shift is not "move the eyes" or "move the head" - it is one act divided
between them, and the division is a function of amplitude:

* below `eye_only_deg`, the eyes alone. Most shifts are this.
* above it, the head takes a share that grows with amplitude, and **the eyes
  lead**. The eyes arrive in tens of milliseconds; the head follows over a few
  hundred. Moving them together is one of the most reliable tells of a rig.
* the eyes counter-roll as the head arrives, so the combined gaze stays on
  target rather than overshooting - which is what the vestibulo-ocular reflex
  does and what makes the pair look like one movement instead of two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["AttentionTarget", "AttentionSystem", "DEFAULT_TARGETS"]

# Degrees of eccentricity the eyes are driven to at `saccade_max_amplitude`
# (0.42 in profile units). Measured against the expression calibration: 0.42
# puts the irises at their natural lateral limit on this face.
OCULAR_LIMIT_DEG = 35.0
GAZE_UNITS_PER_DEG = 0.42 / OCULAR_LIMIT_DEG

# How much a target's appeal decays per recent visit, and the extra penalty for
# being the target two shifts ago. The second is what breaks A-B-A cycles.
RECENCY_DECAY = 0.38
RETURN_PENALTY = 0.30


@dataclass(frozen=True)
class AttentionTarget:
    """Something in the room worth looking at.

    Angles are degrees from the subject's neutral forward. `+azimuth` is toward
    image right, matching the sign convention of `AvatarPose.yaw` and
    `gaze_x`; `+elevation` is up.
    """

    name: str
    azimuth: float
    elevation: float
    # Dwell is drawn log-normally around this median. Fixations have a long
    # right tail - occasionally you just keep looking at something.
    dwell_median: float
    dwell_shape: float = 0.55
    weight: float = 1.0
    # Radius of the region, in degrees. Re-fixations land somewhere inside the
    # target, never on the identical point twice.
    spread: float = 1.2


# The camera sits above the main display, which is why LENS and MAIN_DISPLAY are
# only a few degrees apart. SECOND_DISPLAY and CHAT are the off-axis glances
# that make a streamer look like they are working rather than performing.
DEFAULT_TARGETS: tuple[AttentionTarget, ...] = (
    AttentionTarget("LENS", 0.0, 0.6, dwell_median=6.4, weight=3.4, spread=0.9),
    AttentionTarget("MAIN_DISPLAY", -2.0, -4.5, dwell_median=4.0, weight=2.2, spread=2.2),
    AttentionTarget("SECOND_DISPLAY", -21.0, -3.5, dwell_median=1.9, weight=0.75, spread=3.0),
    AttentionTarget("CHAT", 17.5, -5.5, dwell_median=2.3, weight=0.7, spread=2.6),
    AttentionTarget("DESK", -6.0, -17.0, dwell_median=1.4, weight=0.5, spread=3.2),
    AttentionTarget("MIDDLE_DISTANCE", 10.0, 8.0, dwell_median=2.6, weight=0.5, spread=4.0),
)

# State -> multiplicative bias on each target's selection weight. This is how a
# state means something: READING is not "a label", it is "the main display is
# 9x more interesting than usual and the lens is half as interesting".
STATE_BIAS: dict[str, dict[str, float]] = {
    "READING":        {"MAIN_DISPLAY": 9.0, "LENS": 0.45, "MIDDLE_DISTANCE": 0.2},
    "FOCUSED":        {"MAIN_DISPLAY": 4.0, "LENS": 0.8, "DESK": 1.6},
    "THINKING":       {"MIDDLE_DISTANCE": 6.0, "LENS": 0.35, "MAIN_DISPLAY": 0.6},
    "IDLE_RELAXED":   {"LENS": 0.8, "MIDDLE_DISTANCE": 1.8, "SECOND_DISPLAY": 1.4},
    "IDLE_ATTENTIVE": {"LENS": 1.4},
    "LISTENING":      {"LENS": 2.2, "MIDDLE_DISTANCE": 0.5},
    "SPEAKING":       {"LENS": 2.0, "MAIN_DISPLAY": 0.7},
}


@dataclass
class _Shift:
    """A gaze shift in flight, divided between eyes and head."""

    start: float
    eye_duration: float
    head_duration: float
    head_delay: float
    from_az: float
    from_el: float
    to_az: float
    to_el: float
    head_from_yaw: float
    head_from_pitch: float
    head_to_yaw: float
    head_to_pitch: float


def _min_jerk(t: float) -> float:
    """Flash & Hogan minimum-jerk profile. Zero velocity and acceleration at
    both ends, which is what stops a movement looking keyframed."""
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


class AttentionSystem:
    """Chooses what to look at, and divides each shift between eyes and head."""

    def __init__(self, profile, targets=DEFAULT_TARGETS) -> None:
        self.targets = {t.name: t for t in targets}
        self.profile = profile

        self.current = "LENS"
        self.point_az = 0.0
        self.point_el = 0.6
        self.dwell_until = 0.0
        self.last_change = 0.0

        # Where the eyes and head actually are, in degrees.
        self.eye_az = 0.0
        self.eye_el = 0.6
        self.head_yaw = 0.0
        self.head_pitch = 0.0

        self._shift: _Shift | None = None
        self._recent: list[str] = []
        self.shift_count = 0

    # -- selection ----------------------------------------------------------
    def _weights(self, drives) -> dict[str, float]:
        bias = STATE_BIAS.get(drives.state.value, {})
        affinity = drives.mod.camera_affinity
        out: dict[str, float] = {}
        for name, t in self.targets.items():
            w = t.weight * bias.get(name, 1.0)
            if name == "LENS":
                # camera_affinity is the state's pull toward the audience.
                w *= 1.0 + 1.6 * affinity
            # Behavioural memory, and specifically an anti-alternation term.
            #
            # A plain recency decay is not enough, and the repetition detector
            # proved it: with LENS and MAIN_DISPLAY carrying the two largest
            # weights, the chain LENS -> MAIN_DISPLAY -> LENS appeared 71 times
            # in a thirty-minute run against 1.6 expected from the marginals -
            # a 44x excess, and exactly the two-target cycle the brief names as
            # the most obvious loop an idle avatar can produce.
            #
            # The specific pattern is A -> B -> A, so the specific fix is to
            # penalise the target visited *two steps ago* on top of general
            # recency. Some going back and forth between lens and monitor is
            # correct - it is what a streamer does - so this discourages the
            # cycle rather than forbidding it.
            recency = self._recent.count(name)
            w *= RECENCY_DECAY ** recency
            if len(self._recent) >= 2 and name == self._recent[-2]:
                w *= RETURN_PENALTY
            if name == self.current:
                w *= 0.08
            out[name] = max(w, 1e-4)
        return out

    def _choose(self, drives) -> str:
        w = self._weights(drives)
        total = sum(w.values())
        r = drives.rng.uniform(0.0, total)
        acc = 0.0
        for name, weight in w.items():
            acc += weight
            if r <= acc:
                return name
        return "LENS"

    # -- the shift ----------------------------------------------------------
    def _hold_share(self, az: float) -> float:
        """How much of a *held* eccentricity the head carries.

        Defined on the target's own angle rather than on the size of the shift
        that reached it, and that difference is the whole point. An earlier
        version gave the head a share of each shift and then left it there: with
        the gaze already on target no correction was ever computed, so the head
        never came back and yaw ratcheted to 11.5 degrees over a minute of
        ordinary glancing. Anchoring the head to where he is *looking* makes it
        self-correcting - look back at the lens and the hold share is zero, so
        the head returns on its own with no separate recentring rule.
        """
        p = self.profile
        eye_only = getattr(p, "eye_only_deg", 11.0)
        mag = abs(az)
        if mag <= eye_only:
            return 0.0
        span = max(getattr(p, "head_share_full_deg", 42.0) - eye_only, 1e-3)
        share = min((mag - eye_only) / span, 1.0)
        share *= getattr(p, "head_share_max", 0.62)
        return share * getattr(p, "head_motion_level", 1.0)

    def _begin_shift(self, drives, name: str) -> None:
        t = self.targets[name]
        rng = drives.rng

        # Land somewhere inside the target, not on its centre. Re-fixating on
        # the same object twice at the identical angle is a machine's habit.
        to_az = t.azimuth + rng.gauss(0.0, t.spread * 0.5)
        to_el = t.elevation + rng.gauss(0.0, t.spread * 0.35)

        d_az = to_az - (self.eye_az + self.head_yaw)
        d_el = to_el - (self.eye_el + self.head_pitch)
        magnitude = math.hypot(d_az, d_el)

        share = self._hold_share(to_az)
        head_to_yaw = to_az * share
        head_to_pitch = to_el * share * 0.6

        # Saccade main sequence: duration rises with amplitude, ~20 ms for the
        # smallest to >100 ms for the largest. The eyes are ballistic, so this
        # is short regardless.
        eye_duration = 0.021 + 0.0023 * magnitude
        # ...but a movement that finishes inside one frame reads as a teleport
        # rather than a movement, so it is stretched to stay sampled. The same
        # accommodation the blink system needs, for the same reason.
        eye_duration = max(eye_duration, 2.2 * drives.frame_interval)

        head_duration = 0.0
        head_delay = 0.0
        if abs(head_to_yaw - self.head_yaw) > 0.15:
            # The head is heavy. It starts after the eyes and takes far longer.
            head_delay = rng.uniform(0.02, 0.06)
            head_duration = max(0.26 + 0.010 * magnitude + rng.gauss(0.0, 0.04), 0.18)

        self._shift = _Shift(
            start=drives.now,
            eye_duration=eye_duration,
            head_duration=head_duration,
            head_delay=head_delay,
            from_az=self.eye_az, from_el=self.eye_el,
            to_az=to_az, to_el=to_el,
            head_from_yaw=self.head_yaw, head_from_pitch=self.head_pitch,
            head_to_yaw=head_to_yaw, head_to_pitch=head_to_pitch,
        )

        self.current = name
        self.point_az, self.point_el = to_az, to_el
        self.last_change = drives.now
        self.shift_count += 1

        self._recent.append(name)
        if len(self._recent) > 6:
            self._recent.pop(0)

        dwell = drives.rng.lognormal_interval(
            median=t.dwell_median, shape=t.dwell_shape,
            low=0.35, high=t.dwell_median * 6.0,
        )
        # Arousal shortens dwell: a livelier presenter's attention moves on
        # sooner. Applied to the dwell rather than to a fixed rate so the
        # distribution keeps its shape.
        dwell /= max(1.0 + 0.35 * drives.arousal, 0.4)
        self.dwell_until = drives.now + dwell

        from ..types import BehaviorEvent
        drives.emit(BehaviorEvent(
            time=drives.now,
            kind="attention",
            detail=f"{name} az={to_az:+.1f} el={to_el:+.1f} share={share:.2f}",
            magnitude=magnitude,
            metadata={"target": name, "azimuth": to_az, "elevation": to_el,
                      "magnitude": magnitude, "head_share": share,
                      "dwell": dwell},
        ))

    # -- per frame ----------------------------------------------------------
    def update(self, drives) -> None:
        if self._shift is not None:
            s = self._shift
            te = (drives.now - s.start) / max(s.eye_duration, 1e-4)

            if s.head_duration > 0.0:
                th = (drives.now - s.start - s.head_delay) / s.head_duration
                if th > 0.0:
                    k = _min_jerk(th)
                    self.head_yaw = s.head_from_yaw + (s.head_to_yaw - s.head_from_yaw) * k
                    self.head_pitch = s.head_from_pitch + (s.head_to_pitch - s.head_from_pitch) * k

            # The eye target is the *residual* after whatever the head has
            # covered so far, recomputed every frame. That is the counter-roll:
            # as the head continues toward the target the eyes ease back toward
            # centre, and the combined gaze stays put instead of overshooting.
            res_az = s.to_az - self.head_yaw
            res_el = s.to_el - self.head_pitch
            k = _min_jerk(te)
            self.eye_az = s.from_az + (res_az - s.from_az) * k
            self.eye_el = s.from_el + (res_el - s.from_el) * k

            done_eye = te >= 1.0
            done_head = s.head_duration <= 0.0 or                 (drives.now - s.start - s.head_delay) >= s.head_duration
            if done_eye and done_head:
                self._shift = None
            return

        # Holding. The head eases toward its share of the current target and
        # the eyes take whatever is left, so gaze stays locked while the neck
        # settles.
        share = self._hold_share(self.point_az)
        want_yaw = self.point_az * share
        want_pitch = self.point_el * share * 0.6
        k = 1.0 - math.exp(-drives.dt / 1.4)
        self.head_yaw += (want_yaw - self.head_yaw) * k
        self.head_pitch += (want_pitch - self.head_pitch) * k
        self.eye_az = self.point_az - self.head_yaw
        self.eye_el = self.point_el - self.head_pitch

        if drives.now >= self.dwell_until and drives.allow_voluntary():
            self._begin_shift(drives, self._choose(drives))

    # -- outputs ------------------------------------------------------------
    @property
    def gaze_x(self) -> float:
        return self.eye_az * GAZE_UNITS_PER_DEG

    @property
    def gaze_y(self) -> float:
        return self.eye_el * GAZE_UNITS_PER_DEG

    @property
    def is_shifting(self) -> bool:
        return self._shift is not None

    @property
    def on_lens(self) -> bool:
        return self.current == "LENS"

    @property
    def visual_demand(self) -> float:
        """0..1, how much the current target taxes vision.

        Drives blink suppression. Reading a display suppresses blinking by a
        factor of three in the literature; looking into the middle distance
        does the opposite. This is the coupling that makes blink rate a
        consequence of what the eyes are doing rather than a personality
        constant.
        """
        return {
            "MAIN_DISPLAY": 0.85,
            "SECOND_DISPLAY": 0.75,
            "CHAT": 0.7,
            "DESK": 0.45,
            "LENS": 0.3,
            "MIDDLE_DISTANCE": 0.0,
        }.get(self.current, 0.3)
