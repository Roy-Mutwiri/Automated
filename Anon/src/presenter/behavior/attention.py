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

# What the eye can actually reach in its orbit, in degrees. Asymmetric
# vertically: the eye travels further down than up.
OCULAR_MAX_AZ = 41.0
OCULAR_MAX_UP = 27.0
OCULAR_MAX_DOWN = 39.0


def _hold_eye(az: float, el: float) -> tuple[float, float, float, float]:
    """Clamp an eye residual to the orbit, returning the overflow.

    During a large target change the eyes lead and the head lags, so the
    residual the eye is asked for can briefly exceed its range - going from a
    keyboard 29 degrees down to a thinking point 14 up asks the eye for 43
    degrees of upward travel before the neck has moved.

    The eye simply cannot do that, so it is clamped and the gaze is briefly
    short of its target until the head arrives. That is what happens to a
    person, too. Handing the excess straight to the head was tried and is
    wrong: it teleports the neck, which the frame-jump test caught immediately.
    """
    caz = min(max(az, -OCULAR_MAX_AZ), OCULAR_MAX_AZ)
    cel = min(max(el, -OCULAR_MAX_DOWN), OCULAR_MAX_UP)
    return caz, cel, az - caz, el - cel
GAZE_UNITS_PER_DEG = 0.42 / OCULAR_LIMIT_DEG

# How much a target's appeal decays per recent visit, and the extra penalty for
# being the target two shifts ago. The second discourages A-B-A cycles.
#
# Both were briefly set far harder (0.38 / 0.30) to fight an apparent 44x
# repetition that turned out to be an instrumentation bug. They are back to
# gentle values on purpose: pushed hard enough, an anti-repetition rule stops
# being a randomiser and becomes a deterministic least-recently-used
# round-robin, which is a more obvious loop than the one it removes. That is
# not hypothetical - it happened, and produced a locked five-target cycle.
RECENCY_DECAY = 0.62
RETURN_PENALTY = 0.55


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
    AttentionTarget("CAMERA_LENS", 0.0, 1.0, dwell_median=4.6, weight=2.4, spread=1.0),
    AttentionTarget("MAIN_MONITOR_CENTER", -4.0, -12.0, dwell_median=6.5, weight=2.3, spread=3.0),
    AttentionTarget("MAIN_MONITOR_LOWER", -5.0, -20.0, dwell_median=4.0, weight=1.2, spread=3.2),
    AttentionTarget("SECONDARY_MONITOR", -32.0, -7.0, dwell_median=4.2, weight=1.0, spread=4.0),
    AttentionTarget("CHAT_REGION", 26.0, -11.0, dwell_median=3.4, weight=1.0, spread=3.6),
    AttentionTarget("DESK_MOUSE", -15.0, -30.0, dwell_median=1.9, weight=0.6, spread=3.0),
    AttentionTarget("KEYBOARD_REGION", -3.0, -34.0, dwell_median=1.7, weight=0.4, spread=3.4),
    AttentionTarget("DESK_GENERAL", 9.0, -28.0, dwell_median=2.0, weight=0.4, spread=4.0),
    AttentionTarget("THINKING_POINT_LEFT", -25.0, 14.0, dwell_median=2.2, weight=0.35, spread=4.5),
    AttentionTarget("THINKING_POINT_RIGHT", 23.0, 16.0, dwell_median=2.2, weight=0.3, spread=4.5),
)

# The target layout is the single thing that decides whether this reads as a
# person or a photograph, and the previous one was the reason it read as a
# photograph.
#
# It had the lens and the main display **5.5 degrees apart**, on the reasoning
# that a webcam sits on top of a monitor. That is true and it was the wrong
# conclusion: those two targets carried 75% of all dwell time, 5.5 degrees is
# below the threshold that recruits any head movement, so three quarters of the
# time the presenter was looking at what a viewer perceives as one single point
# with a frozen head. Measured mean head yaw over five minutes: 0.91 degrees.
#
# A monitor is not a point. Its centre sits about 12 degrees below the lens and
# its lower third about 20; a second display is 30-plus degrees off axis; a
# mouse is 30 degrees *down*. Spreading the targets to where they physically are
# is what makes a glance visible, and it costs nothing.
#
# Vertical spread matters as much as horizontal and was almost absent before.
# People look down at their hands.

# Dwell times were raised 1.45x from a first pass that felt reasonable and
# measured badly. With the original figures the stillness audit failed: median
# gap between visible movements 2.57 s against a 3.00 s standard and 43% of gaps
# over three seconds against 50%. The presenter was not doing anything wrong,
# he was simply doing it too often - and "too often" is the failure mode the
# brief cares about most, because the commonest valid human action is nothing.
#
# A nine-second median on the lens sounds long written down. It is not: a silent
# presenter holding the camera for nine seconds reads as composure, and the
# log-normal spread means individual holds run from about three seconds to
# nearly a minute.

# State -> multiplicative bias on each target's selection weight. This is how a
# state means something: READING is not "a label", it is "the main display is
# 9x more interesting than usual and the lens is half as interesting".
# Eye-head recruitment.
#
# Loosened from 8 / 35 / 0.75 after watching the output. The old thresholds were
# tuned against a target layout whose two dominant targets were 5.5 degrees
# apart, so nothing ever crossed them: measured mean head yaw was 0.91 degrees
# over five minutes, which is a frozen head.
#
# A person turns their head. Being conservative here does not produce subtlety,
# it produces a photograph.
EYE_ONLY_DEG = 6.0            # below this, eyes alone
HEAD_SHARE_FULL_DEG = 30.0    # eccentricity at which the head takes its full share
HEAD_SHARE_MAX = 0.85         # the eyes always keep some eccentricity
HEAD_SHARE_CURVE = 0.65       # <1 front-loads the near range

# Vertical share. A dropped chin is far more readable than the same angle of
# yaw, and people really do tilt their head down to read a screen or look at
# their hands rather than rolling their eyes down.
PITCH_SHARE = 1.05


STATE_BIAS: dict[str, dict[str, float]] = {
    "READING":        {"MAIN_MONITOR_CENTER": 7.0, "MAIN_MONITOR_LOWER": 5.0,
                       "CAMERA_LENS": 0.30, "THINKING_POINT_LEFT": 0.2,
                       "THINKING_POINT_RIGHT": 0.2},
    "CHECKING_CHAT":  {"CHAT_REGION": 9.0, "CAMERA_LENS": 0.7,
                       "MAIN_MONITOR_CENTER": 0.5},
    "FOCUSED":        {"MAIN_MONITOR_CENTER": 4.0, "MAIN_MONITOR_LOWER": 2.5,
                       "DESK_MOUSE": 1.8, "CAMERA_LENS": 0.5},
    "THINKING":       {"THINKING_POINT_LEFT": 5.0, "THINKING_POINT_RIGHT": 4.0,
                       "CAMERA_LENS": 0.3, "MAIN_MONITOR_CENTER": 0.4},
    "WAITING":        {"CAMERA_LENS": 1.8, "DESK_GENERAL": 1.6,
                       "SECONDARY_MONITOR": 1.4, "MAIN_MONITOR_CENTER": 0.7},
    "IDLE_RELAXED":   {"CAMERA_LENS": 1.1, "DESK_GENERAL": 1.5,
                       "SECONDARY_MONITOR": 1.4},
    "IDLE_ATTENTIVE": {"CAMERA_LENS": 1.5, "MAIN_MONITOR_CENTER": 1.2},
    "LISTENING":      {"CAMERA_LENS": 2.4, "MAIN_MONITOR_CENTER": 0.6},
    "SPEAKING":       {"CAMERA_LENS": 2.2, "MAIN_MONITOR_CENTER": 0.7},
}


# How much more (or less) head a state recruits at the same angle. Reading and
# focusing turn the head; idle glancing does not.
STATE_HEAD_RECRUITMENT = {
    "READING": 1.45,
    "CHECKING_CHAT": 1.30,
    "WAITING": 1.05,
    "FOCUSED": 1.25,
    "LISTENING": 1.15,
    "IDLE_ATTENTIVE": 1.0,
    "SPEAKING": 0.95,
    "IDLE_RELAXED": 0.85,
    "THINKING": 0.70,
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

        self.current = "CAMERA_LENS"
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
        self._planned_dwell = 3.0
        self._requested: tuple[str, float | None] | None = None
        self._state_at_shift = "IDLE_ATTENTIVE"
        self.shift_count = 0
        self.torso_yaw = 0.0

    # -- selection ----------------------------------------------------------
    def _weights(self, drives) -> dict[str, float]:
        bias = STATE_BIAS.get(drives.state.value, {})
        affinity = drives.mod.camera_affinity
        out: dict[str, float] = {}
        for name, t in self.targets.items():
            w = t.weight * bias.get(name, 1.0)
            if name == "CAMERA_LENS":
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
        return "CAMERA_LENS"

    # -- the shift ----------------------------------------------------------
    def _hold_share(self, az: float, dwell: float = 3.0,
                    state: str = "IDLE_ATTENTIVE", el: float = 0.0) -> float:
        """How much of a held eccentricity the head carries.

        Not `head = eye_angle * k`. Human recruitment depends on more than
        geometry, and the review was right that a fixed ratio produced
        side-eye: a 19 degree glance recruited 3 degrees of head, so the eyes
        carried it alone and the presenter looked shifty rather than
        interested.

        Three inputs:

        **Eccentricity.** Below `eye_only_deg` the eyes do it alone. Above, the
        head's share grows and is capped below 1 - the eyes always keep some
        eccentricity at the end of a turn.

        **How long he means to look.** This is the one that fixes the side-eye,
        and it is the difference between a glance and an interest. Nobody turns
        their head to check something for half a second; everybody turns it to
        read for five. Scaled by the target's own dwell, so a quick chat check
        stays eye-heavy and a sustained second-display read brings the head
        round properly.

        **What he is doing.** Reading and focusing recruit more head than idle
        glancing at the same angle, because the head follows what matters.

        Defined on the target's own angle rather than on the size of the shift
        that reached it, which is what makes it self-correcting: the lens is a
        zero-share target, so looking back at the audience brings the head home
        with no separate recentring rule.
        """
        p = self.profile
        eye_only = getattr(p, "eye_only_deg", EYE_ONLY_DEG)
        # Full eccentricity, not azimuth alone. Using azimuth by itself meant
        # the desk - 6 degrees across but 17 down - recruited no head at all,
        # so the presenter looked at his own hands with his eyes alone. People
        # drop their chin to look down; the vertical component has to count.
        mag = math.hypot(az, el)
        if mag <= eye_only:
            return 0.0

        span = max(getattr(p, "head_share_full_deg", HEAD_SHARE_FULL_DEG)
                   - eye_only, 1e-3)
        # Concave, not linear. A linear ramp gives a 14 degree look - which is
        # what reading your own monitor is - only 12% of the head's share, so
        # the most frequent glance in the whole performance stayed invisible.
        # The exponent front-loads the curve so near targets still turn the
        # head a little, without changing what a large turn does.
        share = min((mag - eye_only) / span, 1.0) ** HEAD_SHARE_CURVE
        share *= getattr(p, "head_share_max", HEAD_SHARE_MAX)

        # Intent. A 0.6 s glance keeps ~55% of the geometric share; a 5 s read
        # gets ~1.35x it. The curve is gentle either side of a ~2.5 s pivot.
        intent = 0.55 + 0.80 * (1.0 - math.exp(-max(dwell, 0.05) / 2.5))
        share *= intent

        share *= STATE_HEAD_RECRUITMENT.get(state, 1.0)
        share *= getattr(p, "head_motion_level", 1.0)
        return min(share, 0.88)

    def _torso_share(self, az: float) -> float:
        """Upper torso participation, for large turns only.

        Below about 28 degrees the neck does the whole job and the chest should
        not move at all; a torso that rotates for every glance reads as
        swivelling.
        """
        mag = abs(az)
        if mag <= 28.0:
            return 0.0
        return min((mag - 28.0) / 40.0, 1.0) * 0.35

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

        planned = t.dwell_median
        share = self._hold_share(to_az, planned, drives.state.value, to_el)
        head_to_yaw = to_az * share
        head_to_pitch = to_el * share * PITCH_SHARE
        self._planned_dwell = planned
        self._state_at_shift = drives.state.value

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
        if len(self._recent) > 4:
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

    # -- external control ----------------------------------------------------
    def request(self, name: str, dwell: float | None = None) -> None:
        """Ask him to look at something.

        The seam the content pipeline will drive: `human.set_attention(...)`
        eventually lands here. It expresses an *intention*, not a pose - the
        shift still goes through the same eye-head division, the same
        minimum-jerk trajectory and the same latency as a self-directed one, so
        a scripted look is indistinguishable in kind from a spontaneous one.
        """
        if name not in self.targets:
            raise ValueError(f"unknown attention target {name!r}")
        self._requested = (name, dwell)

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
            res_az, res_el, _, _ = _hold_eye(
                s.to_az - self.head_yaw, s.to_el - self.head_pitch)
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
        # The head goes on following after the eyes have arrived, and the
        # share itself ramps in over roughly two seconds. That ramp is what
        # produces the sequence the brief describes: the eyes reach the target
        # first, the head catches up over the next second or two, and the eyes
        # *recenter in their sockets* as it does - because the eye angle here
        # is always the residual `target - head`, so it shrinks as the head
        # grows. Nothing has to animate the recentring; it falls out.
        on_target = max(drives.now - self.last_change, 0.0)
        ramp = 1.0 - math.exp(-on_target / 1.8)
        share = self._hold_share(self.point_az, self._planned_dwell,
                                 self._state_at_shift, self.point_el) * ramp
        want_yaw = self.point_az * share
        want_pitch = self.point_el * share * PITCH_SHARE
        k = 1.0 - math.exp(-drives.dt / 1.4)
        self.head_yaw += (want_yaw - self.head_yaw) * k
        self.head_pitch += (want_pitch - self.head_pitch) * k
        want_torso = self.point_az * self._torso_share(self.point_az)
        self.torso_yaw += (want_torso - self.torso_yaw) * (
            1.0 - math.exp(-drives.dt / 2.6))

        self.eye_az, self.eye_el, _, _ = _hold_eye(
            self.point_az - self.head_yaw, self.point_el - self.head_pitch)

        if self._requested is not None:
            name, dwell = self._requested
            self._requested = None
            self._begin_shift(drives, name)
            if dwell is not None:
                self.dwell_until = drives.now + dwell
        elif drives.now >= self.dwell_until and drives.allow_voluntary():
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
        return self.current == "CAMERA_LENS"

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
            "MAIN_MONITOR_CENTER": 0.85,
            "MAIN_MONITOR_LOWER": 0.88,
            "SECONDARY_MONITOR": 0.75,
            "CHAT_REGION": 0.72,
            "KEYBOARD_REGION": 0.5,
            "DESK_MOUSE": 0.45,
            "DESK_GENERAL": 0.35,
            "CAMERA_LENS": 0.3,
            "THINKING_POINT_LEFT": 0.0,
            "THINKING_POINT_RIGHT": 0.0,
        }.get(self.current, 0.3)
