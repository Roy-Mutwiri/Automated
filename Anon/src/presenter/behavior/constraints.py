"""Anatomical limits. The last thing to touch a pose, and it always wins.

The brief's fifth principle is that physics and contact override generated
motion. This module is where that is enforced for the channels the current rig
actually has - head orientation and eyelids - and it is deliberately the final
stage of the engine, after every contributor has had its say.

## Two different limits were being conflated

`MotionProfile.head_max_yaw` is 4.0 degrees on the focused persona and 8.5 on
the energetic one. Those are not necks. A neck yaws past 60 degrees; what those
numbers describe is **how much this particular person moves his head while
sitting still**, which is a question of personality, not anatomy.

While the head system was the only thing writing to yaw the distinction did not
matter, and the test suite asserted the combined pose against the stylistic
figure. Then attention started turning the head for a reason - to look at a
display 21 degrees off axis - and the assertion failed at 5.6 degrees against a
5.4 limit. The failure was correct and the limit was wrong: that head turn is
not a fidget that has exceeded its budget, it is a man looking at his monitor.

So they are separated here:

* **stylistic** (`head_max_*` in the profile) bounds the *idle* contribution -
  sway, micro-corrections, voluntary head moves with no attentional cause.
  Enforced inside the head system.
* **anatomical** (this module) bounds the *total*, including gaze-driven turns.
  Enforced last, on everything.

## Soft saturation rather than a hard clip

A hard clamp produces a head that travels normally and then stops dead against
an invisible wall, which reads worse than the overshoot it prevents. `L *
tanh(x / L)` is linear for small values, compresses gracefully as it approaches
the limit and never exceeds it. The presenter can strain toward a limit; he
cannot pass it, and he cannot hit it with a bang.
"""

from __future__ import annotations

import math

__all__ = ["AnatomicalLimits", "SEATED_PRESENTER", "apply", "SEATED_JOINT_LIMITS",
           "apply_body"]

# Plausible ranges for a *seated presenter at a desk*, in degrees, as
# (min, max) per axis. Not full anatomical range: a spine can flex far further
# than this, but a man in a task chair does not, and a limit set to what the
# body could theoretically do would never catch anything.
#
# These exist so that a generated or learned motion source, when one arrives,
# has something to be clamped by. Principle 5: generated motion is a proposal.
SEATED_JOINT_LIMITS: dict[str, tuple[tuple[float, float], ...]] = {
    #             rx (fwd/back)     ry (turn)       rz (side bend)
    "pelvis":      ((-12.0, 14.0), (-9.0, 9.0),   (-7.0, 7.0)),
    "spine_lower": ((-14.0, 16.0), (-12.0, 12.0), (-10.0, 10.0)),
    "spine_mid":   ((-12.0, 14.0), (-14.0, 14.0), (-10.0, 10.0)),
    "chest":       ((-10.0, 12.0), (-14.0, 14.0), (-9.0, 9.0)),
    "clavicle_l":  ((-8.0, 10.0),  (-8.0, 8.0),   (-12.0, 12.0)),
    "clavicle_r":  ((-8.0, 10.0),  (-8.0, 8.0),   (-12.0, 12.0)),
    "shoulder_l":  ((-45.0, 70.0), (-40.0, 40.0), (-35.0, 35.0)),
    "shoulder_r":  ((-45.0, 70.0), (-40.0, 40.0), (-35.0, 35.0)),
    "elbow_l":     ((-2.0, 150.0), (-25.0, 25.0), (-15.0, 15.0)),
    "elbow_r":     ((-2.0, 150.0), (-25.0, 25.0), (-15.0, 15.0)),
    "wrist_l":     ((-60.0, 60.0), (-25.0, 25.0), (-30.0, 30.0)),
    "wrist_r":     ((-60.0, 60.0), (-25.0, 25.0), (-30.0, 30.0)),
    # Widened after the attention targets were spread out. A mouse sits 30
    # degrees below the lens and a keyboard 34, so looking at one's own hands
    # legitimately needs more downward travel than the first pass allowed - the
    # limits were clamping ordinary behaviour 82 times in five minutes, which
    # means they were describing a body that cannot look at its own desk.
    "neck":        ((-25.0, 34.0), (-45.0, 45.0), (-22.0, 22.0)),
    "head":        ((-22.0, 26.0), (-35.0, 35.0), (-25.0, 25.0)),
    # Asymmetric on purpose: the eye travels further down than up. Looking at a
    # keyboard 34 degrees below the lens is ordinary, looking 34 degrees up is
    # not.
    # Deliberately just outside the gaze system's own clamp (+-0.45 / +-0.55
    # gaze units, i.e. +-37.5 / +-45.8 degrees), which is the real ocular
    # authority and predates this table. Setting these tighter made the
    # backstop bind during ordinary gaze - a limit that fires in normal
    # operation is not a backstop, it is a second, disagreeing opinion.
    "eye_l":       ((-38.5, 38.5), (-46.5, 46.5), (-2.5, 2.5)),
    "eye_r":       ((-38.5, 38.5), (-46.5, 46.5), (-2.5, 2.5)),
}


class AnatomicalLimits:
    """Ranges of the human head and neck, in degrees.

    Values are the conservative end of published cervical range of motion for
    an adult, then reduced for a *seated presenter attending to a desk*: nobody
    holds their head at the edge of its range voluntarily, and a rig that can
    reach the anatomical extreme will eventually be driven there by a summed
    set of plausible contributions.
    """

    def __init__(self, yaw=38.0, pitch_up=22.0, pitch_down=30.0, roll=24.0):
        self.yaw = yaw
        self.pitch_up = pitch_up
        self.pitch_down = pitch_down
        self.roll = roll


SEATED_PRESENTER = AnatomicalLimits()


def _soft(x: float, limit: float) -> float:
    if limit <= 0.0:
        return 0.0
    return limit * math.tanh(x / limit)


def apply_body(motion) -> dict[str, int]:
    """Clamp every joint of a motion state to its seated range, in place.

    Returns a count of how many axes were actually clamped, per joint. A
    non-empty result is not necessarily a bug - it is the constraint stage doing
    its job - but a *persistently* non-empty one means something upstream is
    asking for a pose this body cannot hold.
    """
    hit: dict[str, int] = {}
    joints = motion.joints()
    for name, ranges in SEATED_JOINT_LIMITS.items():
        j = joints.get(name)
        if j is None:
            continue
        for axis, (lo, hi) in zip(("rx", "ry", "rz"), ranges):
            v = getattr(j, axis)
            c = min(max(v, lo), hi)
            if c != v:
                setattr(j, axis, c)
                hit[name] = hit.get(name, 0) + 1
    return hit


def apply(pose, limits: AnatomicalLimits = SEATED_PRESENTER) -> None:
    """Constrain a pose in place. Cheap enough to run every frame."""
    pose.yaw = _soft(pose.yaw, limits.yaw)
    pose.pitch = _soft(pose.pitch,
                       limits.pitch_up if pose.pitch >= 0 else limits.pitch_down)
    pose.roll = _soft(pose.roll, limits.roll)

    # Eyelids are a closed interval, not a soft limit: there is no such thing
    # as being 1.05 open.
    pose.eye_open_l = min(max(pose.eye_open_l, 0.0), 1.0)
    pose.eye_open_r = min(max(pose.eye_open_r, 0.0), 1.0)

    # The eye cannot leave its orbit. Beyond roughly this eccentricity a person
    # turns their head instead, which the attention system already models.
    pose.gaze_x = min(max(pose.gaze_x, -0.55), 0.55)
    pose.gaze_y = min(max(pose.gaze_y, -0.45), 0.45)
