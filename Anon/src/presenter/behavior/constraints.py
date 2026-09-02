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

__all__ = ["AnatomicalLimits", "SEATED_PRESENTER", "apply"]


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
