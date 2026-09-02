"""Interpolation curves for physically plausible motion.

Every visible transition in this system goes through one of these. Linear
interpolation is deliberately absent: constant velocity with instant starts and
stops is the single most recognisable signature of computer animation, and the
brief calls it out explicitly.

The default for voluntary movement is `min_jerk`. Human point-to-point movement
- reaching, saccades, head reorientation - is well described by the
minimum-jerk model (Flash & Hogan 1985): among all trajectories connecting two
points in a fixed time, the one minimising integrated squared jerk is the
quintic below, and measured human trajectories track it closely.

References
----------
Flash, T. & Hogan, N. (1985). The coordination of arm movements: an
    experimentally confirmed mathematical model. J. Neuroscience 5(7).
"""

from __future__ import annotations

import math

__all__ = [
    "clamp",
    "min_jerk",
    "min_jerk_value",
    "ease_in_out_cubic",
    "ease_out_cubic",
    "smoothstep",
    "blink_profile",
    "lerp",
]


def clamp(value: float, low: float, high: float) -> float:
    """Constrain `value` to [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def lerp(a: float, b: float, t: float) -> float:
    """Linear blend. Only for mixing *values*, never for timing a motion."""
    return a + (b - a) * t


def min_jerk(tau: float) -> float:
    """Minimum-jerk position profile over normalised time `tau` in [0, 1].

    Returns the fraction of the movement completed. Zero velocity and zero
    acceleration at both endpoints, which is what makes a movement start and
    stop without a visible mechanical snap.
    """
    t = clamp(tau, 0.0, 1.0)
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def min_jerk_value(start: float, end: float, tau: float) -> float:
    """Minimum-jerk interpolation from `start` to `end`."""
    return start + (end - start) * min_jerk(tau)


def ease_in_out_cubic(tau: float) -> float:
    """Symmetric cubic ease. Slightly snappier than min_jerk at the endpoints."""
    t = clamp(tau, 0.0, 1.0)
    if t < 0.5:
        return 4.0 * t * t * t
    f = -2.0 * t + 2.0
    return 1.0 - (f * f * f) / 2.0


def ease_out_cubic(tau: float) -> float:
    """Fast start, soft landing. Used for settling motions."""
    t = clamp(tau, 0.0, 1.0)
    f = 1.0 - t
    return 1.0 - f * f * f


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite smoothstep, for fading weights in and out."""
    if edge1 <= edge0:
        return 0.0 if x < edge0 else 1.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def blink_profile(tau: float, close_fraction: float = 0.36) -> float:
    """Eyelid closure over one blink, normalised time in -> closure out.

    Returns 0 for fully open, 1 for fully closed.

    A human blink is markedly asymmetric: the lid slams shut and drifts back
    open. Closing takes roughly a third of the blink and reaches far higher
    angular velocity than the reopening phase, which is driven by levator
    palpebrae re-engagement rather than the fast orbicularis twitch. Rendering
    a blink as a symmetric triangle or sine is one of the loudest tells of a
    synthetic face, so the two phases get different curves:

      * closing  - ease-out cubic, so it is already near full speed immediately
      * opening  - min-jerk, a soft settle back to open

    Parameters
    ----------
    tau
        Normalised time within the blink, 0 to 1.
    close_fraction
        Portion of the blink spent closing. ~0.3-0.4 matches reported
        closing/opening duration ratios for spontaneous blinks.
    """
    t = clamp(tau, 0.0, 1.0)
    cf = clamp(close_fraction, 0.05, 0.95)
    if t <= cf:
        return ease_out_cubic(t / cf)
    return 1.0 - min_jerk((t - cf) / (1.0 - cf))


def wrap_angle(degrees: float) -> float:
    """Normalise an angle to (-180, 180]."""
    return (degrees + 180.0) % 360.0 - 180.0


def approach(current: float, target: float, rate: float, dt: float) -> float:
    """Frame-rate independent exponential approach toward `target`.

    `rate` is the reciprocal time constant (larger converges faster). Used for
    continuous quantities that should track a moving target without ever
    snapping - never for discrete voluntary movements, which use min_jerk.
    """
    if rate <= 0.0 or dt <= 0.0:
        return current
    alpha = 1.0 - math.exp(-rate * dt)
    return current + (target - current) * alpha
