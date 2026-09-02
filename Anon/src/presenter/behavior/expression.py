"""Facial micro-expressions.

The instruction here is unusually strict and worth respecting literally: most
of the time the face should be near neutral. The failure mode is not an
under-expressive avatar - a resting human face genuinely is close to neutral -
it is an avatar that cycles through visible emotions for no reason, which reads
as either unstable or insincere.

So this system is deliberately quiet. Events are rare (median ~11 s), small
(peak activation ~0.16 of full range), and slow (onset and release over
hundreds of milliseconds, never a snap). What it produces is closer to shifting
facial *tone* than to expression: a brow that lifts a millimetre, a jaw that
unclenches, tension arriving and leaving the mid-face.

Two mechanisms keep it from looking synthetic:

* **No immediate repeats.** The last expression is excluded from the next
  draw. Repetition is what the eye latches onto, more than any single motion.
* **Asymmetry.** Sides are never driven with identical values. Human facial
  musculature is not symmetric and spontaneous (as opposed to posed)
  expressions are measurably more asymmetric still. The magnitude is kept
  small - enough to break the mirror, not enough to read as a droop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import AvatarPose, BehaviorEvent
from .context import Drives
from .curves import clamp, min_jerk, smoothstep

__all__ = ["ExpressionSystem"]


# Channel weights per expression. Values are fractions of the profile's
# expression_strength, so the whole system scales from one config knob.
_EXPRESSIONS: dict[str, dict[str, float]] = {
    # A brow lift with no emotional content - the commonest idle facial event.
    "brow_lift": {"brow_l": 1.0, "brow_r": 0.92},
    # Slight medial brow pull. Reads as attention, not as a frown, while small.
    "brow_tension": {"brow_furrow": 0.85, "brow_l": -0.18, "brow_r": -0.14},
    # Orbicularis tightening: the eye-narrowing that accompanies genuine
    # engagement. Kept well below anything that looks like a squint.
    "eye_engage": {"squint_l": 0.8, "squint_r": 0.7, "cheek": 0.45},
    # The faintest asymmetric mouth-corner movement. Not a smile.
    "mouth_settle": {"mouth_corner_l": 0.6, "mouth_corner_r": 0.42},
    # Tension leaving the jaw. Mostly invisible; contributes to the sense that
    # the face is a body part rather than a texture.
    "jaw_release": {"jaw": 0.7},
    # Cheek/nasolabial activation, very small.
    "cheek_tension": {"cheek": 0.8, "mouth_corner_l": 0.25, "mouth_corner_r": 0.3},
    # A composite that reads as a flicker of mild positive affect.
    "mild_positive": {
        "mouth_corner_l": 0.9,
        "mouth_corner_r": 0.78,
        "cheek": 0.6,
        "squint_l": 0.45,
        "squint_r": 0.4,
    },
    # A composite that reads as a flicker of mild concern.
    "mild_concern": {"brow_furrow": 0.9, "brow_l": 0.3, "brow_r": 0.22},
}

# Which expressions each state prefers. Absent states use the neutral set.
_STATE_POOL: dict[str, tuple[str, ...]] = {
    "THINKING": ("brow_tension", "eye_engage", "jaw_release", "brow_lift"),
    "LISTENING": ("brow_lift", "eye_engage", "mouth_settle", "cheek_tension"),
    "FOCUSED": ("brow_tension", "eye_engage", "jaw_release"),
    "READING": ("brow_tension", "eye_engage"),
    "MILD_POSITIVE": ("mild_positive", "cheek_tension", "mouth_settle", "brow_lift"),
    "MILD_CONCERN": ("mild_concern", "brow_tension", "eye_engage"),
    "PRE_SPEECH": ("brow_lift", "jaw_release", "eye_engage"),
}

_NEUTRAL_POOL = (
    "brow_lift",
    "brow_tension",
    "eye_engage",
    "mouth_settle",
    "jaw_release",
    "cheek_tension",
)


@dataclass
class _ActiveExpression:
    name: str
    start: float
    attack: float
    hold: float
    release: float
    weights: dict[str, float]
    peak: float
    asym_l: float
    asym_r: float

    @property
    def total(self) -> float:
        return self.attack + self.hold + self.release

    def envelope(self, now: float) -> float:
        """Attack-hold-release envelope, 0..1.

        Both edges are smooth. An expression that switches on or off abruptly
        is the single most artificial thing a face can do, more so than an
        expression that is simply too large.
        """
        t = now - self.start
        if t < 0.0:
            return 0.0
        if t < self.attack:
            return min_jerk(t / max(self.attack, 1e-3))
        if t < self.attack + self.hold:
            return 1.0
        released = (t - self.attack - self.hold) / max(self.release, 1e-3)
        if released >= 1.0:
            return 0.0
        return 1.0 - min_jerk(released)


class ExpressionSystem:
    """Writes brow, squint, cheek, mouth-corner and jaw channels."""

    def __init__(self) -> None:
        self._active: _ActiveExpression | None = None
        self._next_at: float | None = None
        self._last_name: str | None = None
        self.expression_count: int = 0

    def _schedule_next(self, drives: Drives) -> None:
        p = drives.profile
        median = drives.rate(p.expression_median_interval, drives.mod.expression_rate)
        median *= drives.mod.stillness  # persistent state trait only
        interval = drives.rng.lognormal_interval(
            median=median, shape=p.expression_interval_shape, low=2.0, high=180.0
        )
        self._next_at = drives.now + interval

    def _begin(self, drives: Drives) -> None:
        rng = drives.rng
        p = drives.profile

        pool = _STATE_POOL.get(drives.state.value, _NEUTRAL_POOL)
        # Never repeat the previous expression back to back. Repetition is what
        # a viewer notices first over a long watch.
        candidates = [n for n in pool if n != self._last_name] or list(pool)
        name = rng.choice(candidates)

        peak = abs(rng.gauss(p.expression_strength, p.expression_strength * 0.35))
        peak = clamp(peak, 0.02, p.expression_strength * 2.2)

        attack = rng.uniform(0.22, 0.5)
        hold = rng.uniform(0.15, p.expression_duration)
        release = rng.uniform(0.45, 1.3)

        asym = p.brow_asymmetry
        self._active = _ActiveExpression(
            name=name,
            start=drives.now,
            attack=attack,
            hold=hold,
            release=release,
            weights=_EXPRESSIONS[name],
            peak=peak,
            asym_l=1.0 + rng.uniform(-asym, asym),
            asym_r=1.0 + rng.uniform(-asym, asym),
        )
        self._last_name = name
        self._next_at = None
        self.expression_count += 1

        drives.emit(
            BehaviorEvent(
                time=drives.now,
                kind="expression",
                detail=f"{name} peak={peak:.3f} dur={attack + hold + release:.2f}s",
                magnitude=peak,
                metadata={"name": name, "peak": peak},
            )
        )

    def update(self, drives: Drives, pose: AvatarPose) -> None:
        if self._next_at is None and self._active is None:
            self._schedule_next(drives)

        if (
            self._active is None
            and self._next_at is not None
            and drives.now >= self._next_at
        ):
            if drives.allow_voluntary():
                self._begin(drives)
            else:
                self._next_at = drives.now + 0.4

        if self._active is None:
            return

        expression = self._active
        if drives.now - expression.start >= expression.total:
            self._active = None
            self._schedule_next(drives)
            return

        level = expression.envelope(drives.now) * expression.peak

        for channel, weight in expression.weights.items():
            value = level * weight
            # Apply the per-side asymmetry only to paired channels.
            if channel.endswith("_l"):
                value *= expression.asym_l
            elif channel.endswith("_r"):
                value *= expression.asym_r
            setattr(pose, channel, getattr(pose, channel) + value)

    @property
    def active_name(self) -> str:
        return self._active.name if self._active else "neutral"
