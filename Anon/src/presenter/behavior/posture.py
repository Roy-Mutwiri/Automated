"""Slow postural drift and occasional weight shifts.

The slowest layer in the system, and the one that operates below conscious
notice entirely. Its job is to make the avatar's framing change over minutes,
so that a viewer who looks away and back does not find the head in the
identical position it occupied ninety seconds ago.

Two components:

* **Continuous drift** - an Ornstein-Uhlenbeck process with a ~14 s
  correlation time on translation. Never still, never anywhere in particular.
* **Weight shifts** - rare (median ~40 s), small repositionings of the torso
  such as a seated person settling. These are the only postural events large
  enough to be consciously noticed, and they should be noticeable perhaps once
  or twice a minute at most.

Amplitudes are in fractions of face width, so they stay correct if the framing
or output resolution changes.
"""

from __future__ import annotations

from ..types import AvatarPose, BehaviorEvent
from .context import Drives
from .curves import clamp, min_jerk
from .randomness import OrnsteinUhlenbeck

__all__ = ["PostureSystem"]


class PostureSystem:
    """Writes small translation offsets on top of everything else."""

    def __init__(self, profile) -> None:
        self._drift_x = OrnsteinUhlenbeck.from_amplitude(
            profile.posture_amplitude, profile.posture_time
        )
        self._drift_y = OrnsteinUhlenbeck.from_amplitude(
            profile.posture_amplitude * 0.7, profile.posture_time * 1.3
        )

        self._shift_from = (0.0, 0.0)
        self._shift_to = (0.0, 0.0)
        self._shift_start: float | None = None
        self._shift_duration = 1.0
        self._offset = (0.0, 0.0)
        self._next_shift_at: float | None = None
        self.shift_count = 0

    def _schedule_next(self, drives: Drives) -> None:
        p = drives.profile
        median = drives.rate(
            p.posture_shift_median_interval, drives.mod.head_rate
        ) * drives.mod.stillness
        interval = drives.rng.lognormal_interval(
            median=median, shape=0.6, low=8.0, high=400.0
        )
        self._next_shift_at = drives.now + interval

    def _begin_shift(self, drives: Drives) -> None:
        p = drives.profile
        rng = drives.rng
        # Biased back toward centre so posture does not random-walk out of
        # frame over a long run.
        cx, cy = self._offset
        target = (
            clamp(rng.gauss(-cx * 0.6, p.posture_amplitude * 2.2),
                  -p.posture_amplitude * 5.0, p.posture_amplitude * 5.0),
            clamp(rng.gauss(-cy * 0.6, p.posture_amplitude * 1.6),
                  -p.posture_amplitude * 4.0, p.posture_amplitude * 4.0),
        )
        self._shift_from = self._offset
        self._shift_to = target
        self._shift_start = drives.now
        # Slow: a weight shift is not a gesture, and a fast one reads as a
        # flinch.
        self._shift_duration = rng.uniform(1.1, 2.6)
        self._next_shift_at = None
        self.shift_count += 1

        drives.emit(
            BehaviorEvent(
                time=drives.now,
                kind="posture_shift",
                detail=f"to=({target[0]:+.4f},{target[1]:+.4f}) dur={self._shift_duration:.2f}s",
                magnitude=abs(target[0] - cx) + abs(target[1] - cy),
            )
        )

    def update(self, drives: Drives, pose: AvatarPose) -> None:
        if self._next_shift_at is None and self._shift_start is None:
            self._schedule_next(drives)

        if (
            self._shift_start is None
            and self._next_shift_at is not None
            and drives.now >= self._next_shift_at
        ):
            if drives.allow_voluntary():
                self._begin_shift(drives)
            else:
                self._next_shift_at = drives.now + 0.5

        if self._shift_start is not None:
            tau = clamp(
                (drives.now - self._shift_start) / max(self._shift_duration, 1e-3),
                0.0,
                1.0,
            )
            eased = min_jerk(tau)
            self._offset = (
                self._shift_from[0] + (self._shift_to[0] - self._shift_from[0]) * eased,
                self._shift_from[1] + (self._shift_to[1] - self._shift_from[1]) * eased,
            )
            if tau >= 1.0:
                self._shift_start = None
                self._schedule_next(drives)

        self._drift_x.step(drives.dt, drives.rng)
        self._drift_y.step(drives.dt, drives.rng)

        pose.tx += self._offset[0] + self._drift_x.value
        pose.ty += self._offset[1] + self._drift_y.value
