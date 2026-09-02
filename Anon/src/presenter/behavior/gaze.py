"""Gaze: fixations, saccades, microsaccades and drift.

Eyes are the highest-priority realism target in the brief, and the failure mode
it names - "eyes randomly jumping around" - comes from treating gaze as a
single random walk. Real fixational behaviour is three superimposed processes
on very different scales, and reproducing that layering is what separates
"alive" from "twitchy":

* **Fixation.** The gaze holds a target for hundreds of milliseconds to
  seconds. This is the dominant state; the eyes are *mostly* holding still.
* **Microsaccades.** 1-2 per second, under 1 degree of visual angle, 6-30 ms.
  Individually invisible. Their absence is not: a mathematically fixed pupil
  is the single clearest sign of a dead face, which is why this system runs
  even when nothing else is moving.
* **Drift.** Slow ocular wander between microsaccades, smaller and slower
  still. Modelled as an Ornstein-Uhlenbeck process.

Voluntary gaze shifts sit on top of these. They are ballistic: a real saccade
reaches its target in tens of milliseconds with no visible travel, so animating
one as a slow glide across the eye is wrong. It is rendered here as a fast
min-jerk transit, and the eye then *holds*.

The `camera_affinity` drive biases target selection back toward the lens, which
is what makes a presenter read as addressing the viewer rather than staring
past them.

See docs/human_behavior.md for the measurements behind the defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import AvatarPose, BehaviorEvent
from .context import Drives
from .curves import clamp, min_jerk
from .randomness import OrnsteinUhlenbeck

__all__ = ["GazeSystem"]


@dataclass
class _Saccade:
    """A ballistic gaze transit in flight."""

    start: float
    duration: float
    from_x: float
    from_y: float
    to_x: float
    to_y: float

    def progress(self, now: float) -> float:
        if self.duration <= 0.0:
            return 1.0
        return clamp((now - self.start) / self.duration, 0.0, 1.0)


class GazeSystem:
    """Produces gaze_x / gaze_y, and reports fixation age for blink coupling."""

    def __init__(self, profile) -> None:
        self.target_x = 0.0
        self.target_y = 0.0
        self._fixation_x = 0.0
        self._fixation_y = 0.0
        self._saccade: _Saccade | None = None
        self._next_saccade_at: float | None = None
        self.last_shift_time: float = 0.0
        self.saccade_count: int = 0
        self.microsaccade_count: int = 0

        # Microsaccade state: a tiny offset that is re-drawn at ~1-2 Hz and
        # decays between draws, rather than a continuous jitter.
        self._micro_x = 0.0
        self._micro_y = 0.0
        self._next_micro_at: float | None = None

        self._drift_x = OrnsteinUhlenbeck.from_amplitude(
            profile.gaze_drift_amplitude, profile.gaze_drift_time
        )
        self._drift_y = OrnsteinUhlenbeck.from_amplitude(
            profile.gaze_drift_amplitude * 0.75, profile.gaze_drift_time
        )

    # -- target selection ---------------------------------------------------
    def _pick_target(self, drives: Drives) -> tuple[float, float, str]:
        """Choose where to look next.

        Two-thirds of shifts return to the lens; the rest are small excursions.
        That ratio is what produces "mostly looking at you, occasionally
        glancing away" instead of an unfocused wander.
        """
        p = drives.profile
        rng = drives.rng
        affinity = drives.mod.camera_affinity

        return_prob = clamp(p.gaze_return_probability + 0.3 * affinity, 0.05, 0.95)

        # Already away from centre and inclined to come back: return to lens.
        away = math.hypot(self.target_x, self.target_y)
        if away > 0.04 and rng.chance(return_prob):
            # Not exactly (0, 0). Re-fixating on a face lands near the previous
            # point, never on the identical pixel, and an exactly-centred gaze
            # looks mechanically locked.
            jitter = p.microsaccade_amplitude * 1.5
            return (
                rng.uniform(-jitter, jitter),
                rng.uniform(-jitter, jitter),
                "gaze_return",
            )

        amplitude = rng.truncated_gauss(
            p.saccade_amplitude,
            p.saccade_amplitude_sigma,
            0.02,
            p.saccade_max_amplitude,
        )
        # Arousal widens excursions a little; a subdued presenter's eyes move
        # less far, not just less often.
        amplitude *= 1.0 + 0.25 * drives.arousal

        angle = rng.uniform(-math.pi, math.pi)
        dx = math.cos(angle) * amplitude
        dy = math.sin(angle) * amplitude * 0.7  # vertical range is smaller

        # Thinking and reading pull the gaze down and away.
        dy -= drives.mod.downward_bias * abs(dy) * 1.4
        if drives.mod.downward_bias > 0.0 and rng.chance(drives.mod.downward_bias):
            dy = -abs(dy)

        target_x = clamp(dx, -p.saccade_max_amplitude, p.saccade_max_amplitude)
        target_y = clamp(dy, -p.saccade_max_amplitude, p.saccade_max_amplitude)

        label = "gaze_down" if target_y < -0.06 else (
            "gaze_left" if target_x < -0.06 else (
                "gaze_right" if target_x > 0.06 else "gaze_shift"
            )
        )
        return target_x, target_y, label

    def _schedule_next_saccade(self, drives: Drives) -> None:
        p = drives.profile
        median = drives.rate(p.saccade_median_interval, drives.mod.gaze_rate)
        # Stillness damps the *rate* of voluntary shifts without touching
        # microsaccades, so a still avatar still has living eyes.
        median *= drives.mod.stillness  # persistent state trait only
        interval = drives.rng.lognormal_interval(
            median=median, shape=p.saccade_interval_shape, low=0.35, high=45.0
        )
        self._next_saccade_at = drives.now + interval

    def _begin_saccade(self, drives: Drives) -> None:
        target_x, target_y, label = self._pick_target(drives)
        p = drives.profile

        distance = math.hypot(target_x - self.target_x, target_y - self.target_y)
        # Saccade duration scales with amplitude (the main sequence), but stays
        # in the tens of milliseconds - effectively 1-3 frames at 30 FPS.
        duration = clamp(
            p.saccade_duration * (0.6 + 2.2 * distance), 0.022, 0.13
        )

        self._saccade = _Saccade(
            start=drives.now,
            duration=duration,
            from_x=self.target_x,
            from_y=self.target_y,
            to_x=target_x,
            to_y=target_y,
        )
        self.target_x = target_x
        self.target_y = target_y
        self.last_shift_time = drives.now
        self.saccade_count += 1
        self._next_saccade_at = None

        drives.emit(
            BehaviorEvent(
                time=drives.now,
                kind=label,
                detail=f"to=({target_x:+.3f},{target_y:+.3f}) amp={distance:.3f}",
                magnitude=distance,
                metadata={"x": target_x, "y": target_y, "duration": duration},
            )
        )

    # -- microsaccades ------------------------------------------------------
    def _update_microsaccades(self, drives: Drives) -> None:
        p = drives.profile
        rng = drives.rng

        if self._next_micro_at is None:
            self._next_micro_at = drives.now + rng.exponential_interval(
                1.0 / max(p.microsaccade_rate, 0.05)
            )

        if drives.now >= self._next_micro_at:
            amplitude = abs(
                rng.gauss(p.microsaccade_amplitude, p.microsaccade_amplitude * 0.45)
            )
            angle = rng.uniform(-math.pi, math.pi)
            self._micro_x = math.cos(angle) * amplitude
            self._micro_y = math.sin(angle) * amplitude * 0.8
            self.microsaccade_count += 1
            # Poisson arrivals: microsaccades have no strong refractory
            # structure, unlike blinks and voluntary saccades.
            self._next_micro_at = drives.now + rng.exponential_interval(
                1.0 / max(p.microsaccade_rate, 0.05)
            )

        # Decay toward zero between draws so the offset does not accumulate.
        decay = math.exp(-drives.dt / 0.28)
        self._micro_x *= decay
        self._micro_y *= decay

    # -- per-frame ----------------------------------------------------------
    def update(self, drives: Drives, pose: AvatarPose, attention=None) -> None:
        """Write gaze into the pose.

        With an `attention` system supplied, *where* the eyes point is not this
        system's decision any more - the presenter is looking at a thing in the
        room and the attention model owns which thing and when it changes. What
        stays here is everything involuntary: microsaccades and slow drift.
        That split matters because those two are the difference between eyes
        that hold a target and eyes that are dead, and they must keep running
        regardless of what attention is doing.

        Without one, the original self-directed behaviour is used unchanged, so
        the subsystem stays independently testable.
        """
        if attention is not None:
            self._fixation_x = attention.gaze_x
            self._fixation_y = attention.gaze_y
            self.last_shift_time = attention.last_change
            self.saccade_count = attention.shift_count

            self._update_microsaccades(drives)
            self._drift_x.step(drives.dt, drives.rng)
            self._drift_y.step(drives.dt, drives.rng)
            pose.gaze_x = clamp(
                self._fixation_x + self._micro_x + self._drift_x.value, -0.55, 0.55)
            pose.gaze_y = clamp(
                self._fixation_y + self._micro_y + self._drift_y.value, -0.45, 0.45)
            return

        if self._next_saccade_at is None and self._saccade is None:
            self._schedule_next_saccade(drives)

        if (
            self._saccade is None
            and self._next_saccade_at is not None
            and drives.now >= self._next_saccade_at
            # Do not launch a voluntary gaze shift mid-blink: the eye is
            # occluded and the shift would be wasted, and real gaze shifts do
            # tend to be coordinated with lid position.
            and not drives.motion_in_flight
        ):
            if drives.allow_voluntary():
                self._begin_saccade(drives)
            else:
                self._next_saccade_at = drives.now + 0.2

        if self._saccade is not None:
            tau = self._saccade.progress(drives.now)
            eased = min_jerk(tau)
            self._fixation_x = self._saccade.from_x + (
                self._saccade.to_x - self._saccade.from_x
            ) * eased
            self._fixation_y = self._saccade.from_y + (
                self._saccade.to_y - self._saccade.from_y
            ) * eased
            if tau >= 1.0:
                self._saccade = None
                self._schedule_next_saccade(drives)

        self._update_microsaccades(drives)
        self._drift_x.step(drives.dt, drives.rng)
        self._drift_y.step(drives.dt, drives.rng)

        gx = self._fixation_x + self._micro_x + self._drift_x.value
        gy = self._fixation_y + self._micro_y + self._drift_y.value

        # Anatomical limit. Beyond roughly this the eye would be at the edge of
        # its orbit and the head would have turned instead.
        pose.gaze_x = clamp(gx, -0.55, 0.55)
        pose.gaze_y = clamp(gy, -0.45, 0.45)

    @property
    def fixation_age_at(self):
        return self.last_shift_time

    @property
    def is_saccading(self) -> bool:
        return self._saccade is not None
