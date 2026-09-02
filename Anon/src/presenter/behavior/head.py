"""Head pose: deliberate adjustments over a bed of involuntary sway.

The brief forbids sinusoidal head motion, and the reason is worth stating: any
periodic signal becomes visible once a viewer watches for longer than its
period. A sine at 0.2 Hz is invisible for ten seconds and unmistakable after
two minutes. Since this avatar may run for hours, nothing here may be periodic.

Head motion is therefore two aperiodic layers:

* **Involuntary sway** - an Ornstein-Uhlenbeck process per axis. Always on,
  small, mean-reverting, never repeating. This is the postural micro-correction
  of a head being held still, and it is what stops a "still" avatar looking
  frozen.
* **Voluntary adjustments** - infrequent discrete repositionings on a
  min-jerk trajectory, each one settling to a new hold pose. These are the
  movements a viewer actually notices, and they are rare on purpose.

The two are additive: the sway continues during and after a deliberate move,
so a movement never ends in unnatural dead stillness.

Amplitudes are small by design. A seated presenter's head yaw during quiet
attention lives within a few degrees; the intuitive setting is far too large
and produces the "bobbing" the brief lists as a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import AvatarPose, BehaviorEvent
from .context import Drives
from .curves import clamp, min_jerk
from .randomness import OrnsteinUhlenbeck

__all__ = ["HeadSystem"]


@dataclass
class _HeadMove:
    """A deliberate repositioning in flight."""

    start: float
    duration: float
    from_yaw: float
    from_pitch: float
    from_roll: float
    to_yaw: float
    to_pitch: float
    to_roll: float

    def progress(self, now: float) -> float:
        if self.duration <= 0.0:
            return 1.0
        return clamp((now - self.start) / self.duration, 0.0, 1.0)


class HeadSystem:
    """Produces yaw / pitch / roll."""

    def __init__(self, profile) -> None:
        # The pose the head is deliberately holding. Sway rides on top of it.
        self.hold_yaw = 0.0
        self.hold_pitch = 0.0
        self.hold_roll = 0.0

        self._move: _HeadMove | None = None
        self._next_move_at: float | None = None
        self.last_move_time: float = 0.0
        self.move_count: int = 0

        # Independent processes per axis with slightly different correlation
        # times: identical dynamics on all three axes reads as a rigid body on
        # a spring rather than as a neck.
        self._sway_yaw = OrnsteinUhlenbeck.from_amplitude(
            profile.head_sway_amplitude, profile.head_sway_time
        )
        self._sway_pitch = OrnsteinUhlenbeck.from_amplitude(
            profile.head_sway_amplitude * 0.8, profile.head_sway_time * 1.25
        )
        self._sway_roll = OrnsteinUhlenbeck.from_amplitude(
            profile.head_sway_amplitude * 0.55, profile.head_sway_time * 0.85
        )

    # -- scheduling ---------------------------------------------------------
    def _schedule_next(self, drives: Drives) -> None:
        p = drives.profile
        median = drives.rate(p.head_median_interval, drives.mod.head_rate)
        median *= drives.mod.stillness  # persistent state trait only
        interval = drives.rng.lognormal_interval(
            median=median, shape=p.head_interval_shape, low=1.2, high=120.0
        )
        self._next_move_at = drives.now + interval

    def _begin_move(self, drives: Drives) -> None:
        p = drives.profile
        rng = drives.rng

        # Movements are biased back toward neutral when the head has drifted
        # away from it. Without this the head performs a random walk and ends
        # up holding an odd angle for minutes, which reads as a stiff neck.
        recentre = 0.55

        d_yaw = rng.gauss(-self.hold_yaw * recentre, p.head_yaw_amplitude)
        d_pitch = rng.gauss(-self.hold_pitch * recentre, p.head_pitch_amplitude)
        d_roll = rng.gauss(-self.hold_roll * recentre, p.head_roll_amplitude)

        to_yaw = clamp(self.hold_yaw + d_yaw, -p.head_max_yaw, p.head_max_yaw)
        to_pitch = clamp(self.hold_pitch + d_pitch, -p.head_max_pitch, p.head_max_pitch)
        to_roll = clamp(self.hold_roll + d_roll, -p.head_max_roll, p.head_max_roll)

        magnitude = max(
            abs(to_yaw - self.hold_yaw),
            abs(to_pitch - self.hold_pitch),
            abs(to_roll - self.hold_roll),
        )
        # Larger movements take longer, but sub-linearly - the same way human
        # movement duration grows with amplitude.
        duration = clamp(
            p.head_move_duration * (0.55 + 0.42 * magnitude), 0.22, 2.4
        )

        self._move = _HeadMove(
            start=drives.now,
            duration=duration,
            from_yaw=self.hold_yaw,
            from_pitch=self.hold_pitch,
            from_roll=self.hold_roll,
            to_yaw=to_yaw,
            to_pitch=to_pitch,
            to_roll=to_roll,
        )
        self.last_move_time = drives.now
        self.move_count += 1
        self._next_move_at = None

        axis = "yaw" if abs(d_yaw) >= max(abs(d_pitch), abs(d_roll)) else (
            "pitch" if abs(d_pitch) >= abs(d_roll) else "roll"
        )
        drives.emit(
            BehaviorEvent(
                time=drives.now,
                kind=f"head_{axis}",
                detail=(
                    f"to=({to_yaw:+.2f},{to_pitch:+.2f},{to_roll:+.2f})deg "
                    f"dur={duration:.2f}s"
                ),
                magnitude=magnitude,
                metadata={
                    "yaw": to_yaw,
                    "pitch": to_pitch,
                    "roll": to_roll,
                    "duration": duration,
                },
            )
        )

    # -- per-frame ----------------------------------------------------------
    def update(self, drives: Drives, pose: AvatarPose) -> None:
        if self._next_move_at is None and self._move is None:
            self._schedule_next(drives)

        if (
            self._move is None
            and self._next_move_at is not None
            and drives.now >= self._next_move_at
        ):
            if drives.allow_voluntary():
                self._begin_move(drives)
            else:
                # Crowded right now - something else just moved. Defer briefly
                # and re-check rather than re-drawing the interval, which would
                # bias the long-run rate downward.
                self._next_move_at = drives.now + 0.3

        if self._move is not None:
            eased = min_jerk(self._move.progress(drives.now))
            self.hold_yaw = self._move.from_yaw + (
                self._move.to_yaw - self._move.from_yaw
            ) * eased
            self.hold_pitch = self._move.from_pitch + (
                self._move.to_pitch - self._move.from_pitch
            ) * eased
            self.hold_roll = self._move.from_roll + (
                self._move.to_roll - self._move.from_roll
            ) * eased
            if self._move.progress(drives.now) >= 1.0:
                self._move = None
                self._schedule_next(drives)

        self._sway_yaw.step(drives.dt, drives.rng)
        self._sway_pitch.step(drives.dt, drives.rng)
        self._sway_roll.step(drives.dt, drives.rng)

        p = drives.profile
        pose.yaw = clamp(
            self.hold_yaw + self._sway_yaw.value, -p.head_max_yaw * 1.3, p.head_max_yaw * 1.3
        )
        pose.pitch = clamp(
            self.hold_pitch + self._sway_pitch.value,
            -p.head_max_pitch * 1.3,
            p.head_max_pitch * 1.3,
        )
        pose.roll = clamp(
            self.hold_roll + self._sway_roll.value,
            -p.head_max_roll * 1.3,
            p.head_max_roll * 1.3,
        )

    @property
    def is_moving(self) -> bool:
        return self._move is not None
