"""Per-frame drives shared by every behaviour subsystem.

The scheduler resolves profile + state + arousal into one `Drives` object each
frame and hands the same instance to every subsystem. Subsystems never reach
back into the scheduler, which keeps them independently testable: a blink
generator can be exercised over a simulated hour without constructing a
renderer or a clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .randomness import Rng
from .state import BehaviorState, MotionProfile, StateModulation

__all__ = ["Drives"]


@dataclass
class Drives:
    """Everything a subsystem needs to decide what to do this frame."""

    rng: Rng
    profile: MotionProfile
    state: BehaviorState
    mod: StateModulation

    # Slow-varying -1..1 signal. Positive means livelier than baseline. This is
    # what creates minutes-long stretches of relative stillness followed by
    # more active stretches, rather than a uniform density of movement that
    # betrays itself over a long viewing.
    arousal: float = 0.0

    # Seconds since the engine started. Subsystems use elapsed time, never a
    # frame count, so behaviour is identical at 25 and 60 FPS.
    now: float = 0.0
    dt: float = 0.0

    # Cross-subsystem coupling, written by the scheduler before subsystems run.
    # Blink probability rises with time held in fixation, so the gaze system's
    # state has to be visible to the blink system.
    time_since_gaze_shift: float = 0.0
    time_since_head_move: float = 0.0
    # True while any deliberate movement is in flight. Subsystems consult this
    # to avoid stacking voluntary motions on top of each other, which is what
    # makes an avatar look permanently busy.
    motion_in_flight: bool = False

    # Transient crowding factor from the scheduler's motion budget, >= 1.0.
    # Deliberately NOT folded into sampled intervals: an interval drawn just
    # after a movement would bake in that moment's elevated suppression and
    # stay stretched for its whole duration, which pushes the long-run rate far
    # below the profile's intent. It is applied as a fire-time gate instead, so
    # the interval distribution keeps its shape and only genuinely crowded
    # moments get deferred.
    suppression: float = 1.0

    events: list = field(default_factory=list)

    def allow_voluntary(self) -> bool:
        """Whether a discretionary movement may start on this frame.

        Probabilistic rather than a hard threshold: a hard cutoff would make
        movements bunch up immediately after the budget crosses back under it,
        putting a faint rhythm into exactly the behaviours that must not have
        one.
        """
        if self.suppression <= 1.0:
            return True
        return self.rng.chance(1.0 / self.suppression)

    def rate(self, base_interval: float, multiplier: float) -> float:
        """Convert a base median interval into an arousal-adjusted one.

        Higher arousal and higher state multipliers shorten the interval.
        Guarded against collapsing to zero, which would spin the scheduler.
        """
        arousal_gain = 1.0 + 0.45 * self.arousal
        gain = max(multiplier * arousal_gain * self.profile.activity, 1e-3)
        return max(base_interval / gain, 0.05)

    def emit(self, event) -> None:
        self.events.append(event)
