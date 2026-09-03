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

# The frame rate the suppression gate's probability is quoted at. Any other
# frame rate is converted to it, so behaviour rates do not depend on how fast
# the renderer happens to be running.
GATE_REFERENCE_HZ = 30.0


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

    # Smoothed frame interval. Distinct from `dt`, which is the instantaneous
    # step and far too noisy to make decisions from.
    #
    # Timing being frame-rate independent is necessary but not sufficient: a
    # movement also has to be *sampled* often enough to be perceived as motion
    # rather than as a jump. A 145 ms blink at 13 FPS lands on one intermediate
    # frame - the eye appears to teleport shut. Subsystems whose motion is fast
    # relative to the frame interval consult this and stretch themselves to
    # stay visible.
    frame_interval: float = 1.0 / 30.0

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

    # 0..1, how much the current attention target taxes vision. Written by the
    # scheduler from the attention model.
    #
    # This is the coupling that makes blink rate a consequence rather than a
    # constant. The literature is unambiguous that it must be one: spontaneous
    # blink rate runs 1.4-14.4/min while reading and 10.5-32.5/min in
    # conversation, a factor of three driven entirely by what the eyes are
    # doing. A single blink interval in a profile cannot represent that, and an
    # avatar that blinks at its personality rate while staring at a display is
    # wrong in a way viewers feel without being able to name.
    visual_demand: float = 0.3

    events: list = field(default_factory=list)

    def allow_voluntary(self) -> bool:
        """Whether a discretionary movement may start on this frame.

        Probabilistic rather than a hard threshold: a hard cutoff would make
        movements bunch up immediately after the budget crosses back under it,
        putting a faint rhythm into exactly the behaviours that must not have
        one.

        The probability is **per second, converted to this frame**, not per
        frame. A per-frame probability is retried every frame, so at 60 fps a
        deferred movement gets twice as many chances per second as at 30 and
        fires sooner - measured as 12.7% more blinks at 60 fps than at 25. The
        engine is time-driven everywhere else and this was the one place it
        quietly was not.
        """
        if self.suppression <= 1.0:
            return True
        p_ref = 1.0 / self.suppression
        frames = max(self.dt, 1e-6) * GATE_REFERENCE_HZ
        return self.rng.chance(1.0 - (1.0 - p_ref) ** frames)

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
