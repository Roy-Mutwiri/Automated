"""The behaviour engine: composes the subsystems into one coherent presenter.

This is the part the brief identifies as the actual hard problem - "knowing
WHEN NOT to move". Individually plausible subsystems, run independently, still
produce an avatar that fidgets, because their events interleave and the face
ends up doing *something* almost all the time. Three mechanisms here address
that:

**Arousal.** A slow Ornstein-Uhlenbeck signal on a ~26 s timescale, modulating
every subsystem's rate at once. Without it the statistical density of movement
is constant, which no human sustains; with it the avatar has genuinely quiet
minutes and livelier ones, which is the texture a long watch is judged on.

**Motion budget.** A leaky accumulator charged by each voluntary movement.
While it is high, discretionary behaviours are suppressed. This is what
enforces the brief's stillness -> event -> stillness rhythm rather than a
continuous stream of overlapping motions.

**Involuntary floor.** Breathing, ocular drift, microsaccades and head sway are
never suppressed. They are what keeps a *still* avatar from reading as a frozen
photograph. The distinction between "still" and "frozen" is the entire game,
and it lives in this split: voluntary motion is gated, involuntary motion is
not.

Everything is driven by elapsed wall-clock time, never by a frame counter, so
behaviour is identical whether the renderer is achieving 25 or 60 FPS and
degrades gracefully if a frame is late.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import AvatarPose, BehaviorEvent
from .attention import AttentionSystem
from .blinking import BlinkSystem
from .breathing import BreathingSystem
from .constraints import apply as apply_constraints
from .context import Drives
from .curves import clamp
from .expression import ExpressionSystem
from .gaze import GazeSystem
from .head import HeadSystem
from .posture import PostureSystem
from .randomness import OrnsteinUhlenbeck, Rng
from .scheduler import BehaviorMemory, StateScheduler
from .state import (
    PROFILES,
    STATE_MODULATION,
    BehaviorState,
    MotionProfile,
    StateModulation,
)

__all__ = ["BehaviorEngine", "EngineStats"]


@dataclass
class EngineStats:
    """Counters for the debug overlay and the long-run analysis."""

    frames: int = 0
    elapsed: float = 0.0
    blinks: int = 0
    saccades: int = 0
    microsaccades: int = 0
    head_moves: int = 0
    expressions: int = 0
    posture_shifts: int = 0
    breaths: int = 0

    def blinks_per_minute(self) -> float:
        return self.blinks / max(self.elapsed / 60.0, 1e-6)

    def saccades_per_minute(self) -> float:
        return self.saccades / max(self.elapsed / 60.0, 1e-6)

    def head_moves_per_minute(self) -> float:
        return self.head_moves / max(self.elapsed / 60.0, 1e-6)


class BehaviorEngine:
    """Produces an `AvatarPose` for any point in time.

    Deliberately renderer-agnostic and free of any I/O: it can be stepped
    faster than real time for analysis, which is what makes a 30-minute
    behavioural test cost a second rather than half an hour.
    """

    def __init__(
        self,
        profile: MotionProfile | str = "PRESENTER_CALM",
        state: BehaviorState = BehaviorState.IDLE_ATTENTIVE,
        seed: int | None = None,
        autonomous: bool = True,
    ) -> None:
        if isinstance(profile, str):
            if profile not in PROFILES:
                raise ValueError(
                    f"unknown profile {profile!r}; have {sorted(PROFILES)}"
                )
            profile = PROFILES[profile]

        self.profile = profile
        self.state = state
        self.rng = Rng(seed)
        self.now = 0.0
        self.stats = EngineStats()

        self.blink = BlinkSystem()
        self.gaze = GazeSystem(profile)
        self.head = HeadSystem(profile)
        self.breathing = BreathingSystem(profile)
        self.expression = ExpressionSystem()
        self.posture = PostureSystem(profile)
        self.attention = AttentionSystem(profile)

        # Autonomous means the presenter decides his own state. Off, the state
        # is whatever an external caller last set - which is what the eventual
        # content pipeline will want, and what the existing tests assume.
        self.autonomous = autonomous
        self.states = StateScheduler(start=state.value) if autonomous else None
        self.memory = BehaviorMemory()

        self._arousal = OrnsteinUhlenbeck.from_amplitude(
            profile.arousal_amplitude, profile.arousal_time
        )
        # Leaky accumulator; charged on voluntary movement, decays with a ~3 s
        # time constant.
        self._motion_budget = 0.0
        self._frame_interval = 1.0 / 30.0
        self._events: list[BehaviorEvent] = []
        self._recorded = 0
        self._pose = AvatarPose()

    # -- external control ---------------------------------------------------
    def set_state(self, state: BehaviorState | str) -> None:
        """Switch behaviour state.

        Part of the interface Developer A's pipeline will eventually drive.
        State affects rates only; it never causes a discontinuity in pose,
        because every subsystem interpolates from wherever it currently is.
        """
        if isinstance(state, str):
            state = BehaviorState(state)
        if state != self.state:
            self._events.append(
                BehaviorEvent(
                    time=self.now,
                    kind="state_change",
                    detail=f"{self.state.value} -> {state.value}",
                )
            )
        self.state = state
        if self.states is not None and self.states.state != state.value:
            self.states.adopt(state.value, self.now, self.rng)

    def set_profile(self, profile: MotionProfile | str) -> None:
        if isinstance(profile, str):
            profile = PROFILES[profile]
        self.profile = profile

    # -- main loop ----------------------------------------------------------
    def update(self, dt: float) -> AvatarPose:
        """Advance by `dt` seconds and return the pose for this instant."""
        # A long stall - a GPU hiccup, a debugger pause - must not be
        # integrated as one enormous step, which would teleport every
        # subsystem. Clamp and let the engine simply lose that time.
        dt = clamp(dt, 0.0, 0.25)
        self.now += dt
        self.stats.frames += 1
        self.stats.elapsed = self.now

        # Smoothed frame interval, for subsystems that must stay perceptible at
        # whatever rate the renderer is actually achieving. Heavily smoothed on
        # purpose: reacting to a single slow frame would make blinks jitter in
        # length.
        if dt > 0.0:
            self._frame_interval += (dt - self._frame_interval) * min(
                1.0, dt / 0.75
            )

        # The presenter's own state changes before anything reads it, so a
        # transition takes effect on the frame it happens rather than the next.
        if self.states is not None:
            changed = self.states.update(self.now, self.rng)
            if changed is not None:
                self.set_state(BehaviorState(changed))

        arousal = clamp(self._arousal.step(dt, self.rng), -1.0, 1.0)

        # Decay the motion budget before this frame's decisions.
        self._motion_budget *= pow(0.5, dt / 3.0)

        mod = STATE_MODULATION.get(self.state, StateModulation())
        # The budget is passed through as a fire-time gate rather than folded
        # into `mod.stillness`. Folding it in was measurably wrong: an interval
        # sampled immediately after a movement - which is exactly when every
        # interval is sampled - baked in that moment's peak suppression and
        # stayed stretched for its whole duration, dragging the head-move rate
        # to a third of the profile's intent.
        suppression = 1.0 + 2.4 * self._motion_budget

        pose = AvatarPose()
        pose.state = self.state.value

        drives = Drives(
            rng=self.rng,
            profile=self.profile,
            state=self.state,
            mod=mod,
            arousal=arousal,
            now=self.now,
            dt=dt,
            frame_interval=self._frame_interval,
            time_since_gaze_shift=self.now - self.gaze.last_shift_time,
            time_since_head_move=self.now - self.head.last_move_time,
            motion_in_flight=self.blink.is_blinking or self.head.is_moving,
            suppression=suppression,
            visual_demand=self.attention.visual_demand,
            events=self._events,
        )

        # Attention runs first: it decides where he is looking this frame, and
        # both the gaze and the head read that decision rather than inventing
        # their own.
        self.attention.update(drives)

        before = (
            self.blink.blink_count,
            self.attention.shift_count,
            self.head.move_count,
            self.expression.expression_count,
            self.posture.shift_count,
        )

        # Order matters only for additive channels: head and posture write
        # pose fields that breathing then adds to.
        self.head.update(drives, pose)
        self.posture.update(drives, pose)
        self.breathing.update(drives, pose)
        self.blink.update(drives, pose)
        self.gaze.update(drives, pose, attention=self.attention)
        self.expression.update(drives, pose)

        # The head's share of a gaze shift, added on top of whatever the head
        # system is doing on its own. Kept additive rather than replacing the
        # head system so an attention-driven turn and an idle postural
        # adjustment can coexist, which is what happens in a real neck.
        self.idle_head = (pose.yaw, pose.pitch, pose.roll)
        pose.yaw += self.attention.head_yaw
        pose.pitch += self.attention.head_pitch

        # Anatomy last, over everything. Principle 5: generated motion is a
        # proposal, the constraint stage decides what is physically possible.
        apply_constraints(pose)

        after = (
            self.blink.blink_count,
            self.attention.shift_count,
            self.head.move_count,
            self.expression.expression_count,
            self.posture.shift_count,
        )

        # Charge the budget for voluntary movements only. A blink is not a
        # voluntary movement and must not make the avatar hold still
        # afterwards; a head turn is and should.
        if after[1] > before[1]:
            self._motion_budget += 0.35
        if after[2] > before[2]:
            self._motion_budget += 0.75
        if after[3] > before[3]:
            self._motion_budget += 0.4
        if after[4] > before[4]:
            self._motion_budget += 0.6
        self._motion_budget = min(self._motion_budget, 2.5)

        # Record only events not yet seen.
        #
        # The first version re-recorded the last eight events every frame, at
        # 30 Hz. Every event therefore entered the memory dozens of times in a
        # fixed order, and the repetition detector duly reported enormous
        # n-gram excess - 125x on sequences that the behaviour never actually
        # produced. The loop was in the instrumentation, and it was strong
        # enough to survive turning the anti-repetition terms off entirely,
        # which is what gave it away.
        while self._recorded < len(self._events):
            ev = self._events[self._recorded]
            self.memory.record(ev.time, ev.kind, ev.detail)
            self._recorded += 1

        self.stats.blinks = self.blink.blink_count
        self.stats.saccades = self.attention.shift_count
        self.stats.microsaccades = self.gaze.microsaccade_count
        self.stats.head_moves = self.head.move_count
        self.stats.expressions = self.expression.expression_count
        self.stats.posture_shifts = self.posture.shift_count
        self.stats.breaths = self.breathing.breath_count

        self._pose = pose
        return pose

    # -- introspection ------------------------------------------------------
    @property
    def pose(self) -> AvatarPose:
        return self._pose

    @property
    def arousal(self) -> float:
        return self._arousal.value

    @property
    def motion_budget(self) -> float:
        return self._motion_budget

    def drain_events(self) -> list[BehaviorEvent]:
        """Take and clear the accumulated events."""
        events, self._events = self._events, []
        self._recorded = 0
        return events

    def peek_events(self) -> list[BehaviorEvent]:
        return list(self._events)
