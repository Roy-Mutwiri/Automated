"""Behaviour states and motion profiles.

Two orthogonal concepts live here and they are easy to confuse:

* **State** is what the avatar is doing *right now* - listening, thinking,
  about to speak. It changes on the scale of seconds and will eventually be
  driven by Developer A's pipeline.
* **Profile** is the persistent temperament of this particular presenter -
  calm, energetic, focused. It changes rarely or never.

Every numeric knob is a multiplier or a base rate, not a hardcoded constant
buried in a subsystem, so that tuning realism is a config edit rather than a
code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

__all__ = ["BehaviorState", "StateModulation", "MotionProfile", "PROFILES", "STATE_MODULATION"]


class BehaviorState(str, Enum):
    """States the engine can be in.

    Only the idle family is exercised by the current milestone; the speech
    states exist so the interface is stable when audio arrives, and so the
    scheduler's handling of them can be reasoned about now rather than
    retrofitted.
    """

    IDLE_ATTENTIVE = "IDLE_ATTENTIVE"
    IDLE_RELAXED = "IDLE_RELAXED"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    PRE_SPEECH = "PRE_SPEECH"
    SPEAKING = "SPEAKING"
    POST_SPEECH = "POST_SPEECH"
    READING = "READING"
    FOCUSED = "FOCUSED"
    MILD_POSITIVE = "MILD_POSITIVE"
    MILD_CONCERN = "MILD_CONCERN"


@dataclass(frozen=True)
class StateModulation:
    """How a state scales the profile's baseline rates.

    All values multiply the profile baseline; 1.0 means "unchanged". Keeping
    these as multipliers rather than absolute rates means a profile change
    propagates correctly through every state instead of silently overriding it.
    """

    blink_rate: float = 1.0
    gaze_rate: float = 1.0
    head_rate: float = 1.0
    expression_rate: float = 1.0
    breathing_rate: float = 1.0
    # Bias applied to gaze target selection: positive keeps the eyes on the
    # lens, negative lets them wander.
    camera_affinity: float = 0.0
    # Downward gaze bias, used by thinking/reading where people look away and
    # down rather than up.
    downward_bias: float = 0.0
    # Overall damping on discretionary motion. Above 1.0 means "hold still".
    stillness: float = 1.0


# Blink-rate multipliers below follow the behavioural literature: conversation
# roughly doubles the resting rate, sustained focus and reading roughly halve
# it. See docs/human_behavior.md for the sources behind each number.
STATE_MODULATION: dict[BehaviorState, StateModulation] = {
    BehaviorState.IDLE_ATTENTIVE: StateModulation(
        camera_affinity=0.55,
    ),
    BehaviorState.IDLE_RELAXED: StateModulation(
        blink_rate=1.15,
        gaze_rate=1.35,
        head_rate=1.2,
        camera_affinity=0.2,
        stillness=0.85,
    ),
    BehaviorState.LISTENING: StateModulation(
        blink_rate=1.1,
        gaze_rate=0.8,
        head_rate=0.9,
        expression_rate=1.2,
        camera_affinity=0.75,
        stillness=1.1,
    ),
    # Looking away is the signature of thinking; people break eye contact to
    # reduce visual load while retrieving. Blink rate rises with cognitive
    # load, gaze goes down and to one side.
    BehaviorState.THINKING: StateModulation(
        blink_rate=1.25,
        gaze_rate=1.4,
        head_rate=0.8,
        camera_affinity=-0.5,
        downward_bias=0.45,
        stillness=1.05,
    ),
    BehaviorState.PRE_SPEECH: StateModulation(
        blink_rate=0.75,
        gaze_rate=0.5,
        head_rate=0.7,
        expression_rate=1.3,
        camera_affinity=0.9,
        stillness=1.15,
    ),
    BehaviorState.SPEAKING: StateModulation(
        blink_rate=1.9,
        gaze_rate=1.1,
        head_rate=1.5,
        expression_rate=1.4,
        breathing_rate=1.25,
        camera_affinity=0.6,
        stillness=0.8,
    ),
    BehaviorState.POST_SPEECH: StateModulation(
        blink_rate=1.4,
        gaze_rate=1.2,
        head_rate=1.0,
        camera_affinity=0.4,
        stillness=0.9,
    ),
    BehaviorState.READING: StateModulation(
        blink_rate=0.45,
        gaze_rate=1.6,
        head_rate=0.5,
        camera_affinity=-0.8,
        downward_bias=0.8,
        stillness=1.2,
    ),
    BehaviorState.FOCUSED: StateModulation(
        blink_rate=0.5,
        gaze_rate=0.45,
        head_rate=0.4,
        expression_rate=0.8,
        camera_affinity=0.85,
        stillness=1.35,
    ),
    BehaviorState.MILD_POSITIVE: StateModulation(
        blink_rate=1.1,
        expression_rate=1.5,
        camera_affinity=0.6,
        stillness=0.95,
    ),
    BehaviorState.MILD_CONCERN: StateModulation(
        blink_rate=0.9,
        gaze_rate=0.9,
        expression_rate=1.3,
        camera_affinity=0.5,
        stillness=1.1,
    ),
}


@dataclass
class MotionProfile:
    """A presenter's persistent motion temperament.

    Amplitudes are in the units of `AvatarPose`: degrees for head rotation,
    normalised -1..1 for gaze, 0..1 for facial activations.
    """

    name: str = "PRESENTER_CALM"

    # -- blinking ----------------------------------------------------------
    # Median seconds between blinks. 4.0 s is ~15 blinks/min, which sits
    # between the reported resting rate (~12-20/min) and the reduced rate seen
    # under camera/screen attention. Deliberately not the 26-32/min figure
    # reported for active conversation - that is the SPEAKING state's job.
    blink_median_interval: float = 4.0
    blink_interval_shape: float = 0.55   # log-normal sigma; tail length
    blink_min_interval: float = 0.9      # physiological refractory floor
    blink_max_interval: float = 22.0     # forced blink; eyes do not stay open
    # Spontaneous blinks are reported anywhere from 100 to 400 ms. 145 ms sat
    # at the fast end and, sampled at a low render rate, produced a
    # single-frame flash rather than a movement. 0.21 s is mid-range, still
    # physiological, and survives sampling far better.
    blink_duration_mean: float = 0.21    # s, lid closed-to-open
    blink_duration_sigma: float = 0.04
    # Minimum number of *rendered frames* a blink must span to read as motion
    # rather than as a jump. See the note in blinking.py: this is a sampling
    # accommodation for low frame rates, and stops binding above ~25 FPS.
    blink_min_frames: float = 4.5
    blink_max_duration: float = 0.40     # physiological ceiling
    # How much the brow follows the lid down. Orbicularis oculi wraps the whole
    # socket, so an isolated eyelid reads as a shutter, not a blink.
    blink_brow_coupling: float = 0.18
    double_blink_probability: float = 0.09
    blink_asymmetry: float = 0.06        # per-eye timing/closure difference
    blink_partial_probability: float = 0.12  # incomplete closure
    blink_close_fraction: float = 0.36   # portion of blink spent closing

    # -- gaze ---------------------------------------------------------------
    microsaccade_rate: float = 1.4       # Hz; literature says 1-2/s
    microsaccade_amplitude: float = 0.018
    saccade_median_interval: float = 7.0  # between voluntary gaze shifts
    saccade_interval_shape: float = 0.62
    saccade_amplitude: float = 0.16
    saccade_amplitude_sigma: float = 0.09
    saccade_max_amplitude: float = 0.42
    saccade_duration: float = 0.055      # ballistic; barely visible
    gaze_drift_amplitude: float = 0.012  # slow ocular drift between saccades
    gaze_drift_time: float = 1.1
    gaze_return_probability: float = 0.62  # chance a saccade returns to lens

    # -- head ---------------------------------------------------------------
    head_median_interval: float = 15.0   # between voluntary head adjustments
    head_interval_shape: float = 0.7
    head_yaw_amplitude: float = 1.5      # degrees, per voluntary move
    head_pitch_amplitude: float = 1.0
    head_roll_amplitude: float = 0.7
    head_move_duration: float = 0.75
    head_max_yaw: float = 6.0            # hard anatomical-plausibility limits
    head_max_pitch: float = 4.5
    head_max_roll: float = 3.5
    # Involuntary residual sway, always present, never a deliberate movement.
    head_sway_amplitude: float = 0.22    # degrees, stationary sigma
    head_sway_time: float = 2.8          # correlation time, seconds

    # -- breathing -----------------------------------------------------------
    # Quiet seated respiration is 12-18 breaths/min -> 3.3-5.0 s per cycle.
    breath_period: float = 4.1
    breath_period_sigma: float = 0.45
    breath_scale_amount: float = 0.0028  # fraction; must be near-subliminal
    breath_ty_amount: float = 0.0032
    breath_pitch_amount: float = 0.12    # degrees
    breath_inhale_fraction: float = 0.42  # inhale is shorter than exhale

    # -- expression ----------------------------------------------------------
    expression_median_interval: float = 21.0
    expression_interval_shape: float = 0.65
    expression_strength: float = 0.16    # global scale; conservative on purpose
    expression_duration: float = 1.5
    brow_asymmetry: float = 0.14         # fraction of difference between sides

    # -- posture -------------------------------------------------------------
    posture_amplitude: float = 0.004     # translation, fractions of face width
    posture_time: float = 14.0           # very slow
    posture_shift_median_interval: float = 55.0

    # -- global --------------------------------------------------------------
    # Multiplies every discretionary behaviour's probability. The single knob
    # to reach for when the avatar feels busy.
    activity: float = 1.0
    # Slow-varying arousal drives the stillness/liveliness texture. Without
    # this the avatar has a constant statistical density of movement, which
    # over minutes reads as mechanical even when each individual motion is
    # plausible.
    arousal_amplitude: float = 0.28
    arousal_time: float = 26.0

    def scaled(self, **overrides) -> "MotionProfile":
        return replace(self, **overrides)


PRESENTER_CALM = MotionProfile(name="PRESENTER_CALM")

PRESENTER_ENERGETIC = MotionProfile(
    name="PRESENTER_ENERGETIC",
    blink_median_interval=3.4,
    saccade_median_interval=5.8,
    saccade_amplitude=0.21,
    head_median_interval=11.5,
    head_yaw_amplitude=2.4,
    head_pitch_amplitude=1.6,
    head_roll_amplitude=1.1,
    head_max_yaw=8.5,
    head_max_pitch=6.0,
    head_sway_amplitude=0.32,
    expression_median_interval=16.5,
    expression_strength=0.26,
    posture_shift_median_interval=45.0,
    activity=1.05,
    arousal_amplitude=0.34,
)

PRESENTER_FOCUSED = MotionProfile(
    name="PRESENTER_FOCUSED",
    blink_median_interval=6.2,
    blink_interval_shape=0.6,
    saccade_median_interval=5.5,
    saccade_amplitude=0.10,
    gaze_return_probability=0.78,
    head_median_interval=12.0,
    head_yaw_amplitude=0.9,
    head_pitch_amplitude=0.7,
    head_roll_amplitude=0.4,
    head_max_yaw=4.0,
    head_max_pitch=3.0,
    head_sway_amplitude=0.16,
    expression_median_interval=16.0,
    expression_strength=0.12,
    posture_shift_median_interval=60.0,
    activity=0.78,
    arousal_amplitude=0.2,
)

PROFILES: dict[str, MotionProfile] = {
    p.name: p for p in (PRESENTER_CALM, PRESENTER_ENERGETIC, PRESENTER_FOCUSED)
}
