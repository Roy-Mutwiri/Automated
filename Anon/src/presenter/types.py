"""The contract between the behaviour engine and any renderer.

`AvatarPose` is the single interface across the seam in the pipeline. The
behaviour engine knows nothing about LivePortrait, warping fields or implicit
keypoints; the renderer knows nothing about blink probability or saccades. That
separation is what lets the rendering backend be swapped - or benchmarked
against an alternative - without touching a line of behaviour code.

Units are chosen to be human-readable rather than convenient for any particular
model, so that a debug overlay showing "yaw = -2.1 deg" means something. The
renderer adapter is responsible for mapping into whatever the model wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["AvatarPose", "BehaviorEvent"]


@dataclass
class AvatarPose:
    """A complete description of the avatar at one instant.

    All fields are absolute, not deltas: the engine always emits the full
    intended pose, so a dropped or late frame cannot accumulate error. This is
    deliberate - incremental pose updates drift, and drift on a face reads as
    the identity slowly changing, which the brief lists as unacceptable.
    """

    # -- head orientation, degrees, relative to the source portrait ---------
    # Ranges are held small on purpose. A seated presenter facing a camera
    # moves the head far less than intuition suggests.
    yaw: float = 0.0    # +right
    pitch: float = 0.0  # +up
    roll: float = 0.0   # +clockwise from the viewer's side

    # -- head/torso translation, in fractions of face width ----------------
    tx: float = 0.0
    ty: float = 0.0

    # -- overall scale, 1.0 = source framing --------------------------------
    # Breathing rides mostly here and on ty: the chest rising pushes the head
    # up and very slightly toward the camera.
    scale: float = 1.0

    # -- eyelids, 0 = fully closed, 1 = neutral open ------------------------
    # Separate per eye so blinks can carry a small asymmetry.
    eye_open_l: float = 1.0
    eye_open_r: float = 1.0

    # -- gaze direction, normalised, roughly -1..1 --------------------------
    # (0, 0) is looking down the lens. Kept separate from head yaw/pitch
    # because eyes and head move on very different time scales.
    gaze_x: float = 0.0  # +right
    gaze_y: float = 0.0  # +up

    # -- brows, 0 = neutral, + = raised -------------------------------------
    brow_l: float = 0.0
    brow_r: float = 0.0
    brow_furrow: float = 0.0  # medial pull, the "slight concentration" tell

    # -- mid-face and mouth, all near-zero at rest --------------------------
    squint_l: float = 0.0
    squint_r: float = 0.0
    cheek: float = 0.0
    mouth_corner_l: float = 0.0
    mouth_corner_r: float = 0.0
    mouth_open: float = 0.0   # reserved for the lip-sync stage
    jaw: float = 0.0

    # -- diagnostics --------------------------------------------------------
    # Not rendered; carried so the debug overlay and the timeline analyser can
    # report what the engine believed it was doing on this frame.
    state: str = "IDLE_ATTENTIVE"
    breathing_phase: float = 0.0  # 0..1 through the current breath

    def copy(self) -> "AvatarPose":
        return AvatarPose(**vars(self))


@dataclass
class BehaviorEvent:
    """A discrete thing the engine decided to do, for the timeline log.

    The 30-minute analysis in tools/behavior_timeline.py consumes these. Any
    behaviour that is not logged here is invisible to the repetition detector,
    so subsystems should emit an event for every voluntary action.
    """

    time: float
    kind: str
    detail: str = ""
    magnitude: float = 0.0
    metadata: dict = field(default_factory=dict)

    def timestamp(self) -> str:
        minutes, seconds = divmod(self.time, 60.0)
        return f"{int(minutes):02d}:{seconds:06.3f}"

    def __str__(self) -> str:
        line = f"{self.timestamp()}  {self.kind}"
        if self.detail:
            line += f"  {self.detail}"
        return line
