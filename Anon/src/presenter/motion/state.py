"""The canonical human motion state. One per timestamp, renderer-independent.

This is the contract between "how the human moves" and "what the human looks
like", and it exists because those two questions belong to different terminals
and must not be able to contaminate each other.

**Nothing in this module may import a renderer, a rig, a camera or an identity.**
If it ever needs to, the boundary has been drawn in the wrong place.

## Why a joint state rather than an `AvatarPose`

`AvatarPose` describes what the current 2D face renderer can consume: yaw,
pitch, roll, two eyelids, a scale. It is a *renderer's* vocabulary, and building
behaviour on top of it produced exactly the deformity the last review found -
breathing implemented as head scaling, because head scale was the only channel
available. That is not a shortcut, it is physically wrong: breathing does not
change the size of a head.

A joint state has somewhere to put a rib cage. The behaviour engine writes what
the *body* is doing; each adapter then answers "what can I show of that?".
The 2D face adapter throws almost all of it away, which is fine and honest - it
is a face renderer. The MPFB adapter uses the rest. When the final human
arrives, a third adapter is written and **the behaviour engine is not touched**.

## Conventions, fixed once

* Rotations are **degrees**, local to the parent joint, applied XYZ.
* `+rx` pitches forward (nod down), `+ry` yaws toward the character's left,
  `+rz` rolls the character's right ear toward the right shoulder.
* Left/right are the **character's**, never the viewer's. The rig data uses the
  same convention (`joint-l-*` is at +X), so there is one place to get this
  wrong and it is here.
* Everything is absolute, never a delta. A dropped frame must not accumulate.

## What is deliberately not here

No bone lengths, no rest pose, no mesh, no blendshape indices, no latent
dimensions. Those are rig facts and they live in adapters. A behaviour author
should never need to know that this particular skeleton numbers its spine from
the top down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "JointRotation", "HandPose", "FaceParameters", "BreathingState",
    "PostureState", "AttentionState", "EmotionState", "HumanMotionState",
]


@dataclass
class JointRotation:
    """Local rotation of one joint, in degrees."""

    rx: float = 0.0     # + = forward / down
    ry: float = 0.0     # + = toward the character's left
    rz: float = 0.0     # + = roll right ear toward right shoulder

    def __iadd__(self, other: "JointRotation") -> "JointRotation":
        self.rx += other.rx
        self.ry += other.ry
        self.rz += other.rz
        return self

    def scaled(self, k: float) -> "JointRotation":
        return JointRotation(self.rx * k, self.ry * k, self.rz * k)

    def magnitude(self) -> float:
        return (self.rx ** 2 + self.ry ** 2 + self.rz ** 2) ** 0.5


@dataclass
class HandPose:
    """One hand, summarised.

    Per-finger curl rather than twenty joint angles: at the stage this project
    is at, nothing generates individually articulated fingers and a state that
    can express more than the behaviour can decide is a lie about the system's
    capability. `curl` is 0 (straight) to 1 (fully flexed) per finger, thumb
    first. The adapter distributes each curl across that finger's joints.
    """

    # Resting curls for a hand lying on a desk: never straight, never equal.
    # Perfectly straight fingers are the clearest mannequin tell there is.
    curl: list[float] = field(default_factory=lambda: [0.26, 0.34, 0.31, 0.36, 0.42])
    spread: float = 0.0          # abduction, + = fingers apart
    # What the hand is resting on. The adapter turns this into an IK target;
    # None means the hand is posed by its joint chain alone.
    contact: str | None = None
    contact_weight: float = 1.0  # 0 releases IK, 1 pins to the contact
    # Offset from the contact target, in rig units. This is how a hand moves
    # without leaving the surface - nudging a mouse, shifting a resting palm.
    # Measured at 10 px of total wrist travel at 1080p before this existed,
    # which is to say the hands were static.
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class FaceParameters:
    """Semantic facial controls. Not latent dimensions, not blendshape indices.

    Every value is 0..1 unless noted. An adapter maps these onto whatever its
    renderer actually has - for the 2D path that is a calibrated set of
    LivePortrait latent dimensions, for a rig it would be blendshapes.
    """

    eye_open_l: float = 1.0
    eye_open_r: float = 1.0

    brow_inner: float = 0.0          # medial raise; + = up
    brow_outer_l: float = 0.0
    brow_outer_r: float = 0.0
    brow_furrow: float = 0.0

    eye_squint_l: float = 0.0        # lower-lid raise: the Duchenne component
    eye_squint_r: float = 0.0

    cheek_l: float = 0.0             # cheek raise
    cheek_r: float = 0.0

    mouth_corner_l: float = 0.0      # + = up (smile), - = down
    mouth_corner_r: float = 0.0
    upper_lip: float = 0.0
    lower_lip: float = 0.0
    jaw: float = 0.0                 # jaw opening, reserved for speech

    nose_wrinkle: float = 0.0


@dataclass
class BreathingState:
    """Where the breath is, not what it looks like.

    Adapters decide how much rib cage, chest, clavicle and shoulder each of
    these produces. Keeping the *phase* here and the *coupling* in the adapter
    is what stops breathing from being reinvented per renderer.
    """

    phase: float = 0.0               # 0..1 through the current cycle
    depth: float = 1.0               # multiplier on amplitude, slowly varying
    rate: float = 14.6               # breaths per minute, slowly varying
    # Signed drive: + = inhaled/expanded, - = fully exhaled. This is the value
    # adapters actually use; `phase` is for diagnostics and coupling.
    drive: float = 0.0


@dataclass
class PostureState:
    """A point on a continuum, never one of four clips."""

    engagement: float = 0.0          # -1 relaxed back .. +1 forward focus
    lean: float = 0.0                # degrees of forward torso lean
    settle: float = 0.0              # 0..1 how far back into the chair
    # 0 = back clear of the rest, 1 = leaning on it. Derived from where the
    # spine actually ends up, not asserted, so it cannot claim contact while
    # the torso is somewhere else.
    back_contact: float = 0.0
    # Persistent, person-specific asymmetry. Set once from the persona and not
    # re-randomised; a body that is asymmetric differently every second reads
    # as noise rather than as a person.
    shoulder_drop_l: float = 0.0
    shoulder_drop_r: float = 0.0


@dataclass
class AttentionState:
    """Where the human is attending, in the ROOM.

    Azimuth and elevation are degrees from the character's neutral forward, in
    world space. No camera appears here, which is what makes a camera cut
    incapable of changing where he is looking.
    """

    target: str = "LENS"
    azimuth: float = 0.0
    elevation: float = 0.0
    # Split of the current eccentricity between eyes and head, 0..1.
    head_share: float = 0.0
    shifting: bool = False
    visual_demand: float = 0.3


@dataclass
class EmotionState:
    """Continuous affect. Categorical labels are a convenience on top."""

    valence: float = 0.0             # -1 negative .. +1 positive
    arousal: float = 0.0             # -1 subdued .. +1 activated
    confidence: float = 0.0
    engagement: float = 0.0
    label: str = "NEUTRAL"


@dataclass
class HumanMotionState:
    """Everything the body is doing at one instant."""

    timestamp: float = 0.0

    # Root: world placement. Present so a rig can be moved without the
    # behaviour engine knowing about the room; the 2D adapter ignores it.
    root_x: float = 0.0
    root_y: float = 0.0
    root_z: float = 0.0
    root_yaw: float = 0.0

    pelvis: JointRotation = field(default_factory=JointRotation)
    spine_lower: JointRotation = field(default_factory=JointRotation)
    spine_mid: JointRotation = field(default_factory=JointRotation)
    chest: JointRotation = field(default_factory=JointRotation)

    clavicle_l: JointRotation = field(default_factory=JointRotation)
    clavicle_r: JointRotation = field(default_factory=JointRotation)
    shoulder_l: JointRotation = field(default_factory=JointRotation)
    shoulder_r: JointRotation = field(default_factory=JointRotation)
    elbow_l: JointRotation = field(default_factory=JointRotation)
    elbow_r: JointRotation = field(default_factory=JointRotation)
    wrist_l: JointRotation = field(default_factory=JointRotation)
    wrist_r: JointRotation = field(default_factory=JointRotation)

    hand_l: HandPose = field(default_factory=HandPose)
    hand_r: HandPose = field(default_factory=HandPose)

    neck: JointRotation = field(default_factory=JointRotation)
    head: JointRotation = field(default_factory=JointRotation)

    # Eye rotations, degrees, relative to the head. NOT relative to the room -
    # the room-space direction is `attention`, and the difference between the
    # two is exactly what the head has already covered.
    eye_l: JointRotation = field(default_factory=JointRotation)
    eye_r: JointRotation = field(default_factory=JointRotation)

    face: FaceParameters = field(default_factory=FaceParameters)
    breathing: BreathingState = field(default_factory=BreathingState)
    posture: PostureState = field(default_factory=PostureState)
    attention: AttentionState = field(default_factory=AttentionState)
    emotion: EmotionState = field(default_factory=EmotionState)

    # Diagnostics. Never rendered.
    behavior_state: str = "IDLE_ATTENTIVE"

    # -- convenience ---------------------------------------------------------
    def head_world_yaw(self) -> float:
        """Total yaw of the head in the room: the whole chain, summed.

        Adapters that only have a head - the 2D face renderer - need this,
        because for them the neck and torso are invisible but their
        contribution to where the face points is not.
        """
        return (self.root_yaw + self.pelvis.ry + self.spine_lower.ry
                + self.spine_mid.ry + self.chest.ry + self.neck.ry
                + self.head.ry)

    def head_world_pitch(self) -> float:
        return (self.pelvis.rx + self.spine_lower.rx + self.spine_mid.rx
                + self.chest.rx + self.neck.rx + self.head.rx)

    def head_world_roll(self) -> float:
        return (self.pelvis.rz + self.spine_lower.rz + self.spine_mid.rz
                + self.chest.rz + self.neck.rz + self.head.rz)

    def joints(self) -> dict[str, JointRotation]:
        """Named joint rotations, for adapters and debug overlays."""
        return {
            "pelvis": self.pelvis, "spine_lower": self.spine_lower,
            "spine_mid": self.spine_mid, "chest": self.chest,
            "clavicle_l": self.clavicle_l, "clavicle_r": self.clavicle_r,
            "shoulder_l": self.shoulder_l, "shoulder_r": self.shoulder_r,
            "elbow_l": self.elbow_l, "elbow_r": self.elbow_r,
            "wrist_l": self.wrist_l, "wrist_r": self.wrist_r,
            "neck": self.neck, "head": self.head,
            "eye_l": self.eye_l, "eye_r": self.eye_r,
        }
