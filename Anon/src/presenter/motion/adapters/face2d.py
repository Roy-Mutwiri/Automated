"""Adapter: canonical motion state -> the 2D LivePortrait face renderer.

This adapter throws most of the body away, and that is correct rather than
regrettable. The renderer animates a face crop pasted into a static plate; it
has no torso to move. The adapter's job is to answer honestly what *can* be
shown of a full-body motion state, not to invent a channel for everything.

## The consequence of removing head-scale breathing

There is one worth stating plainly, because it makes this renderer's output
temporarily worse.

Breathing used to be written into `pose.scale` - the head grew and shrank. That
was the only channel available, and it was wrong anatomy. Now breathing lives in
the rib cage, and **a rib cage is not inside the animated crop**. The torso in
the 2D render is a photograph; it cannot breathe.

So through this adapter, breathing produces almost nothing: a residual head
pitch of a few hundredths of a degree, which is what a real breath actually does
to a head. The 2D clip is measurably less alive than it was, and the fix is not
to put head scaling back - it is the body rig, which is why the body work is
ordered first.

## What does survive

Head orientation, which accumulates the *whole* chain. The renderer sees only a
head, but the torso's contribution to where that head points is real and must
not be dropped: a presenter who leans forward and turns his chest is pointing
his face somewhere different, and `head_world_*` sums pelvis through skull to
get there.
"""

from __future__ import annotations

from ...types import AvatarPose
from ..state import HumanMotionState

__all__ = ["to_avatar_pose", "GAZE_UNITS_PER_DEG"]

# Matches the measured expression calibration: gaze_x = +-0.42 puts the irises
# at their natural lateral limit, which is about 35 degrees of eccentricity.
GAZE_UNITS_PER_DEG = 0.42 / 35.0


def to_avatar_pose(motion: HumanMotionState) -> AvatarPose:
    """Project a full-body motion state onto what a face renderer can show."""
    pose = AvatarPose()

    # Whole-chain orientation. The renderer only has a head, but everything
    # below it still decides where that head points.
    pose.yaw = motion.head_world_yaw()
    pose.pitch = motion.head_world_pitch()
    pose.roll = motion.head_world_roll()

    # Eyes are stored relative to the head, which is exactly what the renderer
    # wants: it drives irises inside a face it is already orienting.
    pose.gaze_x = 0.5 * (motion.eye_l.ry + motion.eye_r.ry) * GAZE_UNITS_PER_DEG
    pose.gaze_y = -0.5 * (motion.eye_l.rx + motion.eye_r.rx) * GAZE_UNITS_PER_DEG

    f = motion.face
    pose.eye_open_l = f.eye_open_l
    pose.eye_open_r = f.eye_open_r
    pose.brow_l = f.brow_outer_l + 0.55 * f.brow_inner
    pose.brow_r = f.brow_outer_r + 0.55 * f.brow_inner
    pose.brow_furrow = f.brow_furrow
    pose.mouth_corner_l = f.mouth_corner_l
    pose.mouth_corner_r = f.mouth_corner_r
    pose.squint_l = f.eye_squint_l
    pose.squint_r = f.eye_squint_r
    pose.cheek = 0.5 * (f.cheek_l + f.cheek_r)
    pose.mouth_open = f.jaw
    pose.jaw = f.jaw

    # Deliberately NOT written from breathing. `scale` is a framing control on
    # this renderer, and breathing does not change the size of a head.
    pose.scale = 1.0
    pose.tx = 0.0
    pose.ty = 0.0

    pose.state = motion.behavior_state
    pose.breathing_phase = motion.breathing.phase
    return pose
