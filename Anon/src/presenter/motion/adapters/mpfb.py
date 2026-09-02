"""Adapter: canonical motion state -> the MPFB test rig in Blender.

The second consumer of `HumanMotionState`, and the reason the state exists. It
imports `bpy` and knows about bone names and local axes; the behaviour engine
knows none of that and must never learn it.

## Axis mapping

A Blender bone's local Y runs along its own length, so what "rx" means depends
on which way the bone points. For the spine, neck and head - which point up -
the mapping is direct:

    rx (forward/back)  -> local X
    ry (turn)          -> local Y, the twist along the bone
    rz (side bend)     -> local Z

For the clavicles and upper arms, which point sideways, the same three
rotations land on different local axes, so those bones carry their own entry in
`AXIS_MAP` rather than being special-cased in code.

## Signs

Every entry carries an explicit sign triple. These are **not** derived from
first principles - a bone's roll decides them and the roll comes from the
fitted joint positions, so they are established by posing the rig and looking
at it. Getting one wrong makes the character bend backwards, which is obvious;
getting one subtly wrong makes it bend backwards by three degrees, which is
not, and that is why they are written down rather than inlined.

## Rib cage

Breathing's circumference change is applied as a **scale** on `chest_top`, not
a rotation, because a rotation cannot express a rib cage expanding. It is the
one channel here that the 2D face adapter has no counterpart for at all.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from mathutils import Vector

from ..breathing import RIB_EXPANSION
from ..state import HumanMotionState

__all__ = ["MPFBAdapter"]

# joint in HumanMotionState -> (bone, (axis for rx, ry, rz), (sign, sign, sign))
AXIS_MAP: dict[str, tuple[str, tuple[int, int, int], tuple[float, float, float]]] = {
    "pelvis":      ("pelvis",      (0, 1, 2), (+1.0, +1.0, +1.0)),
    "spine_lower": ("spine_lower", (0, 1, 2), (+1.0, +1.0, +1.0)),
    "spine_mid":   ("spine_mid",   (0, 1, 2), (+1.0, +1.0, +1.0)),
    "chest":       ("chest",       (0, 1, 2), (+1.0, +1.0, +1.0)),
    "neck":        ("neck",        (0, 1, 2), (+1.0, +1.0, +1.0)),
    "head":        ("head",        (0, 1, 2), (+1.0, +1.0, +1.0)),
    # Clavicles run outward along their own Y, so a shrug is a rotation about
    # local X and a forward roll is about local Z.
    "clavicle_l":  ("clavicle_l",  (2, 1, 0), (+1.0, +1.0, +1.0)),
    "clavicle_r":  ("clavicle_r",  (2, 1, 0), (+1.0, +1.0, -1.0)),
    "shoulder_l":  ("shoulder_l",  (2, 1, 0), (+1.0, +1.0, +1.0)),
    "shoulder_r":  ("shoulder_r",  (2, 1, 0), (+1.0, +1.0, -1.0)),
    "elbow_l":     ("elbow_l",     (2, 1, 0), (+1.0, +1.0, +1.0)),
    "elbow_r":     ("elbow_r",     (2, 1, 0), (+1.0, +1.0, -1.0)),
    "wrist_l":     ("wrist_l",     (2, 1, 0), (+1.0, +1.0, +1.0)),
    "wrist_r":     ("wrist_r",     (2, 1, 0), (+1.0, +1.0, -1.0)),
    "eye_l":       ("eye_l",       (0, 2, 1), (+1.0, +1.0, +1.0)),
    "eye_r":       ("eye_r",       (0, 2, 1), (+1.0, +1.0, +1.0)),
}

# The breath's chest rotation is shared between `chest` and `chest_top`; the
# upper bone is the rib cage proper and carries most of it.
CHEST_TOP_SHARE = 0.62

# Finger axes are NOT uniform across the hand and must not be hardcoded.
#
# Measured (tools/rig_axes.py): the thumb and index hinge about local X, while
# the middle, ring and little fingers hinge about local **Z**, with signs that
# differ between hands. A single constant was right for two fingers out of five
# and drove the other three sideways, which is why the hand fanned open instead
# of closing.
#
# The map is loaded from config/rig_axes.json, regenerated whenever the rig is
# rebuilt. There is deliberately no axis index written in this file.
AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
_AXIS_MAP_PATH = Path(__file__).resolve().parents[4] / "config" / "rig_axes.json"


def _load_axis_map() -> dict:
    try:
        return json.loads(_AXIS_MAP_PATH.read_text()).get("axis_map", {})
    except (OSError, ValueError) as exc:
        print(f"[mpfb] no measured axis map ({exc}); fingers will not curl "
              f"correctly - run tools/rig_axes.py")
        return {}

D2R = math.pi / 180.0


class MPFBAdapter:
    """Poses a Blender armature from a canonical motion state."""

    def __init__(self, rig_object) -> None:
        self.rig = rig_object
        for pb in self.rig.pose.bones:
            pb.rotation_mode = "XYZ"
        self._targets = {
            name.replace("target_", ""): obj
            for name, obj in _scene_objects().items()
            if name.startswith("target_")
        }
        self._rest_target = {key: tuple(obj.location)
                             for key, obj in self._targets.items()}
        self._axis_map = _load_axis_map()

    # -- posing --------------------------------------------------------------
    def apply(self, motion: HumanMotionState) -> None:
        pb = self.rig.pose.bones
        joints = motion.joints()

        for name, rot in joints.items():
            entry = AXIS_MAP.get(name)
            if entry is None:
                continue
            bone_name, axes, signs = entry
            bone = pb.get(bone_name)
            if bone is None:
                continue
            values = (rot.rx, rot.ry, rot.rz)
            e = [0.0, 0.0, 0.0]
            for i in range(3):
                e[axes[i]] += values[i] * signs[i] * D2R
            bone.rotation_euler = e

        # Rib cage. The chest rotation is split with chest_top, and the
        # circumference change rides on scale where a rotation cannot reach.
        top = pb.get("chest_top")
        if top is not None:
            drive = (motion.breathing.drive - 0.5) * 2.0 * motion.breathing.depth
            chest = joints["chest"]
            top.rotation_euler = (chest.rx * CHEST_TOP_SHARE * D2R,
                                  chest.ry * CHEST_TOP_SHARE * D2R,
                                  chest.rz * CHEST_TOP_SHARE * D2R)
            k = 1.0 + RIB_EXPANSION * drive
            top.scale = (k, 1.0 + RIB_EXPANSION * 0.35 * drive, k)

        # Root translation: the pelvis sliding in the seat.
        #
        # `pose_bone.location` is in **bone-local** space, not world. The root
        # bone points up the spine, so assigning a world-space delta to it slid
        # the pelvis *along the bone* - settling back lifted the body 4.7 cm off
        # the seat instead of moving it backward. The delta is rotated into the
        # bone's own frame first.
        root = pb.get("root")
        if root is not None:
            world = Vector((motion.root_x, -motion.root_z, motion.root_y))
            basis = root.bone.matrix_local.to_3x3()
            root.location = basis.inverted() @ world

        self._apply_hands(motion)
        self._apply_contacts(motion)

    def _apply_hands(self, motion: HumanMotionState) -> None:
        """Distribute each finger's curl across its three segments.

        The proximal joint takes the most and the distal the least, which is
        how a relaxed hand actually closes. Each bone is driven on *its own*
        measured hinge axis; the segments before the first measured joint
        inherit that finger's axis, since a finger flexes in one plane.
        """
        pb = self.rig.pose.bones
        share = (0.45, 0.33, 0.22)
        for side, hand in (("l", motion.hand_l), ("r", motion.hand_r)):
            for f, curl in enumerate(hand.curl, start=1):
                ref = self._axis_map.get(f"finger_{side}_{f}_2")
                for seg in range(1, 4):
                    name = f"finger_{side}_{f}_{seg}"
                    bone = pb.get(name)
                    if bone is None:
                        continue
                    entry = self._axis_map.get(name) or ref
                    if entry is None:
                        continue
                    idx = AXIS_INDEX[entry["curl_axis"]]
                    sign = entry["curl_sign"]
                    e = [0.0, 0.0, 0.0]
                    e[idx] = curl * share[seg - 1] * 78.0 * D2R * sign
                    if seg == 1 and hand.spread:
                        # Spread is about a perpendicular axis, whichever of the
                        # other two is not the bone's own length.
                        spread_idx = 2 if idx != 2 else 0
                        e[spread_idx] = (hand.spread * 9.0 * D2R
                                         * (1 if side == "l" else -1))
                    bone.rotation_euler = e

    def _apply_contacts(self, motion: HumanMotionState) -> None:
        """Raise or release the hand IK according to what the hand is on."""
        pb = self.rig.pose.bones
        for side, hand in (("l", motion.hand_l), ("r", motion.hand_r)):
            bone = pb.get(f"elbow_{side}")
            if bone is None:
                continue
            con = bone.constraints.get("hand_contact")
            if con is None:
                continue
            con.influence = hand.contact_weight if hand.contact else 0.0
            if hand.contact and hand.contact in self._targets:
                con.target = self._targets[hand.contact]


def _scene_objects():
    import bpy
    return {o.name: o for o in bpy.data.objects}
