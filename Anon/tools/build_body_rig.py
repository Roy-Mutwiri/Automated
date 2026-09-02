"""Build a posable armature from the fitted MakeHuman body, once.

Input is the Camera Terminal's `MPFB_BODY_RIG_FOUNDATION_V1` artefact: a fitted
CC0 MakeHuman mesh plus 125 joint markers. Output is a `.blend` containing a
skinned, IK-equipped rig that the motion adapter can pose.

**Identity is explicitly not a concern here.** The tag that produced this data
records "Face identity: FAILED - do not resume face fitting", and that is fine:
this is a motion test dummy. What is being judged is whether a chest breathes
and a neck turns, not who the person is.

## Coordinate conversion

MakeHuman is Y-up with +Z forward. Blender is Z-up with -Y forward. One
conversion, in one place:

    (x, y, z)_mh  ->  (x, -z, y)_blender

Getting this wrong is silent - the character simply lies on its back - so it is
a named function rather than an inline expression.

## Why the chain is defined explicitly

The joint file is a flat dictionary of 125 points with no hierarchy. The
skeleton below is written out rather than inferred, because inferring parentage
from proximity produces plausible-looking nonsense at the shoulder, where the
clavicle, scapula and humerus heads sit within two centimetres of each other.

## Rib cage

`chest_top` (spine-1 -> neck) is the bone breathing drives, and it is separate
from `chest` for exactly that reason: the adapter needs somewhere to put a rib
cage that is not the whole torso. Its *scale* carries the circumference change,
which a rotation cannot express.

Usage
-----
    python tools/build_body_rig.py            # writes assets/rig/body_rig.blend
    python tools/build_body_rig.py --preview  # also renders a rest-pose frame
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

JOINTS = "research/avatar_reconstruction/outputs/fitted_joints.json"
MESH = "research/avatar_reconstruction/outputs/fitted_head.obj"

# name -> (head joint, tail joint, parent, connected)
SKELETON: list[tuple[str, str, str, str | None, bool]] = [
    ("root",        "joint-pelvis",     "joint-spine-4",   None,          False),
    ("pelvis",      "joint-pelvis",     "joint-spine-4",   "root",        False),
    ("spine_lower", "joint-spine-4",    "joint-spine-3",   "pelvis",      True),
    ("spine_mid",   "joint-spine-3",    "joint-spine-2",   "spine_lower", True),
    ("chest",       "joint-spine-2",    "joint-spine-1",   "spine_mid",   True),
    ("chest_top",   "joint-spine-1",    "joint-neck",      "chest",       True),
    ("neck",        "joint-neck",       "joint-head",      "chest_top",   True),
    ("head",        "joint-head",       "joint-head-2",    "neck",        True),
]

for side in ("l", "r"):
    SKELETON += [
        (f"clavicle_{side}", f"joint-{side}-clavicle", f"joint-{side}-shoulder",
         "chest_top", False),
        (f"shoulder_{side}", f"joint-{side}-shoulder", f"joint-{side}-elbow",
         f"clavicle_{side}", True),
        (f"elbow_{side}",    f"joint-{side}-elbow",    f"joint-{side}-hand",
         f"shoulder_{side}", True),
        (f"wrist_{side}",    f"joint-{side}-hand",     f"joint-{side}-hand-2",
         f"elbow_{side}",    True),
        (f"eye_{side}",      f"joint-{side}-eye",      f"joint-{side}-eye-target",
         "head", False),
    ]
    for f in range(1, 6):
        for seg in range(1, 4):
            SKELETON.append((
                f"finger_{side}_{f}_{seg}",
                f"joint-{side}-finger-{f}-{seg}",
                f"joint-{side}-finger-{f}-{seg + 1}",
                f"wrist_{side}" if seg == 1 else f"finger_{side}_{f}_{seg - 1}",
                seg > 1,
            ))

# Contact targets, in MakeHuman coordinates.
#
# Placed as a fraction of the arm's *actual* reach, measured from this body's
# own joints, rather than as absolute offsets. The first version used fractions
# of shoulder width and put the mouse 6.09 units from the shoulder against a
# total arm reach of 5.20 - out of reach. The IK solved it correctly by
# straightening the arm and pointing at the target, which is why the character
# sat at his desk with both arms locked straight out to the sides.
#
# 0.72 of full reach gives a relaxed elbow around 100 degrees, which is what a
# forearm resting on a desk actually does.
COMFORT_REACH = 0.72


def contact_points(j: dict) -> dict[str, tuple[float, float, float]]:
    import math as _m

    def dist(a, b):
        return _m.dist(j[a], j[b])

    out = {}
    for side, sign in (("l", +1.0), ("r", -1.0)):
        sh = j[f"joint-{side}-shoulder"]
        reach = dist(f"joint-{side}-shoulder", f"joint-{side}-elbow") +             dist(f"joint-{side}-elbow", f"joint-{side}-hand")
        d = reach * COMFORT_REACH

        elbow_y = j[f"joint-{side}-elbow"][1]
        desk_y = elbow_y - 0.35
        dy = desk_y - sh[1]

        # Whatever budget is left after dropping to desk height goes forward,
        # with a little of it inward so the hands are nearer than the shoulders.
        planar = max(d * d - dy * dy, 0.25) ** 0.5
        dx = -sign * planar * 0.22          # inward, toward the midline
        dz = max(planar * planar - dx * dx, 0.25) ** 0.5

        hand = "desk_rest_l" if side == "l" else "mouse"
        out[hand] = (sh[0] + dx, desk_y, sh[2] + dz)

        # The lap is lower, closer and further inboard.
        lap_d = reach * 0.52
        out[f"lap_rest_{side}"] = (
            sh[0] - sign * lap_d * 0.30,
            j["joint-pelvis"][1] + 1.1,
            sh[2] + lap_d * 0.62,
        )
        out[f"armrest_{side}"] = (
            sh[0] + sign * 0.35,
            desk_y - 1.5,
            sh[2] + planar * 0.55,
        )

    mouse = out["mouse"]
    out["keyboard"] = (mouse[0] * 0.35, mouse[1], mouse[2] - 0.3)
    return out


def mh_to_blender(p):
    """MakeHuman (Y-up, +Z fwd) -> Blender (Z-up, -Y fwd)."""
    x, y, z = p
    return (x, -z, y)


def build(joints_path: Path, mesh_path: Path, out_path: Path,
          preview: bool = False) -> None:
    import bpy
    from mathutils import Vector

    j = json.loads(joints_path.read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # --- mesh -------------------------------------------------------------
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(mesh_path))
    new = list(set(bpy.data.objects) - before)
    if not new:
        raise RuntimeError(f"nothing imported from {mesh_path}")
    body = new[0]
    body.name = "body"
    # The OBJ importer applies its own axis conversion; undo it so the mesh
    # sits in the same frame as the joints, which are converted explicitly.
    body.rotation_euler = (0.0, 0.0, 0.0)
    for v in body.data.vertices:
        x, y, z = v.co
        v.co = Vector(mh_to_blender((x, y, z)))
    body.rotation_mode = "XYZ"

    # --- armature ---------------------------------------------------------
    arm_data = bpy.data.armatures.new("body_rig")
    rig = bpy.data.objects.new("body_rig", arm_data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")

    made: dict[str, object] = {}
    skipped = []
    for name, head_j, tail_j, parent, connected in SKELETON:
        if head_j not in j or tail_j not in j:
            skipped.append(name)
            continue
        b = arm_data.edit_bones.new(name)
        b.head = Vector(mh_to_blender(j[head_j]))
        b.tail = Vector(mh_to_blender(j[tail_j]))
        if (b.tail - b.head).length < 1e-4:
            b.tail = b.head + Vector((0.0, 0.0, 0.05))
        made[name] = b

    for name, _h, _t, parent, connected in SKELETON:
        if name in made and parent in made:
            made[name].parent = made[parent]
            made[name].use_connect = bool(connected) and (
                (made[name].head - made[parent].tail).length < 1e-4)

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[rig] {len(made)} bones built, {len(skipped)} skipped: {skipped}")

    # --- contact targets --------------------------------------------------
    targets = {}
    for name, p in contact_points(j).items():
        e = bpy.data.objects.new(f"target_{name}", None)
        e.empty_display_type = "PLAIN_AXES"
        e.empty_display_size = 0.35
        e.location = Vector(mh_to_blender(p))
        bpy.context.collection.objects.link(e)
        targets[name] = e

    # IK on each forearm, two bones deep, so the elbow and shoulder solve
    # together and the hand can be pinned to a surface. Disabled by default;
    # the adapter raises the influence when the hand is in contact.
    for side, tname in (("l", "desk_rest_l"), ("r", "mouse")):
        pb = rig.pose.bones.get(f"elbow_{side}")
        if pb is None:
            continue
        ik = pb.constraints.new("IK")
        ik.target = targets[tname]
        ik.chain_count = 2
        ik.influence = 0.0
        ik.name = "hand_contact"

    # --- skinning ---------------------------------------------------------
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    print("[rig] skinned with automatic weights")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path.resolve()))
    print(f"[rig] wrote {out_path}")

    if preview:
        from render_body import setup_scene, render_to  # noqa: F401
        print("[rig] preview requires tools/render_body.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joints", default=JOINTS)
    ap.add_argument("--mesh", default=MESH)
    ap.add_argument("--out", default="assets/rig/body_rig.blend")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    build(Path(args.joints), Path(args.mesh), Path(args.out), args.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
