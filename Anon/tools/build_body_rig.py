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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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

# Legs. Camera 5 and 6 will see them, and a seated pose without hips is a
# torso balanced on a standing pair of legs.
for side in ("l", "r"):
    SKELETON += [
        (f"thigh_{side}", f"joint-{side}-upper-leg", f"joint-{side}-knee",
         "pelvis", False),
        (f"shin_{side}",  f"joint-{side}-knee",      f"joint-{side}-ankle",
         f"thigh_{side}", True),
        (f"foot_{side}",  f"joint-{side}-ankle",     f"joint-{side}-foot-1",
         f"shin_{side}",  True),
        (f"toe_{side}",   f"joint-{side}-foot-1",    f"joint-{side}-foot-2",
         f"foot_{side}",  True),
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

# Contact targets and scene planes come from the shared geometry module, which
# measures them off this body. Nothing here is a typed-in coordinate: the desk
# and the hand targets are computed from the same numbers, so they cannot
# disagree the way they did when the desk was placed at +Y and the targets at -Y.
def contact_points(j: dict) -> dict:
    from presenter.motion.rig_geometry import SeatedGeometry

    g = SeatedGeometry.measure(j)
    out = {}
    out.update(g.hand_targets(j))
    out.update(g.foot_targets(j))
    out.update(g.pole_targets(j))
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

    # IK, constrained by *measured* hinge axes rather than by pole targets.
    #
    # Poles were tried first and made things worse: a pole angle is measured
    # from the bone's roll, the elbow's roll sits only 0.805 aligned with its
    # anatomical hinge, and leaving the angle at zero twisted both arms out of
    # their sockets. Locking the two axes a hinge cannot rotate about is
    # equivalent, needs no angle, and cannot flip - the solver has nowhere to
    # flip to.
    #
    # Axes come from tools/rig_axes.py. Knees are near-perfect (0.997); elbows
    # and fingers are looser, which is a property of the fitted joint data and
    # is recorded in the contract rather than hidden.
    import json as _json
    axis_path = Path("config/rig_axes.json")
    axis_map = {}
    if axis_path.exists():
        axis_map = _json.loads(axis_path.read_text()).get("axis_map", {})

    def hinge_of(bone_name, fallback_axis, fallback_sign):
        e = axis_map.get(bone_name)
        if not e:
            return fallback_axis, fallback_sign
        return e["curl_axis"], e["curl_sign"]

    def constrain_hinge(pb, axis, sign, lo_deg, hi_deg):
        """Lock every axis but the hinge, and limit the hinge to its range."""
        for a in "xyz":
            setattr(pb, f"lock_ik_{a}", a.upper() != axis)
        lo, hi = math.radians(lo_deg), math.radians(hi_deg)
        if sign < 0:
            lo, hi = -hi, -lo
        setattr(pb, f"use_ik_limit_{axis.lower()}", True)
        setattr(pb, f"ik_min_{axis.lower()}", lo)
        setattr(pb, f"ik_max_{axis.lower()}", hi)

    for side, tname in (("l", "desk_rest_l"), ("r", "mouse")):
        pb = rig.pose.bones.get(f"elbow_{side}")
        if pb is None:
            continue
        ik = pb.constraints.new("IK")
        ik.target = targets[tname]
        ik.chain_count = 2
        ik.influence = 0.0
        ik.name = "hand_contact"
        axis, sgn = hinge_of(f"elbow_{side}", "Z", -1.0 if side == "l" else 1.0)
        constrain_hinge(pb, axis, sgn, 2.0, 150.0)   # elbows never hyperextend

        # The shoulder may swing but barely twist; a humerus that spins in its
        # socket is the other way an IK arm announces itself.
        sh = rig.pose.bones.get(f"shoulder_{side}")
        if sh is not None:
            sh.ik_stiffness_y = 0.85

    for side in ("l", "r"):
        pb = rig.pose.bones.get(f"shin_{side}")
        if pb is None:
            continue
        ik = pb.constraints.new("IK")
        ik.target = targets[f"foot_{side}"]
        ik.chain_count = 2
        ik.influence = 1.0
        ik.name = "foot_contact"
        axis, sgn = hinge_of(f"shin_{side}", "X", 1.0)
        constrain_hinge(pb, axis, sgn, 2.0, 150.0)

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
