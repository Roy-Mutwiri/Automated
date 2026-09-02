"""Measure the body against every contact surface. The physical acceptance test.

Renders nothing. It evaluates the *deformed* mesh - the armature modifier
applied, not the rest shape - and reports, for each contact:

    gap     >0  the body is floating above the surface
            =0  touching
            <0  penetrating

A seated human touches four things: the seat, the backrest (sometimes), the
floor through both feet, and the desk through both hands. Each has a different
acceptable answer, and "looks about right from this angle" is not one of them.
Both failure directions matter and they look identical in a still: a hand
hovering two centimetres above a desk and a hand two centimetres inside it are
both wrong, and neither is visible head-on.

Usage
-----
    python tools/contact_check.py
    python tools/contact_check.py --engagement -1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def evaluated_vertices(bpy, obj):
    """World-space vertices of the mesh *after* the armature deforms it."""
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    mesh = ev.to_mesh()
    mw = ev.matrix_world
    verts = [mw @ v.co for v in mesh.vertices]
    ev.to_mesh_clear()
    return verts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="assets/rig/body_rig.blend")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--engagement", type=float, default=None,
                    help="force posture engagement instead of letting it drift")
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.rig).resolve()))

    from presenter.behavior.engine import BehaviorEngine
    from presenter.motion.adapters.mpfb import MPFBAdapter
    from presenter.motion.rig_geometry import (UNITS_PER_CM, SeatedGeometry,
                                               load_joints)

    j = load_joints()
    g = SeatedGeometry.measure(j)
    rig = bpy.data.objects["body_rig"]
    body = bpy.data.objects["body"]
    adapter = MPFBAdapter(rig)

    engine = BehaviorEngine(seed=args.seed)
    for _ in range(int(args.seconds * 30)):
        engine.update(1.0 / 30.0)
    if args.engagement is not None:
        engine.body._engagement = args.engagement
        engine.body._engagement_target = args.engagement
        engine.update(1.0 / 30.0)

    adapter.apply(engine.motion)
    bpy.context.view_layer.update()
    verts = evaluated_vertices(bpy, body)

    # Blender space: x left, y back, z up. MakeHuman y -> Blender z, and
    # MakeHuman z -> Blender -y.
    seat_z = g.seat_y
    floor_z = g.floor_y
    desk_z = g.desk_y
    back_y = -g.seat_back_z            # backrest face, in Blender +Y

    cm = 1.0 / UNITS_PER_CM

    def band(pred):
        return [v for v in verts if pred(v)]

    results = []
    pbones = rig.pose.bones

    # Every region is anchored to a posed bone rather than to a fixed
    # coordinate band. Bands move out from under the body the moment it
    # translates: sliding the pelvis 6.5 cm back to settle into the chair took
    # the buttocks out of a fixed seat band, and the check happily measured the
    # thigh instead and reported the pelvis floating.
    pelvis_b = pbones.get("pelvis")
    if pelvis_b is not None:
        centre = rig.matrix_world @ pelvis_b.head
        pts = [v for v in verts
               if abs(v.x - centre.x) < 2.4 and abs(v.y - centre.y) < 2.2
               and v.z < centre.z + 0.6]
        if pts:
            gap = min(v.z for v in pts) - seat_z
            # 0 to 1.5 cm. A rigid debug seat has no cushion to compress into.
            results.append(("pelvis / seat", gap, "rests on", -0.02, 0.15))

    # Backrest: the rearmost point of the torso against the rest.
    chest_b = pbones.get("chest")
    if chest_b is not None:
        centre = rig.matrix_world @ chest_b.head
        pts = [v for v in verts
               if abs(v.x - centre.x) < 2.2
               and centre.z - 1.8 < v.z < centre.z + 1.8]
        if pts:
            gap = back_y - max(v.y for v in pts)
            # Floating clear is correct when upright; only penetration fails,
            # and the settled case is checked separately.
            results.append(("back / backrest", gap, "may float", -0.15, 9.0))

    # Feet: soles against the floor.
    for side, xs in (("left", lambda x: x > 0), ("right", lambda x: x < 0)):
        pts = band(lambda v: v.z < floor_z + 1.8 and xs(v.x))
        if pts:
            gap = min(v.z for v in pts) - floor_z
            results.append((f"{side} foot / floor", gap, "planted", -0.15, 0.9))

    # Hands: palms on the desk.
    #
    # Anchored to the posed wrist bone, not to a coordinate band. A band around
    # desk height in front of the body also contains the forearm and the point
    # of the elbow, and reported the hands as 14 cm inside the desk when what it
    # had actually found was the elbow hanging below the desk edge - which is
    # where an elbow belongs.
    pbones = rig.pose.bones
    for side, label in (("l", "left"), ("r", "right")):
        wb = pbones.get(f"wrist_{side}")
        if wb is None:
            continue
        wrist = rig.matrix_world @ wb.head
        hand_len = (rig.matrix_world @ wb.tail - wrist).length
        radius = max(hand_len * 2.2, 0.9)
        pts = [v for v in verts if (v - wrist).length < radius]
        if pts:
            gap = min(v.z for v in pts) - desk_z
            results.append((f"{label} hand / desk", gap, "rests on", -0.20, 0.9))

    eng = engine.motion.posture
    print(f"[contact] engagement {eng.engagement:+.2f}  "
          f"back_contact {eng.back_contact:.2f}  "
          f"root slide {engine.motion.root_z:+.3f} u")
    print(f"[contact] {'surface':<22}{'gap (u)':>9}{'gap (cm)':>10}   verdict")
    ok = True
    for name, gap, kind, lo, hi in results:
        if gap < lo:
            verdict, good = "PENETRATING", False
        elif gap > hi:
            verdict, good = "FLOATING", False
        else:
            verdict, good = "ok", True
        ok &= good
        print(f"[contact] {name:<22}{gap:>9.3f}{gap * cm:>10.1f}   {verdict}")

    print(f"[contact] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
