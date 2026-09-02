"""Measure the rig's coordinate conventions. Nothing here is assumed.

This exists because two separate bugs came from guessing bone axes: finger curl
was mapped to local Z, which fans the fingers apart instead of closing them, and
arm IK pole angles were left at zero, which twisted both arms out of their
sockets.

Both are the same mistake. A Blender bone's local Y runs along its own length,
and its X and Z are decided by the bone's *roll*, which here comes from fitted
joint positions rather than from anything a human chose. So the axes are
whatever the data made them, and the only way to know is to read them.

## What is measured

For every bone: rest head and tail, length, and its three local axes expressed
as directions in armature space.

For hinge joints - elbows, knees - additionally: the **anatomical hinge axis**,
computed as the normal of the plane containing the parent segment and the child
segment. A hinge can only rotate about that normal, so whichever local axis is
closest to it is the axis to drive, and the closeness itself says how well the
rig's roll matches its anatomy.

For fingers: the same calculation against the plane of flexion.

## Output

    docs/rig_coordinate_contract.md   human-readable, with the numbers
    config/rig_axes.json              machine-readable, consumed by the adapter

The adapter reads the JSON. It is not allowed to contain a hand-written axis
index anywhere.

Usage
-----
    python tools/rig_axes.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

AXIS_NAMES = ("X", "Y", "Z")


def _fmt(v):
    return f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"


def measure(rig_path: Path) -> dict:
    import bpy
    from mathutils import Vector

    bpy.ops.wm.open_mainfile(filepath=str(rig_path.resolve()))
    rig = bpy.data.objects["body_rig"]
    arm = rig.data

    bones = {}
    for b in arm.bones:
        m = b.matrix_local.to_3x3()
        axes = {AXIS_NAMES[i]: tuple(round(x, 6) for x in m.col[i]) for i in range(3)}
        bones[b.name] = dict(
            head=tuple(round(x, 4) for x in b.head_local),
            tail=tuple(round(x, 4) for x in b.tail_local),
            length=round(b.length, 4),
            parent=b.parent.name if b.parent else None,
            axes=axes,
        )

    # --- hinge axes -------------------------------------------------------
    #
    # A hinge rotates about the normal of the plane its two segments lie in.
    # Comparing that normal against the bone's own local axes says which axis
    # to drive and how well the rig's roll agrees with the anatomy.
    HINGES = {}
    for side in ("l", "r"):
        HINGES[f"elbow_{side}"] = (f"shoulder_{side}", f"elbow_{side}")
        HINGES[f"shin_{side}"] = (f"thigh_{side}", f"shin_{side}")
        for f in range(1, 6):
            for seg in (2, 3):
                HINGES[f"finger_{side}_{f}_{seg}"] = (
                    f"finger_{side}_{f}_{seg - 1}", f"finger_{side}_{f}_{seg}")

    hinges = {}
    for name, (parent_name, child_name) in HINGES.items():
        pb = arm.bones.get(parent_name)
        cb = arm.bones.get(child_name)
        if pb is None or cb is None:
            continue
        u = (Vector(pb.tail_local) - Vector(pb.head_local)).normalized()
        v = (Vector(cb.tail_local) - Vector(cb.head_local)).normalized()
        n = u.cross(v)
        if n.length < 1e-6:
            # Perfectly straight in the rest pose - the plane is undefined, so
            # there is nothing to measure. Recorded rather than silently
            # skipped, because a straight rest limb is itself worth knowing.
            hinges[name] = dict(degenerate=True,
                                note="segments colinear at rest; hinge plane undefined")
            continue
        n.normalize()

        m = cb.matrix_local.to_3x3()
        scores = {AXIS_NAMES[i]: abs(n.dot(Vector(m.col[i]))) for i in range(3)}
        best = max(scores, key=scores.get)
        sign = 1.0 if n.dot(Vector(m.col[AXIS_NAMES.index(best)])) > 0 else -1.0
        hinges[name] = dict(
            normal=tuple(round(x, 4) for x in n),
            alignment={k: round(v, 4) for k, v in scores.items()},
            hinge_axis=best,
            hinge_sign=sign,
            # 1.0 means the rig's roll puts an axis exactly on the anatomical
            # hinge; below ~0.9 the bone is rolled away from its own joint and
            # driving a single axis will introduce twist.
            quality=round(scores[best], 4),
        )

    return dict(bones=bones, hinges=hinges)


def write_contract(data: dict, out_md: Path, out_json: Path) -> None:
    bones, hinges = data["bones"], data["hinges"]

    # Machine-readable: only what an adapter needs.
    axis_map = {}
    for name, h in hinges.items():
        if h.get("degenerate"):
            continue
        axis_map[name] = dict(curl_axis=h["hinge_axis"],
                              curl_sign=h["hinge_sign"],
                              quality=h["quality"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        dict(generated_by="tools/rig_axes.py", axis_map=axis_map,
             bones={k: v["axes"] for k, v in bones.items()}), indent=2))

    good = [h for h in hinges.values() if not h.get("degenerate")]
    worst = min((h["quality"] for h in good), default=0.0)
    fingers = [h for k, h in hinges.items()
               if k.startswith("finger") and not h.get("degenerate")]

    lines = [
        "# Rig coordinate contract",
        "",
        "**Generated by `tools/rig_axes.py`. Do not edit by hand.**",
        "",
        "Every number here is read off the rig, not chosen. Two bugs came from",
        "guessing these: finger curl was mapped to local Z, which fans the",
        "fingers apart rather than closing them, and arm IK pole angles were left",
        "at zero, which twisted both arms out of their sockets.",
        "",
        "## World",
        "",
        "| | |",
        "|---|---|",
        "| +X | character's **left** |",
        "| +Y | **behind** the character |",
        "| +Z | **up** |",
        "",
        "Blender's convention. The character faces **-Y**.",
        "",
        "## Human local",
        "",
        "| | |",
        "|---|---|",
        "| forward | -Y |",
        "| right | -X |",
        "| up | +Z |",
        "",
        "## Source data",
        "",
        "MakeHuman joints are Y-up with +Z forward. One conversion, in",
        "`presenter/motion/rig_geometry.py`:",
        "",
        "```",
        "(x, y, z)_mh  ->  (x, -z, y)_blender",
        "```",
        "",
        "## Bone local axes",
        "",
        "A bone's local **Y always runs along its own length**. X and Z are set",
        "by the bone's roll, which here derives from fitted joint positions, so",
        "they differ per bone and must be looked up rather than assumed.",
        "",
        "| bone | length | local X | local Y (along bone) | local Z |",
        "|---|---|---|---|---|",
    ]
    order = [b for b in ("pelvis", "spine_lower", "spine_mid", "chest",
                         "chest_top", "neck", "head",
                         "clavicle_l", "shoulder_l", "elbow_l", "wrist_l",
                         "thigh_l", "shin_l", "foot_l",
                         "finger_l_2_1", "finger_l_2_2") if b in bones]
    for name in order:
        b = bones[name]
        lines.append(f"| `{name}` | {b['length']:.3f} | {_fmt(b['axes']['X'])} | "
                     f"{_fmt(b['axes']['Y'])} | {_fmt(b['axes']['Z'])} |")

    lines += [
        "",
        "## Hinge axes, measured",
        "",
        "A hinge rotates about the normal of the plane containing its two",
        "segments. `quality` is how closely the bone's own axis lines up with",
        "that normal: **1.0** means the roll puts an axis exactly on the",
        "anatomical hinge, and anything below about 0.9 means driving a single",
        "axis will introduce twist as well as flexion.",
        "",
        "| joint | hinge axis | sign | quality | X / Y / Z alignment |",
        "|---|---|---|---|---|",
    ]
    show = [k for k in ("elbow_l", "elbow_r", "shin_l", "shin_r",
                        "finger_l_1_2", "finger_l_2_2", "finger_l_3_2",
                        "finger_l_4_2", "finger_l_5_2",
                        "finger_r_2_2", "finger_r_3_2") if k in hinges]
    for k in show:
        h = hinges[k]
        if h.get("degenerate"):
            lines.append(f"| `{k}` | - | - | - | {h['note']} |")
            continue
        a = h["alignment"]
        lines.append(
            f"| `{k}` | **{h['hinge_axis']}** | {h['hinge_sign']:+.0f} | "
            f"{h['quality']:.3f} | {a['X']:.2f} / {a['Y']:.2f} / {a['Z']:.2f} |")

    lines += [
        "",
        f"Worst hinge alignment across the rig: **{worst:.3f}**.",
        "",
        "## How this is consumed",
        "",
        "`config/rig_axes.json` carries the machine-readable map and",
        "`adapters/mpfb.py` reads it. There is no hand-written axis index in the",
        "adapter any more; if the rig is rebuilt with different joint data the",
        "map regenerates and the adapter follows.",
        "",
        f"Fingers measured: {len(fingers)} hinge joints.",
    ]
    lines += _scene_section()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def _scene_section() -> list[str]:
    """Scene geometry, reach fractions and contact tolerances.

    Generated here rather than appended to the file afterwards: this document
    is regenerated whenever the rig changes, and anything written into it by
    another hand is silently lost on the next run.
    """
    import math

    from presenter.motion.rig_geometry import (UNITS_PER_CM, SeatedGeometry,
                                               load_joints)

    j = load_joints()
    g = SeatedGeometry.measure(j)
    cm = 1.0 / UNITS_PER_CM

    reach_rows = []
    for k, v in g.hand_targets(j).items():
        side = "r" if k in ("mouse", "keyboard") or k.endswith("_r") else "l"
        d = math.dist(j[f"joint-{side}-shoulder"], v)
        reach_rows.append(f"| `{k}` | {d:.2f} | {100 * d / g.arm_reach:.0f}% |")

    return [
        "",
        "## Scene geometry",
        "",
        "Derived once in `presenter/motion/rig_geometry.py` from the body's own",
        "measurements, and read by the rig builder, the debug renderer and the",
        "contact check alike. Nothing places furniture independently - doing so",
        "is how the desk ended up behind the character.",
        "",
        "Scale is recovered rather than assumed: the mesh spans 17.53 units from",
        "`joint-ground` to the skull, so for a ~1.75 m adult **1 unit ~ 10 cm**.",
        "",
        "| | units | cm |",
        "|---|---|---|",
        f"| thigh | {g.thigh:.2f} | {g.thigh * cm:.0f} |",
        f"| shin | {g.shin:.2f} | {g.shin * cm:.0f} |",
        f"| upper arm | {g.upper_arm:.2f} | {g.upper_arm * cm:.0f} |",
        f"| forearm | {g.forearm:.2f} | {g.forearm * cm:.0f} |",
        f"| **arm reach** (shoulder to wrist) | **{g.arm_reach:.2f}** | "
        f"**{g.arm_reach * cm:.0f}** |",
        f"| seat above floor | {g.seat_y - g.floor_y:.2f} | "
        f"{(g.seat_y - g.floor_y) * cm:.0f} |",
        f"| desk above floor | {g.desk_y - g.floor_y:.2f} | "
        f"{(g.desk_y - g.floor_y) * cm:.0f} |",
        "",
        "The last two are the check that the derivation is sane: a 44 cm seat",
        "and a 73 cm desk are ordinary furniture, arrived at from the skeleton",
        "rather than typed in.",
        "",
        "## Reach fractions",
        "",
        "Every resting IK target is placed as a fraction of **measured** reach.",
        "The failure this replaces put the mouse at 117% of reach; the solver",
        "did the only thing available and straightened the arm to point at it,",
        "which is why the character sat with both arms locked out sideways.",
        "",
        "| target | distance (u) | % of reach |",
        "|---|---|---|",
        *reach_rows,
        "",
        "Desk and mouse sit at 80%, inside the 70-90% band. The lap is higher at",
        "88% and that is anatomically right: with the torso upright the arm",
        "hangs almost straight to the lap, so the elbow is barely bent.",
        "",
        "## Contact tolerances",
        "",
        "`tools/contact_check.py` evaluates the *deformed* mesh against each",
        "surface. Gaps are positive when floating, negative when penetrating.",
        "",
        "| contact | acceptable |",
        "|---|---|",
        "| pelvis / seat | -0.2 .. +1.5 cm |",
        "| back / backrest | > -1.5 cm (floating clear is correct when upright) |",
        "| foot / floor | -1.5 .. +9 cm |",
        "| hand / desk | -2 .. +9 cm |",
        "",
        "Every region is anchored to a **posed bone**, never to a coordinate",
        "band. A band stops covering the body the moment it moves: sliding the",
        "pelvis back to settle into the chair took the buttocks out of a fixed",
        "seat band and the check measured the thigh instead, reporting the",
        "pelvis floating when it was resting.",
        "",
        "## Root translation",
        "",
        "`pose_bone.location` is **bone-local**. The root bone points up the",
        "spine, so a world-space slide assigned to it moves the body *along the",
        "spine*: settling back lifted the pelvis 4.7 cm off the seat. The",
        "adapter rotates the delta into the bone's own frame before assigning",
        "it.",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="assets/rig/body_rig.blend")
    ap.add_argument("--md", default="docs/rig_coordinate_contract.md")
    ap.add_argument("--json", default="config/rig_axes.json")
    args = ap.parse_args()

    data = measure(Path(args.rig))
    write_contract(data, Path(args.md), Path(args.json))

    print(f"[axes] {len(data['bones'])} bones, {len(data['hinges'])} hinges measured")
    for k in ("elbow_l", "elbow_r", "shin_l", "finger_l_2_2", "finger_r_2_2"):
        h = data["hinges"].get(k)
        if h and not h.get("degenerate"):
            print(f"[axes]   {k:<16} hinge {h['hinge_axis']} "
                  f"sign {h['hinge_sign']:+.0f}  quality {h['quality']:.3f}")
        elif h:
            print(f"[axes]   {k:<16} DEGENERATE ({h['note']})")
    print(f"[axes] -> {args.md}, {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
