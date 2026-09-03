"""Search for CAM4 and CAM5 placements, and score them on geometry.

Positions are generated from the avatar's derived basis (`tools/camera_layout`),
never from a world axis chosen by hand - putting cameras "behind him" by
reasoning about +Y from memory is what produced the two shots this is
correcting.

Scoring is geometric rather than by eye. Each candidate projects a set of scene
landmarks through the real camera and ray-casts to each one, so "the desk is
visible" means the desk projects inside the frame and nothing is in the way -
not that a thumbnail looked about right.

    python tools/search_camera_placement.py --camera cam4
    python tools/search_camera_placement.py --camera cam5
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

OUT = ROOT / "renders" / "camera_preview"


def landmarks(g):
    """The things a shot is judged on, as named world points."""
    pts = {}
    for m in g["monitors"]:
        pts[m["id"]] = np.array(m["centre"], float)
    d = g["desk"]
    cx, cy = d["centre"][0], d["centre"][1]
    sx, sy = d["top_size"][0], d["top_size"][1]
    z = d["top_z"]
    pts["desk_centre"] = np.array([cx, cy, z])
    pts["desk_left"] = np.array([cx - sx / 2 + 0.1, cy, z])
    pts["desk_right"] = np.array([cx + sx / 2 - 0.1, cy, z])
    pts["mic_capsule"] = np.array(g["microphone"]["capsule_centre"], float)
    ch = g["chair"]
    pts["chair_back"] = np.array([ch["base_centre"][0], ch["base_centre"][1] - 0.2,
                                  ch["seat_height"] + ch["back_size"][2] / 2])
    for lm in g.get("landmarks", []):
        pts[lm["id"]] = np.array(lm["centre"], float)
    return pts


def visibility(scn, dg, cam_ob, points, head, bpy):
    """Which landmarks this camera can actually see, and how big the head is."""
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    origin = cam_ob.matrix_world.translation
    seen = {}
    for name, p in points.items():
        v = Vector(p.tolist())
        uv = world_to_camera_view(scn, cam_ob, v)
        inside = 0.0 <= uv.x <= 1.0 and 0.0 <= uv.y <= 1.0 and uv.z > 0.0
        if not inside:
            seen[name] = False
            continue
        # In frame is not the same as visible. Ray-cast and accept the hit only
        # if it lands at (or past) the landmark itself.
        d = v - origin
        dist = d.length
        hit, loc, *_ = scn.ray_cast(dg, origin, d.normalized())
        seen[name] = (not hit) or ((loc - origin).length >= dist - 0.16)

    hv = Vector(head.tolist())
    huv = world_to_camera_view(scn, cam_ob, hv)
    head_dist = (hv - origin).length
    # Solid angle of a 0.16 m head sphere as a fraction of the frame.
    lens = cam_ob.data.lens
    sensor = cam_ob.data.sensor_width
    hfov = 2.0 * math.atan(sensor / (2.0 * lens))
    vfov = hfov * scn.render.resolution_y / scn.render.resolution_x
    ang = 2.0 * math.atan(0.16 / max(head_dist, 0.2))
    head_frac = (ang / hfov) * (ang / vfov)
    head_in = 0.0 <= huv.x <= 1.0 and 0.0 <= huv.y <= 1.0 and huv.z > 0
    return seen, (head_frac if head_in else 0.0), head_dist


def clearance(pos, g):
    """Distance to the nearest wall. Negative means outside the room."""
    r = g["room"]
    W, D, H = r["width"], r["depth"], r["height"]
    return min(pos[0] + W / 2, W / 2 - pos[0], pos[1], D - pos[1],
               pos[2], H - pos[2])


def score(kind, seen, head_frac, dark, clear):
    """Turn measurements into one number, with the shot's intent weighted in."""
    monitors = sum(bool(seen.get(k)) for k in
                   ("monitor_left", "monitor_main", "monitor_right"))
    desk = sum(bool(seen.get(k)) for k in
               ("desk_centre", "desk_left", "desk_right"))
    s = 0.0
    notes = []
    if kind == "cam4":
        # Over the shoulder: the head must be present but must not dominate,
        # and the working surface is the subject.
        s += 22.0 * desk / 3.0
        s += 10.0 * monitors / 3.0
        s += 8.0 * bool(seen.get("mic_capsule"))
        if 0.06 <= head_frac <= 0.34:
            s += 22.0
            notes.append("head foreground ok")
        elif head_frac > 0.34:
            s -= 16.0 * (head_frac - 0.34) / 0.34
            notes.append("head dominates")
        else:
            s -= 14.0
            notes.append("no shoulder in frame")
    else:
        # Rear wide: readability of the whole battlestation.
        s += 26.0 * desk / 3.0
        s += 16.0 * monitors / 3.0
        s += 8.0 * bool(seen.get("chair_back"))
        s += 6.0 * bool(seen.get("mic_capsule"))
        if 0.02 <= head_frac <= 0.18:
            s += 14.0
            notes.append("subject reads at wide")
        elif head_frac > 0.18:
            s -= 12.0
            notes.append("too tight for a wide")
    context = sum(bool(seen.get(k)) for k in
                  ("shelf_right", "pc_tower", "headphone_stand", "speaker_right"))
    s += 4.0 * min(context, 2)
    # Empty black frame is the exact failure being corrected.
    s -= 40.0 * max(dark - 0.35, 0.0)
    if clear < 0.12:
        s -= 50.0
        notes.append("camera collides with the room")
    return s, monitors, desk, notes


def candidates(kind, basis, g):
    """A small, deliberate grid in the avatar's own frame."""
    eye = basis.eye
    desk_work = np.array([g["desk"]["centre"][0], g["desk"]["centre"][1],
                          g["desk"]["top_z"] + 0.06])
    if kind == "cam4":
        # Intimate over-the-shoulder: close behind, offset to one side.
        target = desk_work
        for behind in (0.55, 0.80):
            for lateral in (-0.52, -0.34, 0.34, 0.52):
                for height in (1.46, 1.64):
                    yield (basis.place(behind=behind, lateral=lateral,
                                       height=height), target, 40.0,
                           dict(behind=behind, lateral=lateral, height=height))
    else:
        # Rear wide: the composition centre of subject + desk, not his origin.
        target = (eye + desk_work) / 2.0
        for behind in (0.80, 1.00):
            for lateral in (-0.70, -0.35, 0.35, 0.70):
                for height in (1.60, 1.95):
                    yield (basis.place(behind=behind, lateral=lateral,
                                       height=height), target, 30.0,
                           dict(behind=behind, lateral=lateral, height=height))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", choices=["cam4", "cam5"], required=True)
    ap.add_argument("--time", type=float, default=314.5)
    args = ap.parse_args()

    import bpy

    from camera_layout import AvatarBasis, load
    from presenter.scene3d.world import build_world
    from render_multicam import frozen_pose

    g, _ = load()
    basis = AvatarBasis(g)
    pose, _ = frozen_pose(args.time)
    world = build_world(pose)
    print(basis.describe(), "\n")

    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE"
    scn.render.resolution_x, scn.render.resolution_y = 480, 270
    scn.view_settings.view_transform = "Filmic"
    scn.view_settings.look = "None"
    if hasattr(scn, "eevee"):
        scn.eevee.taa_render_samples = 8

    data = bpy.data.cameras.new("probe")
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.clip_start = 0.02
    probe = bpy.data.objects.new("probe", data)
    scn.collection.objects.link(probe)

    pts = landmarks(g)
    head = np.array([basis.eye[0], basis.eye[1], basis.eye[2]])
    tmp = OUT / f"_{args.camera}_candidates"
    tmp.mkdir(parents=True, exist_ok=True)

    from presenter.scene3d.world import _look_at_euler

    rows = []
    for i, (pos, target, lens, meta) in enumerate(candidates(args.camera, basis, g)):
        probe.location = pos.tolist()
        probe.rotation_euler = _look_at_euler(pos.tolist(), target.tolist())
        data.lens = lens
        scn.camera = probe
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()

        seen, head_frac, head_dist = visibility(scn, dg, probe, pts, head, bpy)
        path = tmp / f"{i:02d}.png"
        scn.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)

        import cv2
        im = cv2.imread(str(path))
        dark = float((cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 26).mean())
        clear = clearance(pos, g)
        s, mons, desk, notes = score(args.camera, seen, head_frac, dark, clear)
        rows.append(dict(i=i, path=path, pos=pos, target=target, lens=lens,
                         meta=meta, score=s, monitors=mons, desk=desk,
                         head_frac=head_frac, dark=dark, clear=clear,
                         notes=notes, seen=seen))
        print(f"{i:02d} b={meta['behind']:.2f} lat={meta['lateral']:+.2f} "
              f"z={meta['height']:.2f}  score {s:6.1f}  mon {mons}/3  desk {desk}/3 "
              f"head {head_frac*100:4.1f}%  dark {dark*100:4.1f}%  "
              f"{'; '.join(notes)}")

    rows.sort(key=lambda r: -r["score"])
    best = rows[0]
    print(f"\nBEST {args.camera}: candidate {best['i']}  score {best['score']:.1f}")
    print(f"  position {np.round(best['pos'], 3).tolist()}")
    print(f"  look_at  {np.round(best['target'], 3).tolist()}")
    print(f"  lens     {best['lens']:.0f} mm")
    print(f"  monitors visible {best['monitors']}/3   desk {best['desk']}/3   "
          f"dark {best['dark']*100:.1f}%")

    sheet(rows, args.camera)
    return 0


def sheet(rows, kind):
    import cv2

    cols = 4
    tiles = []
    for r in sorted(rows, key=lambda x: x["i"]):
        im = cv2.imread(str(r["path"]))
        if im is None:
            continue
        im = cv2.resize(im, (480, 270))
        band = im.copy()
        cv2.rectangle(band, (0, 0), (480, 62), (0, 0, 0), -1)
        cv2.addWeighted(band, 0.66, im, 0.34, 0, im)
        p = r["pos"]
        cv2.putText(im, f"{r['i']:02d}  score {r['score']:5.1f}"
                        f"{'   <-- BEST' if r is rows[0] else ''}",
                    (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (120, 240, 160) if r is rows[0] else (225, 232, 245), 1,
                    cv2.LINE_AA)
        cv2.putText(im, f"pos [{p[0]:+.2f} {p[1]:+.2f} {p[2]:.2f}]  {r['lens']:.0f}mm",
                    (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 208, 222), 1,
                    cv2.LINE_AA)
        t = r["target"]
        cv2.putText(im, f"aim [{t[0]:+.2f} {t[1]:+.2f} {t[2]:.2f}]  "
                        f"mon {r['monitors']}/3 desk {r['desk']}/3 "
                        f"dark {r['dark']*100:.0f}%",
                    (8, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 208, 222), 1,
                    cv2.LINE_AA)
        if r is rows[0]:
            cv2.rectangle(im, (1, 1), (478, 268), (120, 240, 160), 3)
        tiles.append(im)

    import numpy as _np
    while len(tiles) % cols:
        tiles.append(_np.zeros((270, 480, 3), _np.uint8))
    grid = _np.vstack([_np.hstack(tiles[i:i + cols])
                       for i in range(0, len(tiles), cols)])
    dest = OUT / f"{kind}_candidates.png"
    cv2.imwrite(str(dest), grid)
    print(f"candidate sheet -> {dest}")


if __name__ == "__main__":
    raise SystemExit(main())
