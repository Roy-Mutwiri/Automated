"""Freeze the simulation at one timestamp and render every enabled camera.

This is the acceptance test the whole architecture exists to pass. The world is
built **once**, from one `AvatarPose` at one simulation time, and then each
camera renders that same scene. The human transforms cannot differ between
cameras because there is only one human and it was placed before any camera was
asked for a picture.

That is the difference from the previous approach, where "camera 5" meant
"generate another image and hope". Here a camera has no ability to change
anything.

Usage
-----
    python tools/render_multicam.py --time 314.5
    python tools/render_multicam.py --time 314.5 --cameras cam1 cam2 cam3
    python tools/render_multicam.py --turntable          # identity check
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "renders"


def frozen_pose(sim_time: float, fps: float = 30.0, seed: int = 7):
    """Advance the behaviour engine to `sim_time` and stop.

    The engine is time-driven, so this is the same state the live application
    would be in at that moment - the multicam test is not a special code path
    with its own animation.
    """
    from presenter.behavior import BehaviorEngine

    engine = BehaviorEngine(seed=seed)
    dt = 1.0 / fps
    steps = max(int(round(sim_time / dt)), 1)
    pose = None
    for _ in range(steps):
        pose = engine.update(dt)
    return pose, engine


def render(world, camera_ids, width, height, samples, tag=""):
    import bpy

    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE"
    scn.render.resolution_x = width
    scn.render.resolution_y = height
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    # Matched exposure and view transform for every camera. Auto-exposure
    # "breathing" between cuts is one of the loudest multicam tells.
    scn.view_settings.view_transform = "Filmic"
    scn.view_settings.look = "None"
    scn.view_settings.exposure = 0.0

    OUT.mkdir(exist_ok=True)
    written = []
    for cam_id in camera_ids:
        ob = world.cameras.get(cam_id)
        if ob is None:
            print(f"[multicam] {cam_id}: not in the rig, skipped")
            continue
        scn.camera = ob
        path = OUT / f"{tag}{cam_id}.png"
        scn.render.filepath = str(path)
        t = _time.perf_counter()
        bpy.ops.render.render(write_still=True)
        print(f"[multicam] {cam_id}: {_time.perf_counter() - t:.1f}s -> {path.name}",
              flush=True)
        written.append(path)
    return written


def contact_sheet(paths, out, cols=4, cell=(640, 360)):
    import cv2
    import numpy as np

    tiles = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        tiles.append(cv2.resize(im, cell))
    if not tiles:
        return None
    while len(tiles) % cols:
        tiles.append(np.zeros((cell[1], cell[0], 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"[multicam] contact sheet -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time", type=float, default=314.5,
                    help="simulation timestamp to freeze at, in seconds")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cameras", nargs="*", default=None,
                    help="camera ids; default is every enabled camera")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--turntable", action="store_true",
                    help="render the human from seven yaw angles instead, to "
                         "check identity holds around the head")
    args = ap.parse_args()

    pose, engine = frozen_pose(args.time, seed=args.seed)
    print(f"[multicam] frozen at t={args.time:.3f}s  "
          f"yaw={pose.yaw:+.2f} pitch={pose.pitch:+.2f} roll={pose.roll:+.2f} "
          f"lids={pose.eye_open_l:.3f}/{pose.eye_open_r:.3f} "
          f"breath={pose.scale:.5f}")

    from presenter.scene3d.world import build_world

    world = build_world(pose)
    problems = world.validate_cameras()
    for p in problems:
        print(f"[multicam] CAMERA PROBLEM: {p}")

    if args.turntable:
        import bpy
        import math

        head = world.landmarks["head_centre"]
        radius = 1.5
        ids = []
        for k, deg in enumerate((-70, -40, -20, 0, 20, 40, 70)):
            a = math.radians(deg)
            pos = (head[0] + radius * math.sin(a),
                   head[1] + radius * math.cos(a),
                   head[2] + 0.02)
            data = bpy.data.cameras.new(f"tt{k}")
            data.sensor_width = 36.0
            data.lens = 85.0
            ob = bpy.data.objects.new(f"tt{k}", data)
            ob.location = pos
            from presenter.scene3d.world import _look_at_euler
            ob.rotation_euler = _look_at_euler(pos, head)
            bpy.context.scene.collection.objects.link(ob)
            world.cameras[f"tt{k}"] = ob
            ids.append(f"tt{k}")
        paths = render(world, ids, 480, 480, args.samples, tag="turntable_")
        contact_sheet(paths, OUT / "identity_turntable.png", cols=7,
                      cell=(320, 320))
        return 0

    enabled = [c["id"] for c in world.c["cameras"] if c.get("enabled", True)]
    ids = args.cameras if args.cameras else enabled
    paths = render(world, ids, args.width, args.height, args.samples)
    contact_sheet(paths, OUT / "multicam_same_timestamp_contact_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
