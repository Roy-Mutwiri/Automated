"""Humanity preview: watch what the behaviour engine is actually doing.

The photoreal 2D renderer animates a face pasted into a photograph. It cannot
show a torso, shoulders, arms or hands, so judging human presence through it
judges the wrong layer. This renders the same `HumanMotionState` on the MPFB
test rig, where all of it is visible.

It is not meant to look photoreal. It answers one question: **does the body move
like a person?**

## Visibility measurement

Rendering is only half of it. Every frame, the tool projects a set of landmarks
- eye centres, head, shoulders, wrists, chest - into **1920x1080 screen space**
and records their pixel positions, whatever resolution it actually renders at.

That gives the number this project has been missing: not "the chest rotated
0.95 degrees" but "the chest moved 11 pixels at 1080p". A motion that is
physically real and moves two pixels is perceptually absent, and until now
nothing here could tell the difference.

Usage
-----
    python tools/humanity_preview.py --seconds 60 --camera cam3
    python tools/humanity_preview.py --seconds 60 --switch --debug
    python tools/humanity_preview.py --measure-only --seconds 300
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The three the brief names: front, three-quarter, high-angle mechanics.
CAM_ALIAS = {"cam1": "front", "cam3": "three_q", "cam6": "high"}

# Landmarks whose screen-space motion decides whether a viewer sees anything.
LANDMARKS = {
    "eye_l": ("eye_l", "head"), "eye_r": ("eye_r", "head"),
    "head": ("head", "head"), "neck": ("neck", "head"),
    "chest": ("chest", "head"), "shoulder_l": ("shoulder_l", "head"),
    "shoulder_r": ("shoulder_r", "head"), "wrist_l": ("wrist_l", "head"),
    "wrist_r": ("wrist_r", "head"), "pelvis": ("pelvis", "head"),
}


def project(bpy, scene, cam, world_pt, w=1920, h=1080):
    """World point -> pixel coordinates at 1920x1080, whatever we render at."""
    from bpy_extras.object_utils import world_to_camera_view
    co = world_to_camera_view(scene, cam, world_pt)
    return co.x * w, (1.0 - co.y) * h, co.z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="assets/rig/body_rig.blend")
    ap.add_argument("--camera", default="cam3", choices=sorted(CAM_ALIAS))
    ap.add_argument("--switch", action="store_true",
                    help="hard-cut between cam1/cam3/cam6 during the clip")
    ap.add_argument("--switch-every", type=float, default=12.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=854)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--debug", action="store_true", help="overlay state labels")
    ap.add_argument("--measure-only", action="store_true",
                    help="no rendering; landmark projections only")
    ap.add_argument("--frames-dir", default="humanity_frames")
    ap.add_argument("--csv", default="visibility_landmarks.csv")
    args = ap.parse_args()

    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.rig).resolve()))

    from render_body import CAMERAS, _look_at, render_to, setup_scene
    from presenter.behavior.engine import BehaviorEngine
    from presenter.motion.adapters.mpfb import MPFBAdapter
    from presenter.motion.rig_geometry import (SeatedGeometry, load_joints,
                                               mh_to_blender)

    rig = bpy.data.objects["body_rig"]
    adapter = MPFBAdapter(rig)
    scene = setup_scene(CAM_ALIAS[args.camera], args.width, args.height,
                        args.samples)
    cam = bpy.data.objects["cam"]
    g = SeatedGeometry.measure(load_joints())
    span = g.shoulder_y - g.floor_y

    def place(alias):
        sx, fz, hy, look_hy, lens = CAMERAS[alias]
        cam.location = mh_to_blender(
            (sx * span, g.seat_y + hy * span, g.seat_front_z + fz * span))
        _look_at(cam, mh_to_blender(
            (0.0, g.seat_y + look_hy * span, g.seat_back_z + 0.20 * span)))
        bpy.data.cameras["cam"].lens = lens

    engine = BehaviorEngine(seed=args.seed)
    dt = 1.0 / args.fps
    n = int(args.seconds * args.fps)
    out = Path(args.frames_dir)
    if not args.measure_only:
        out.mkdir(parents=True, exist_ok=True)

    rows = []
    header = ["t", "camera", "intention", "attention", "engagement"]
    for k in LANDMARKS:
        header += [f"{k}_x", f"{k}_y"]

    current = args.camera
    t0 = time.perf_counter()
    for i in range(n):
        t = i * dt
        if args.switch:
            want = ["cam1", "cam3", "cam6"][int(t // args.switch_every) % 3]
            if want != current:
                current = want
        place(CAM_ALIAS[current])

        engine.update(dt)
        m = engine.motion
        adapter.apply(m)
        bpy.context.view_layer.update()

        row = [f"{t:.3f}", current, m.behavior_state, m.attention.target,
               f"{m.posture.engagement:.4f}"]
        for name, (bone, _) in LANDMARKS.items():
            pb = rig.pose.bones.get(bone)
            if pb is None:
                row += ["", ""]
                continue
            px, py, _d = project(bpy, scene, cam, rig.matrix_world @ pb.head)
            row += [f"{px:.2f}", f"{py:.2f}"]
        rows.append(row)

        if args.measure_only:
            if i % 900 == 0 and i:
                print(f"[preview] measured {t:.0f}s", flush=True)
            continue

        render_to(scene, out / f"f{i:05d}.png")
        if args.debug:
            im = cv2.imread(str(out / f"f{i:05d}.png"))
            bar = np.full((74, im.shape[1], 3), 16, np.uint8)
            cv2.putText(bar, f"{current.upper()}   {m.behavior_state}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 235, 255), 1)
            cv2.putText(bar, f"attention {m.attention.target}", (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
            cv2.putText(bar,
                        f"head yaw {m.head_world_yaw():+6.1f}  "
                        f"pitch {m.head_world_pitch():+6.1f}  "
                        f"posture {m.posture.engagement:+5.2f}", (10, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 170, 170), 1)
            cv2.imwrite(str(out / f"f{i:05d}.png"), np.vstack([bar, im]))

        if i % 60 == 0 and i:
            el = time.perf_counter() - t0
            print(f"[preview] {i}/{n}  {i / el:.2f} fps  "
                  f"eta {(n - i) / max(i / el, 1e-3) / 60:.1f} min", flush=True)

    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"[preview] {n} frames, landmarks -> {args.csv}")
    if not args.measure_only:
        print(f"[preview] frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
