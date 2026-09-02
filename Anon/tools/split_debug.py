"""Both renderers, one motion state, side by side.

The architectural claim is that behaviour is renderer-independent. This renders
it: a single `BehaviorEngine` is stepped once per frame, and the *same*
`HumanMotionState` is handed to two adapters that share nothing - one drives a
2D diffusion face pasted into a photograph, the other poses 48 bones in
Blender.

If the engine had any renderer-specific code the two halves would diverge. They
do not: the head turns at the same instant, to the same angle, for the same
reason, because neither adapter decided anything.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out", default="split_debug_raw.mp4")
    args = ap.parse_args()

    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(Path("assets/rig/body_rig.blend").resolve()))

    from render_body import render_to, setup_scene
    from presenter.behavior.engine import BehaviorEngine
    from presenter.motion.adapters.face2d import to_avatar_pose
    from presenter.motion.adapters.mpfb import MPFBAdapter
    from presenter.render.liveportrait import LivePortraitRenderer

    rig = bpy.data.objects["body_rig"]
    mpfb = MPFBAdapter(rig)
    scene = setup_scene("three_q", 640, 720, 8)

    face = LivePortraitRenderer(
        source_image="assets/master/master_v04_final.png",
        liveportrait_root="third_party/LivePortrait",
        output_size=(1280, 720), framing="full", environment="source",
        neutralize_pose=0.0)

    engine = BehaviorEngine(seed=args.seed)
    dt = 1.0 / args.fps
    n = int(args.seconds * args.fps)
    tmp = Path("split_tmp")
    tmp.mkdir(exist_ok=True)

    w = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps, (1920, 720))
    t0 = time.perf_counter()
    for i in range(n):
        engine.update(dt)
        motion = engine.motion                       # ONE state

        left = face.render(to_avatar_pose(motion))   # adapter A
        mpfb.apply(motion)                           # adapter B
        bpy.context.view_layer.update()
        render_to(scene, tmp / "f.png")
        right = cv2.imread(str(tmp / "f.png"))
        right = cv2.resize(right, (640, 720))

        frame = np.hstack([left, right])
        for x, label in ((14, "2D face plate  <- adapters/face2d.py"),
                         (1294, "MPFB rig  <- adapters/mpfb.py")):
            cv2.rectangle(frame, (x - 8, 8), (x + 470, 42), (18, 18, 18), -1)
            cv2.putText(frame, label, (x, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 235, 255), 1)
        cv2.putText(frame, f"one HumanMotionState   t={i * dt:5.2f}s   "
                           f"head yaw {motion.head_world_yaw():+6.2f} deg",
                    (14, 706), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        w.write(frame)
        if i % 30 == 0 and i:
            el = time.perf_counter() - t0
            print(f"[split] {i}/{n}  {i / el:.2f} fps", flush=True)
    w.release()
    print(f"[split] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
