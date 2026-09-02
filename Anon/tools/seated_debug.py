"""Scripted seated-mechanics clip, three cameras at once.

Each moment is rendered from the front, three-quarter and high angles
simultaneously rather than in three passes, because the whole point is to check
one pose against three views: a hand that looks like it is on the desk from the
front is only actually on it if the high angle agrees.

The script drives the same public seams the content pipeline will use -
`attention.request` and the posture engagement - so nothing here is a special
debug path that could diverge from what really runs.
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

# (start seconds, label, action)
SCRIPT = [
    (0.0,  "neutral / breathing",   None),
    (5.0,  "look left",             ("attend", "CHAT")),
    (9.0,  "look right",            ("attend", "SECOND_DISPLAY")),
    (14.0, "look down at desk",     ("attend", "DESK")),
    (18.0, "return to lens",        ("attend", "LENS")),
    (22.0, "small lean forward",    ("engage", 0.85)),
    (28.0, "settle back",           ("engage", -0.95)),
    (35.0, "hand to lap",           ("hand_l", "lap_rest_l")),
    (40.0, "hand back to desk",     ("hand_l", "desk_rest_l")),
    (45.0, "mouse reposition",      ("mouse", 0.9)),
    (49.0, "neutral",               ("engage", 0.0)),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="assets/rig/body_rig.blend")
    ap.add_argument("--seconds", type=float, default=54.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", type=int, default=460)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--frames-dir", default="seated_debug_frames")
    args = ap.parse_args()

    import bpy
    from mathutils import Vector
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.rig).resolve()))

    from render_body import CAMERAS, _look_at, render_to, setup_scene
    from presenter.behavior.engine import BehaviorEngine
    from presenter.motion.adapters.mpfb import MPFBAdapter
    from presenter.motion.rig_geometry import (SeatedGeometry, load_joints,
                                               mh_to_blender)

    rig = bpy.data.objects["body_rig"]
    adapter = MPFBAdapter(rig)
    scene = setup_scene("front", args.size, args.size, args.samples)

    g = SeatedGeometry.measure(load_joints())
    span = g.shoulder_y - g.floor_y
    cam = bpy.data.objects["cam"]
    views = ("front", "three_q", "high")

    def place(name):
        sx, fz, hy, look_hy, lens = CAMERAS[name]
        cam.location = Vector(mh_to_blender(
            (sx * span, g.seat_y + hy * span, g.seat_front_z + fz * span)))
        _look_at(cam, mh_to_blender(
            (0.0, g.seat_y + look_hy * span, g.seat_back_z + 0.20 * span)))
        bpy.data.cameras["cam"].lens = lens

    engine = BehaviorEngine(seed=20260902)
    dt = 1.0 / args.fps
    out = Path(args.frames_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)

    pending = list(SCRIPT)
    label = "neutral"
    mouse_shift = 0.0
    t0 = time.perf_counter()
    n = int(args.seconds * args.fps)

    for i in range(n):
        t = i * dt
        while pending and t >= pending[0][0]:
            _, label, action = pending.pop(0)
            if action:
                kind, value = action
                if kind == "attend":
                    engine.attention.request(value, 4.5)
                elif kind == "engage":
                    engine.body._engagement_target = value
                elif kind == "hand_l":
                    engine.body._left_rest = value
                elif kind == "mouse":
                    mouse_shift = value
            print(f"[seated] t={t:5.1f}s  {label}", flush=True)

        engine.update(dt)

        # A small mouse reposition: the target itself moves, and the arm
        # follows through IK rather than the hand being animated.
        tgt = bpy.data.objects.get("target_mouse")
        if tgt is not None and mouse_shift:
            base = adapter._rest_target["mouse"]
            k = min((t - 45.0) / 1.2, 1.0) if t > 45.0 else 0.0
            tgt.location = (base[0] + 0.55 * k * mouse_shift,
                            base[1] - 0.35 * k * mouse_shift, base[2])

        adapter.apply(engine.motion)
        bpy.context.view_layer.update()

        panels = []
        for v in views:
            place(v)
            render_to(scene, tmp / "f.png")
            panels.append(cv2.imread(str(tmp / "f.png")))

        frame = np.hstack(panels)
        m = engine.motion
        bar = np.full((66, frame.shape[1], 3), 18, np.uint8)
        cv2.putText(bar, f"{t:5.1f}s   {label}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 235, 255), 1)
        cv2.putText(bar,
                    f"breath {m.breathing.drive:4.2f}  chest {m.chest.rx:+5.2f}  "
                    f"engage {m.posture.engagement:+5.2f}  "
                    f"back_contact {m.posture.back_contact:4.2f}  "
                    f"attend {m.attention.target}",
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1)
        for k, v in enumerate(views):
            cv2.putText(frame, v, (k * args.size + 10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 235, 255), 1)
        cv2.imwrite(str(out / f"f{i:05d}.png"), np.vstack([bar, frame]))

        if i % 30 == 0 and i:
            el = time.perf_counter() - t0
            print(f"[seated] {i}/{n}  {i / el:.2f} fps  "
                  f"eta {(n - i) / max(i / el, 1e-3) / 60:.1f} min", flush=True)

    print(f"[seated] {n} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
