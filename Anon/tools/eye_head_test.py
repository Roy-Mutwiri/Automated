"""Scripted eye-head coordination test: four shifts, chosen to expose the failure.

The review found a 19 degree glance recruiting 3 degrees of head. The fix made
recruitment depend on intent as well as geometry, and the point of this clip is
that the *same angle* now produces different behaviour depending on why he is
looking.

    0-4 s    lens, settling
    4 s      tiny shift within the lens zone      - eyes only, no neck at all
    9 s      quick glance at chat                 - eyes lead, head barely
    15 s     sustained read of the second display - head follows substantially
    26 s     return to the lens                   - eyes lead the return

Rendered on the body rig rather than the 2D face, because the neck is the thing
under test and the 2D renderer has no neck to show.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SCRIPT = [
    (4.0,  "MAIN_DISPLAY",   1.6, "tiny shift, eyes only"),
    (9.0,  "CHAT",           0.9, "quick glance"),
    (15.0, "SECOND_DISPLAY", 9.0, "sustained read"),
    (26.0, "LENS",           6.0, "return to camera"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="assets/rig/body_rig.blend")
    ap.add_argument("--camera", default="three_q")
    ap.add_argument("--seconds", type=float, default=33.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=854)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--frames-dir", default="eyehead_frames")
    ap.add_argument("--csv", default="eye_head_coordination.csv")
    args = ap.parse_args()

    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.rig).resolve()))

    from render_body import render_to, setup_scene
    from presenter.behavior.engine import BehaviorEngine
    from presenter.motion.adapters.mpfb import MPFBAdapter

    rig = bpy.data.objects["body_rig"]
    adapter = MPFBAdapter(rig)
    scene = setup_scene(args.camera, args.width, args.height, args.samples)

    engine = BehaviorEngine(seed=5)
    dt = 1.0 / args.fps
    out = Path(args.frames_dir)
    out.mkdir(parents=True, exist_ok=True)

    pending = list(SCRIPT)
    rows = ["t,event,target,eye_az,head_yaw,neck_ry,head_ry,total_az"]
    t0 = time.perf_counter()
    n = int(args.seconds * args.fps)
    for i in range(n):
        t = i * dt
        label = ""
        if pending and t >= pending[0][0]:
            when, target, dwell, label = pending.pop(0)
            engine.attention.request(target, dwell)
            print(f"[eyehead] t={t:5.1f}s  {target:<15} {label}", flush=True)
        engine.update(dt)
        a = engine.attention
        m = engine.motion
        rows.append(f"{t:.3f},{label},{a.current},{a.eye_az:.3f},"
                    f"{a.head_yaw:.3f},{m.neck.ry:.3f},{m.head.ry:.3f},"
                    f"{a.eye_az + a.head_yaw:.3f}")
        adapter.apply(m)
        bpy.context.view_layer.update()
        render_to(scene, out / f"f{i:05d}.png")
        if i % 60 == 0 and i:
            el = time.perf_counter() - t0
            print(f"[eyehead] {i}/{n}  {i / el:.2f} fps", flush=True)

    Path(args.csv).write_text("\n".join(rows))
    print(f"[eyehead] {n} frames -> {out}, trace -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
