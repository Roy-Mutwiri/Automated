"""The same attention events at three amplitudes, side by side.

The point is to find the threshold where "movement exists" becomes "I can see
where he is looking". The middle column is the current tuning; the outer two
are the same behaviour with head recruitment scaled, rendered from the identical
motion state so nothing else differs.

The clip carries no labels on purpose. If the event is not readable without one,
it is not strong enough.
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

EVENTS = [
    (0.0,  "CAMERA_LENS",          3.0),
    (3.0,  "MAIN_MONITOR_CENTER",  2.2),   # tiny side/down glance
    (5.2,  "CAMERA_LENS",          2.0),
    (7.2,  "CHAT_REGION",          1.6),   # quick check
    (8.8,  "CAMERA_LENS",          1.8),
    (10.6, "MAIN_MONITOR_LOWER",   5.0),   # sustained read
    (15.6, "DESK_MOUSE",           2.2),   # monitor -> desk
    (17.8, "CAMERA_LENS",          2.4),   # desk -> camera
    (20.2, "SECONDARY_MONITOR",    4.2),   # large turn
    (24.4, "CAMERA_LENS",          3.6),   # long focus -> return
]
AMPLITUDES = (1.0, 1.5, 2.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="assets/rig/body_rig.blend")
    ap.add_argument("--seconds", type=float, default=29.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", type=int, default=440)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--frames-dir", default="salience_frames")
    args = ap.parse_args()

    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.rig).resolve()))

    from render_body import render_to, setup_scene
    from presenter.behavior.engine import BehaviorEngine
    from presenter.motion.adapters.mpfb import MPFBAdapter
    import presenter.behavior.attention as A

    rig = bpy.data.objects["body_rig"]
    adapter = MPFBAdapter(rig)
    # Framed on the head and shoulders, not the whole room.
    #
    # The first pass used the wide three-quarter view and even a 28 degree head
    # turn read weakly, because the head was a small part of the frame. Salience
    # is a property of the *shot* as much as of the motion, and the shot that
    # matters is the one the audience sees - which is a streamer framed from the
    # chest up.
    scene = setup_scene("torso", args.size, args.size, args.samples)

    n = int(args.seconds * args.fps)
    dt = 1.0 / args.fps
    out = Path(args.frames_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)

    # One engine per amplitude, all driven through the identical script, so the
    # only difference between columns is head recruitment.
    engines = []
    base_max = A.HEAD_SHARE_MAX
    for amp in AMPLITUDES:
        engines.append(dict(engine=BehaviorEngine(seed=4242),
                            amp=amp, pending=list(EVENTS)))

    t0 = time.perf_counter()
    for i in range(n):
        t = i * dt
        panels = []
        for cfg in engines:
            A.HEAD_SHARE_MAX = min(base_max * cfg["amp"], 0.98)
            e = cfg["engine"]
            while cfg["pending"] and t >= cfg["pending"][0][0]:
                _, target, dwell = cfg["pending"].pop(0)
                e.attention.request(target, dwell)
            e.update(dt)
            adapter.apply(e.motion)
            bpy.context.view_layer.update()
            render_to(scene, tmp / "f.png")
            panels.append(cv2.imread(str(tmp / "f.png")))
        A.HEAD_SHARE_MAX = base_max

        frame = np.hstack(panels)
        strip = np.full((30, frame.shape[1], 3), 16, np.uint8)
        for k, amp in enumerate(AMPLITUDES):
            cv2.putText(strip, f"x{amp:.1f}", (k * args.size + 12, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        cv2.imwrite(str(out / f"f{i:05d}.png"), np.vstack([strip, frame]))

        if i % 30 == 0 and i:
            el = time.perf_counter() - t0
            print(f"[salience] {i}/{n}  {i / el:.2f} fps  "
                  f"eta {(n - i) / max(i / el, 1e-3) / 60:.1f} min", flush=True)

    print(f"[salience] {n} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
