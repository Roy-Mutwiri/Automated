"""Finger curl at four intensities, both hands. The axis map's acceptance test.

Fingers are the check that the measured axis map is right, because a wrong axis
is unmistakable here and nearly invisible anywhere else: the hand fans open, or
twists, or the knuckles invert.

The map is not uniform - thumb and index hinge about local X, middle, ring and
little about local Z, with signs that differ between hands - so a single
hardcoded axis drives three fingers out of five sideways. That is what this
renders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LEVELS = (0.0, 0.25, 0.50, 0.75)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="assets/rig/body_rig.blend")
    ap.add_argument("--out", default="finger_axis_test.png")
    args = ap.parse_args()

    import bpy
    from mathutils import Vector
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.rig).resolve()))

    from render_body import render_to, setup_scene
    from presenter.behavior.engine import BehaviorEngine
    from presenter.motion.adapters.mpfb import MPFBAdapter

    rig = bpy.data.objects["body_rig"]
    adapter = MPFBAdapter(rig)
    scene = setup_scene("hands", 560, 560, 10)
    cam = bpy.data.objects["cam"]

    engine = BehaviorEngine(seed=3)
    for _ in range(120):
        engine.update(1.0 / 30.0)

    tmp = Path("finger_tmp")
    tmp.mkdir(exist_ok=True)
    rows = []
    for side, target in (("r", "target_mouse"), ("l", "target_desk_rest_l")):
        t = bpy.data.objects[target]
        cam.location = Vector((t.location.x - (1.6 if side == "r" else -1.6),
                               t.location.y - 4.2, t.location.z + 2.4))
        d = Vector(t.location) - cam.location
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        bpy.data.cameras["cam"].lens = 78

        tiles = []
        for lv in LEVELS:
            hand = engine.motion.hand_l if side == "l" else engine.motion.hand_r
            base = [0.26, 0.34, 0.31, 0.36, 0.42]
            hand.curl = [min(b + lv, 1.0) for b in base] if lv else base
            adapter.apply(engine.motion)
            bpy.context.view_layer.update()
            render_to(scene, tmp / "f.png")
            im = cv2.imread(str(tmp / "f.png"))
            im = cv2.copyMakeBorder(im, 28, 4, 3, 3, cv2.BORDER_CONSTANT,
                                    value=(20, 20, 20))
            label = f"{'right' if side == 'r' else 'left'} hand  " + \
                    (f"rest +{int(lv * 100)}%" if lv else "rest")
            cv2.putText(im, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 235, 255), 1)
            tiles.append(im)
        rows.append(np.hstack(tiles))

    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(20, 20, 20)) for r in rows]
    cv2.imwrite(args.out, np.vstack(rows))
    print(f"[finger] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
