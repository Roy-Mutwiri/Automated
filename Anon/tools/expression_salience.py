"""How strong does an expression have to be before a viewer can see it?

The smile measured 0.3-0.6 px of landmark displacement at 1080p, which is
invisible. That number was also partly an artefact: it was a median over all 203
landmarks, and a smile moves the mouth, not the skull. Measuring the whole face
dilutes a local change until it disappears.

So each expression is measured **per region** - mouth corners, cheeks, brows,
lids - by taking only the landmarks that sit inside that region at rest, and
reporting how far they move.

Then it is rendered and watched, because the number decides which intensities
are worth looking at and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LEVELS = (0.2, 0.4, 0.6, 0.8)

# Bands of the *landmark bounding box*, not of a hand-picked head crop.
#
# The first version used fractions of a guessed head box and gave the brow
# region zero landmarks - the same mistake that once ranked expression latents
# by measuring the forehead. The landmarks occupy only x 0.12-0.66 of that box,
# so every fraction in it was wrong. Anchoring to the landmarks themselves
# cannot drift: whatever the crop, the point cloud is still a face.
#
# Bands read off a gridded render (landmark_map.png). The box runs brow-top to
# chin, so the brows are the very top of it, not a quarter of the way down.
REGIONS = {
    "brows":  (0.00, 0.10),
    "lids":   (0.10, 0.24),
    "cheeks": (0.25, 0.52),
    "mouth":  (0.55, 0.82),
}

EXPRESSIONS = ("SMALL_SMILE", "AMUSED", "SKEPTICAL", "SURPRISED", "FOCUSED")


def pose_for(name: str, level: float):
    """Build the face pose an expression produces at a given intensity."""
    from presenter.motion.expression import EXPRESSIONS as SPECS
    from presenter.types import AvatarPose

    spec = SPECS[name]
    peak = spec["peak"] * level
    a = spec.get("asymmetry", 0.14)
    p = AvatarPose()
    p.mouth_corner_l = peak * (1.0 + a)
    p.mouth_corner_r = peak * (1.0 - a)
    cheek = abs(peak) * spec.get("cheek", 0.0)
    p.cheek = cheek
    sq = abs(peak) * spec.get("squint", 0.0) if peak >= 0 else level * spec.get("squint", 0.0)
    if spec.get("squint", 0.0) < 0:
        sq = level * spec["squint"]
    p.squint_l = sq * (1.0 + a)
    p.squint_r = sq * (1.0 - a)
    brow = spec.get("brow", 0.0) * level
    split = spec.get("brow_split", 0.0)
    p.brow_l = brow * (1.0 + split)
    p.brow_r = brow * (1.0 - split)
    p.brow_furrow = spec.get("furrow", 0.0) * level
    # The head tilt is part of the expression's visible signature, not a
    # separate effect. Leaving it out of the measurement under-reported every
    # expression that leans on it - scepticism most of all.
    p.roll = spec.get("head_tilt", 0.0) * level
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_v04_final.png")
    ap.add_argument("--root", default="third_party/LivePortrait")
    ap.add_argument("--out", default="expression_salience.json")
    ap.add_argument("--sheet", default="expression_salience_sheet.png")
    ap.add_argument("--frames-dir", default="expr_frames")
    args = ap.parse_args()

    from presenter.render.liveportrait import LivePortraitRenderer
    from presenter.types import AvatarPose

    r = LivePortraitRenderer(
        source_image=args.source, liveportrait_root=args.root,
        output_size=(1920, 1080), framing="full", environment="source",
        neutralize_pose=0.0)

    sys.path.insert(0, str(Path(args.root).resolve()))
    from src.utils.human_landmark_runner import LandmarkRunner
    runner = LandmarkRunner(
        ckpt_path=str(Path(args.root) / "pretrained_weights/liveportrait/landmark.onnx"),
        onnx_provider="cpu", device_id=0)
    runner.warmup()

    HEAD = (790, 60, 1330, 620)

    def lmk(frame):
        x0, y0, x1, y1 = HEAD
        crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        sc = 512.0 / max(crop.shape[:2])
        crop = cv2.resize(crop, (int(crop.shape[1] * sc), int(crop.shape[0] * sc)))
        pts = runner.run(crop, runner.run(crop)).astype(np.float32) / sc
        pts[:, 0] += x0
        pts[:, 1] += y0
        return pts

    neutral = r.render(AvatarPose())
    base = lmk(neutral)

    fy0, fh = base[:, 1].min(), np.ptp(base[:, 1])
    masks = {}
    for name, (a, b) in REGIONS.items():
        masks[name] = ((base[:, 1] >= fy0 + a * fh) & (base[:, 1] < fy0 + b * fh))
    # A smile moves the corners of the mouth far more than the middle of the
    # lip, and averaging the two hides the only signal that matters.
    mo = masks["mouth"]
    xs = base[:, 0]
    lo, hi = np.quantile(xs[mo], 0.2), np.quantile(xs[mo], 0.8)
    masks["mouth_corner"] = mo & ((xs <= lo) | (xs >= hi))
    order = list(masks)
    for name, m in masks.items():
        print(f"[expr] region {name:<13}{int(m.sum()):3d} landmarks")

    out_dir = Path(args.frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    tiles = []

    print(f"\n[expr] {'expression':<13}{'lvl':>5}" +
          "".join(f"{k:>13}" for k in order) + f"{'peak':>8}")
    for name in EXPRESSIONS:
        for lv in LEVELS:
            frame = r.render(pose_for(name, lv))
            pts = lmk(frame)
            d = np.linalg.norm(pts - base, axis=1)
            # p90, not mean: a region is visible if its most-displaced points
            # move, and a mean over a whole band buries them.
            row = {k: float(np.quantile(d[m], 0.9)) if m.any() else 0.0
                   for k, m in masks.items()}
            peak = float(d.max())
            results.append(dict(expression=name, level=lv, peak_px=peak, **row))
            print(f"[expr] {name:<13}{lv:>5.1f}" +
                  "".join(f"{row[k]:>13.2f}" for k in order) + f"{peak:>8.1f}")
            cv2.imwrite(str(out_dir / f"{name}_{int(lv*100)}.png"), frame)
            tiles.append((f"{name} {int(lv*100)}%", frame))

    Path(args.out).write_text(json.dumps(results, indent=2))

    cells = []
    for label, frame in tiles:
        c = frame[140:600, 830:1300]
        c = cv2.resize(c, (250, int(250 * c.shape[0] / c.shape[1])))
        c = cv2.copyMakeBorder(c, 22, 3, 2, 2, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(c, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 235, 255), 1)
        cells.append(c)
    rows = [np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)]
    cv2.imwrite(args.sheet, np.vstack(rows))
    print(f"\n[expr] sheet -> {args.sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
