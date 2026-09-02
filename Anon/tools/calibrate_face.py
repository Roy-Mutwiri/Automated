"""Measure which expression dimensions drive the mouth, cheeks, lids and jaw.

The gaze calibration established the method and the reason for it: this latent
has no published semantic map, the inherited guesses were wrong in ways that
looked plausible, and the only honest way to read it is to drive every
dimension and look at the pixels. `brow_r` had been mapped for months to a
dimension that does nothing at all.

So nothing here guesses which dimension is "smile". All 21 keypoints are driven
on all three axes, and the regions that move are measured.

## Regions

Finer than the gaze pass needed. A smile is not "the mouth moved" - it is the
corners going up *while the cheeks lift*, and if the cheeks do not move it
reads as a rictus. So the mouth is split into corners, upper lip and lower lip,
the cheeks are separate boxes, and the lower lids have their own boxes because
the Duchenne component lives there.

All boxes were read off a gridded render of the actual 512x512 crop. Measuring
the wrong box produces confident, precise, wrong numbers - which happened once
already, when an assumed eye box turned out to be sitting on the forehead.

## What comes out

For every dimension and axis: which region moved most, how selectively, and how
much collateral motion there was. Plus a contact sheet, because the numbers can
only rank candidates - whether a mouth corner went *up* rather than sideways is
a question for a person looking at the image.

Usage
-----
    python tools/calibrate_face.py --scan
    python tools/calibrate_face.py --sheet 20 17 19 --axis 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Fractions of the generated 512x512 crop. Subject-left is image-right.
REGIONS = {
    "brow_r":     (0.28, 0.355, 0.45, 0.425),
    "brow_l":     (0.47, 0.355, 0.64, 0.425),
    "lid_r":      (0.30, 0.485, 0.43, 0.535),
    "lid_l":      (0.49, 0.485, 0.62, 0.535),
    "cheek_r":    (0.28, 0.525, 0.40, 0.625),
    "cheek_l":    (0.52, 0.525, 0.64, 0.625),
    "corner_r":   (0.355, 0.635, 0.435, 0.715),
    "corner_l":   (0.505, 0.635, 0.585, 0.715),
    "upper_lip":  (0.425, 0.618, 0.515, 0.668),
    "lower_lip":  (0.425, 0.672, 0.515, 0.722),
    "jaw":        (0.38, 0.760, 0.60, 0.880),
    "nose":       (0.435, 0.555, 0.505, 0.618),
}

# Regions we actually want a control for.
WANTED = ("corner_l", "corner_r", "cheek_l", "cheek_r", "lid_l", "lid_r",
          "brow_l", "brow_r", "upper_lip", "lower_lip", "jaw")


def box(shape, key):
    h, w = shape[:2]
    x0, y0, x1, y1 = REGIONS[key]
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def region_mean(diff, shape, key):
    x0, y0, x1, y1 = box(shape, key)
    return float(diff[y0:y1, x0:x1].mean())


def scan(probe, amount, out_dir):
    from presenter.types import AvatarPose

    neutral = probe.render_crop(AvatarPose())
    ng = cv2.cvtColor(neutral, cv2.COLOR_BGR2GRAY).astype(np.float32)

    rows = []
    print(f"[face] scanning 21 keypoints x 3 axes at {amount:+.3f}")
    print(f"[face] {'idx':>3} {'ax':>3} " +
          " ".join(f"{k[:9]:>9}" for k in WANTED) + f" {'best':>10} {'sel':>5}")
    for idx in range(21):
        for axis in range(3):
            frame = probe.render_raw_delta(idx, axis, amount)
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            d = np.abs(g - ng)
            vals = {k: region_mean(d, frame.shape, k) for k in REGIONS}
            best = max(WANTED, key=lambda k: vals[k])
            others = np.mean([vals[k] for k in REGIONS if k != best])
            sel = vals[best] / max(others, 1e-4)
            rows.append(dict(index=idx, axis=axis, best=best,
                             selectivity=sel, **vals))
            if vals[best] > 1.2:
                print(f"[face] {idx:>3} {axis:>3} " +
                      " ".join(f"{vals[k]:>9.2f}" for k in WANTED) +
                      f" {best:>10} {sel:>5.2f}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "face_scan.json").write_text(json.dumps(rows, indent=2))

    print("\n[face] best candidate per wanted control:")
    for k in WANTED:
        cands = sorted(rows, key=lambda r: -(r[k] / max(
            np.mean([r[o] for o in REGIONS if o != k]), 1e-4)))
        top = cands[0]
        others = np.mean([top[o] for o in REGIONS if o != k])
        print(f"[face]   {k:<11} idx {top['index']:>2} axis {top['axis']}  "
              f"effect {top[k]:6.2f}  selectivity {top[k] / max(others, 1e-4):5.2f}")
    return rows


def sheet(probe, indices, axis, amount, out_dir):
    from presenter.types import AvatarPose

    neutral = probe.render_crop(AvatarPose())
    rows = []
    for idx in indices:
        tiles = []
        for amt, lab in ((-amount, f"i{idx}a{axis} -"), (0.0, "neutral"),
                         (amount, f"i{idx}a{axis} +")):
            f = neutral if amt == 0.0 else probe.render_raw_delta(idx, axis, amt)
            h, w = f.shape[:2]
            c = f[int(0.50 * h):int(0.92 * h), int(0.26 * w):int(0.68 * w)]
            c = cv2.resize(c, (0, 0), fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)
            c = cv2.copyMakeBorder(c, 26, 4, 4, 4, cv2.BORDER_CONSTANT,
                                   value=(20, 20, 20))
            cv2.putText(c, lab, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 235, 255), 1)
            tiles.append(c)
        rows.append(np.hstack(tiles))
    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(20, 20, 20)) for r in rows]
    out = Path(out_dir) / "expression_calibration_contact_sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"[face] contact sheet -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_v04_final.png")
    ap.add_argument("--root", default="third_party/LivePortrait")
    ap.add_argument("--amount", type=float, default=0.030)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--sheet", type=int, nargs="+", default=None)
    ap.add_argument("--axis", type=int, default=1)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "calexp", str(Path(__file__).with_name("calibrate_expression.py")))
    calexp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calexp)

    probe = calexp.Probe(args.source, args.root)
    if args.sheet:
        sheet(probe, args.sheet, args.axis, args.amount, args.out_dir)
    else:
        scan(probe, args.amount, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
