"""How much visible motion can the current 2D avatar actually produce?

Layer B of the two-layer problem. The behaviour engine can describe a body; this
measures what the photoreal renderer is able to *show* of it, and where it
breaks.

Two numbers per pose, both at 1920x1080:

* **displacement** - how far facial landmarks move on screen. A pose that is
  physically real and moves four pixels is perceptually absent.
* **distortion** - how much the region *outside* the head changes. The head
  crop is pasted into a fixed plate, so a large drive that starts smearing the
  hair line or dragging the ear shows up here as change where there should be
  none.

The sweep deliberately runs past the useful range. The point is to find the
ceiling, not to confirm a value already chosen.

Usage
-----
    python tools/avatar_visibility.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

YAW_SWEEP = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 26.0, 32.0)
PITCH_SWEEP = (0.0, -4.0, -8.0, -12.0, -16.0, -22.0)
GAZE_SWEEP = (0.0, 0.12, 0.24, 0.36, 0.48)


def landmarks(runner, frame, box):
    x0, y0, x1, y1 = box
    crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    s = 512.0 / max(crop.shape[:2])
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
    lmk = runner.run(crop, runner.run(crop)).astype(np.float32) / s
    lmk[:, 0] += x0
    lmk[:, 1] += y0
    return lmk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_v04_final.png")
    ap.add_argument("--root", default="third_party/LivePortrait")
    ap.add_argument("--out", default="current_avatar_visibility.json")
    ap.add_argument("--sheet", default="avatar_visibility_sheet.png")
    ap.add_argument("--frames-dir", default="avatar_vis_frames")
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

    neutral = r.render(AvatarPose())
    # The head sits in the upper-middle of this plate; a generous box so the
    # landmark pass has hair and jaw for context.
    box = (760, 40, 1360, 560)
    base_lmk = landmarks(runner, neutral, box)

    # Region that must not change: everything outside the animated crop.
    lock = np.ones(neutral.shape[:2], bool)
    lock[0:620, 620:1420] = False

    out_dir = Path(args.frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    tiles = []

    def probe(label, pose, tag):
        frame = r.render(pose)
        lmk = landmarks(runner, frame, box)
        disp = float(np.median(np.linalg.norm(lmk - base_lmk, axis=1)))
        peak = float(np.max(np.linalg.norm(lmk - base_lmk, axis=1)))
        outside = int(np.abs(frame[lock].astype(int)
                             - neutral[lock].astype(int)).max())
        # Smearing shows as lost high-frequency detail in the face region.
        g = cv2.cvtColor(frame[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).var())
        results.append(dict(kind=tag, label=label, median_px=disp, peak_px=peak,
                           outside_max=outside, sharpness=sharp))
        cv2.imwrite(str(out_dir / f"{tag}_{label}.png"), frame)
        return frame, disp, peak, sharp

    base_sharp = None
    print(f"[vis] {'pose':<22}{'median px':>11}{'peak px':>10}{'sharp':>9}"
          f"{'outside':>9}")
    for y in YAW_SWEEP:
        f, d, p, s = probe(f"yaw{y:+.0f}", AvatarPose(yaw=y), "yaw")
        if base_sharp is None:
            base_sharp = s
        print(f"[vis] head yaw {y:+5.0f} deg      {d:>10.1f}{p:>10.1f}"
              f"{s:>9.0f}{results[-1]['outside_max']:>9d}")
        tiles.append((f"yaw {y:+.0f}", f))
    for pt in PITCH_SWEEP[1:]:
        f, d, p, s = probe(f"pitch{pt:+.0f}", AvatarPose(pitch=pt), "pitch")
        print(f"[vis] head pitch {pt:+5.0f} deg    {d:>10.1f}{p:>10.1f}"
              f"{s:>9.0f}{results[-1]['outside_max']:>9d}")
        tiles.append((f"pitch {pt:+.0f}", f))
    for gx in GAZE_SWEEP[1:]:
        f, d, p, s = probe(f"gaze{gx:.2f}", AvatarPose(gaze_x=gx), "gaze")
        print(f"[vis] gaze_x {gx:.2f}          {d:>10.1f}{p:>10.1f}"
              f"{s:>9.0f}{results[-1]['outside_max']:>9d}")
        tiles.append((f"gaze {gx:.2f}", f))
    for sm in (0.2, 0.4, 0.7):
        f, d, p, s = probe(f"smile{sm:.1f}",
                           AvatarPose(mouth_corner_l=sm, mouth_corner_r=sm * 0.85),
                           "smile")
        print(f"[vis] smile {sm:.1f}           {d:>10.1f}{p:>10.1f}"
              f"{s:>9.0f}{results[-1]['outside_max']:>9d}")
        tiles.append((f"smile {sm:.1f}", f))

    Path(args.out).write_text(json.dumps(
        dict(baseline_sharpness=base_sharp, poses=results), indent=2))

    cells = []
    for label, frame in tiles:
        c = frame[60:600, 700:1420]
        c = cv2.resize(c, (300, int(300 * c.shape[0] / c.shape[1])))
        c = cv2.copyMakeBorder(c, 24, 4, 3, 3, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(c, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 235, 255), 1)
        cells.append(c)
    rows = [np.hstack(cells[i:i + 6]) for i in range(0, len(cells), 6)]
    W = max(x.shape[1] for x in rows)
    rows = [cv2.copyMakeBorder(x, 0, 0, 0, W - x.shape[1], cv2.BORDER_CONSTANT,
                               value=(20, 20, 20)) for x in rows]
    cv2.imwrite(args.sheet, np.vstack(rows))
    print(f"[vis] sheet -> {args.sheet}, data -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
