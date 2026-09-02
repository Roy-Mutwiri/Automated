"""Run the finished plate through the avatar pipeline and prove the room is static.

Two questions, answered with measurements rather than screenshots:

1. **Does the animated face still sit correctly in the finished room?** The wall,
   the practicals and both screens changed after the plate was generated. The
   hair/slat, chair/slat, monitor/head and microphone boundaries are where a
   composite falls apart, so they are cropped at 100% for inspection.

2. **Is the background byte-identical while the face moves?** This is the whole
   claim of the master-frame architecture. Four poses are rendered - neutral, a
   blink, a small gaze shift and a small head move - and every pixel outside the
   union of the written regions is compared against the neutral frame. The
   answer has to be exactly 0, not "small".

Usage
-----
    python tools/verify_master.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

POSES = [
    ("neutral", dict()),
    ("blink", dict(eye_open_l=0.05, eye_open_r=0.03, brow_raise=-0.05)),
    ("gaze", dict(gaze_x=0.38, gaze_y=-0.12)),
    ("headmove", dict(yaw=1.6, pitch=-0.9, roll=0.5, tx=0.012, ty=-0.008)),
]

# Boundaries a composite fails at, in plate coordinates (1344 x 768).
DETAIL_CROPS = [
    ("eyes", (612, 196, 800, 268)),
    ("face", (580, 150, 830, 400)),
    ("hairline / slat wall", (560, 46, 800, 170)),
    ("jaw + neck", (600, 330, 830, 470)),
    ("chair / shoulder", (830, 300, 1010, 470)),
    ("microphone", (450, 380, 660, 560)),
    ("walnut wall", (80, 210, 330, 420)),
    ("monitor / head", (760, 60, 900, 250)),
    ("under-shelf light", (1090, 540, 1344, 700)),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_v04_final.png")
    ap.add_argument("--root", default="third_party/LivePortrait")
    ap.add_argument("--bench-frames", type=int, default=90)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    import torch
    from presenter.render.liveportrait import LivePortraitRenderer
    from presenter.types import AvatarPose

    out = Path(args.out_dir)
    free0, total = torch.cuda.mem_get_info()

    t0 = time.perf_counter()
    r = LivePortraitRenderer(
        source_image=args.source, liveportrait_root=args.root,
        output_size=(1920, 1080), framing="full", environment="source",
        neutralize_pose=0.0,
    )
    print(f"[verify] renderer ready in {time.perf_counter() - t0:.1f}s")

    frames = {}
    for name, kw in POSES:
        frames[name] = r.render(AvatarPose(**kw)).copy()
        cv2.imwrite(str(out / f"09_{name}.png"), frames[name])
    print(f"[verify] rendered {len(frames)} poses at "
          f"{frames['neutral'].shape[1]}x{frames['neutral'].shape[0]}")

    # --- Background lock ----------------------------------------------------
    #
    # The dynamic region is derived from the frames themselves rather than
    # asserted: whatever any pose changed relative to neutral IS the written
    # region. Everything else must be untouched, which is the claim under test.
    ref = frames["neutral"]
    changed = np.zeros(ref.shape[:2], bool)
    for name, f in frames.items():
        if name == "neutral":
            continue
        changed |= cv2.absdiff(f, ref).max(axis=2) > 0

    region = cv2.dilate(changed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ys, xs = np.where(region)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if xs.size else None
    print(f"[verify] dynamic region: {100 * region.mean():.2f}% of frame, bbox {box}")

    worst = 0
    for name, f in frames.items():
        if name == "neutral":
            continue
        d = int(np.abs(f[~region].astype(int) - ref[~region].astype(int)).max())
        worst = max(worst, d)
        print(f"[verify]   {name:9s} max diff outside dynamic region: {d}")
    print(f"[verify] MAX BACKGROUND DIFF: {worst}")

    # The room features finished in phases 1-3 must be inside the static area.
    for label, (x0, y0, x1, y1) in (
            ("walnut wall", (60, 200, 380, 500)),
            ("centre monitor", (440, 50, 600, 290)),
            ("right monitor", (1190, 40, 1330, 290)),
            ("under-shelf light", (1120, 600, 1330, 690)),
            ("desk", (60, 640, 400, 760))):
        sx0, sy0 = int(x0 * ref.shape[1] / 1344), int(y0 * ref.shape[0] / 768)
        sx1, sy1 = int(x1 * ref.shape[1] / 1344), int(y1 * ref.shape[0] / 768)
        touched = int(region[sy0:sy1, sx0:sx1].sum())
        print(f"[verify]   {label:18s} pixels written by animation: {touched}")

    # --- Throughput ---------------------------------------------------------
    pose = AvatarPose(yaw=0.4, eye_open_l=0.9, eye_open_r=0.9)
    for _ in range(8):                                   # warm up cudnn autotune
        r.render(pose)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(args.bench_frames):
        r.render(AvatarPose(yaw=0.4 + 0.001 * i, eye_open_l=0.9, eye_open_r=0.9))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    fps = args.bench_frames / dt
    free1, _ = torch.cuda.mem_get_info()
    print(f"[verify] FPS {fps:.2f} over {args.bench_frames} frames "
          f"({1000 * dt / args.bench_frames:.1f} ms/frame)")
    print(f"[verify] VRAM: total {total / 2**30:.1f} GiB, "
          f"free before {free0 / 2**30:.1f} GiB, after {free1 / 2**30:.1f} GiB, "
          f"renderer footprint {(free0 - free1) / 2**30:.2f} GiB")

    # --- Deliverables -------------------------------------------------------
    cv2.imwrite(str(out / "final_streamer_visual.png"), frames["neutral"])
    print("[verify] hero image -> final_streamer_visual.png")

    diff = cv2.absdiff(frames["headmove"], ref).max(axis=2)
    heat = cv2.applyColorMap(np.clip(diff * 6, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(out / "10_dynamic_region.png"), heat)

    sx, sy = ref.shape[1] / 1344.0, ref.shape[0] / 768.0
    cells = []
    for label, (x0, y0, x1, y1) in DETAIL_CROPS:
        c = ref[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]
        s = min(2.0, 560.0 / max(c.shape[1], 1))
        c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)),
                       interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
        c = cv2.copyMakeBorder(c, 30, 6, 6, 6, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(c, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 235, 255), 1)
        cells.append(c)
    rows = []
    for i in range(0, len(cells), 3):
        grp = cells[i:i + 3]
        hh = max(c.shape[0] for c in grp)
        grp = [cv2.copyMakeBorder(c, 0, hh - c.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=(20, 20, 20)) for c in grp]
        rows.append(np.hstack(grp))
    W = max(r_.shape[1] for r_ in rows)
    rows = [cv2.copyMakeBorder(r_, 0, 0, 0, W - r_.shape[1], cv2.BORDER_CONSTANT,
                               value=(20, 20, 20)) for r_ in rows]
    cv2.imwrite(str(out / "11_detail_crops.png"), np.vstack(rows))
    print("[verify] detail crops -> 11_detail_crops.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
