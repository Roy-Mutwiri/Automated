"""Build a precise mask of the plain wall surfaces in the approved master frame.

Phase 1 of surgical finishing. The master frame is approved and must never be
regenerated; only the wall material changes, and only where it is genuinely
wall.

## What must be excluded, and why each one matters

* **The person** - a slat line crossing his shoulder or walnut bleeding into
  his hair is the most conspicuous failure available. Excluded via segmentation
  and then *dilated*, so the exclusion is deliberately larger than the
  silhouette. Hair is the specific risk: segmentation under-covers flyaway
  strands, and the boundary is exactly where a generated texture would show.
* **The chair** - dark, adjacent to the person, and would read as wood-grained
  upholstery.
* **Monitors and screens** - bright or dark rectangles that must keep their own
  content.
* **The desk, microphone, gear** - foreground objects, nearer the lens.

## How the wall is found

Not by a semantic model - there is no "wall" class in the segmentation network
available here. The plain wall in this master has two properties that separate
it cleanly from everything else in frame: it is **low saturation** and it sits
in a **mid-to-high luminance band**. Every object in the room is either
noticeably darker (chair, monitors, gear, desk) or coloured (skin, screen
content).

So the mask is: low saturation AND mid-bright AND not-person AND not in the
foreground band, then morphologically cleaned to remove speckle and close small
holes.

That is a heuristic tuned to this specific frame, which is the correct scope -
this runs once, on one approved image, and the preview exists so a human checks
it before any pixels are generated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def person_mask(img_bgr: np.ndarray) -> np.ndarray:
    import torch
    import torchvision

    w8 = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        weights=w8).eval().cuda()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    batch = w8.transforms()(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
        probs = model(batch)["out"][0].softmax(0)[15].cpu().numpy()
    del model
    torch.cuda.empty_cache()
    h, w = img_bgr.shape[:2]
    return cv2.resize(probs.astype(np.float32), (w, h))


# Screens and hardware, read off a coordinate grid on the approved master.
#
# Hard-coded on purpose. This runs once, on one locked image, and a retoucher
# masking a specific plate would do exactly this. The saturation/luminance
# heuristic cannot separate them: the right-hand screen is desaturated and
# mid-bright, so it passed the "wall" test and would have been painted over
# with walnut - which is precisely the kind of damage this phase must not do.
#
# (x0, y0, x1, y1) in master-frame pixels, 1344x768.
EXCLUDE_RECTS = [
    (0, 15, 380, 205),        # left monitor, dark panel
    (415, 25, 885, 300),      # centre monitor, blue content
    (1040, 15, 1145, 275),    # speaker / tower
    (1170, 20, 1344, 305),    # right monitor, the AI figure
]


def build(img_bgr: np.ndarray,
          sat_max: int = 46,
          val_lo: int = 70,
          val_hi: int = 225,
          bottom_keep: float = 0.62,
          person_grow: float = 0.030) -> tuple[np.ndarray, dict]:
    """Return a float wall mask in [0,1] plus the intermediate layers."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]

    # Plain wall: desaturated and mid-bright.
    wall = ((sat < sat_max) & (val > val_lo) & (val < val_hi)).astype(np.uint8)

    # Foreground band: desk, gear, hands. Nearer the lens, never wall.
    wall[int(h * bottom_keep):, :] = 0

    # Screens and hardware. Excluded by explicit geometry, not by heuristic.
    for (ex0, ey0, ex1, ey1) in EXCLUDE_RECTS:
        pad = 6
        wall[max(ey0 - pad, 0):ey1 + pad, max(ex0 - pad, 0):ex1 + pad] = 0

    # Person, generously grown. Bigger than the silhouette on purpose - hair is
    # where a generated texture would betray itself.
    pm = person_mask(img_bgr)
    grow = max(int(person_grow * min(h, w)) | 1, 9)
    person = cv2.dilate((pm > 0.4).astype(np.uint8),
                        np.ones((grow, grow), np.uint8))
    wall[person > 0] = 0

    # Clean: drop speckle, close small holes, then keep only regions large
    # enough to be architecture rather than gaps between objects.
    wall = cv2.morphologyEx(wall, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(wall, 8)
    keep = np.zeros_like(wall)
    min_area = 0.004 * h * w
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    wall = keep

    # Feather. A hard mask edge produces a visible material seam; the join has
    # to fall off over enough pixels to read as the same surface.
    soft = cv2.GaussianBlur(wall.astype(np.float32),
                            (max(int(0.012 * min(h, w)) | 1, 11),) * 2, 0)
    return np.clip(soft, 0, 1), {"person": pm, "raw": wall}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_locked_original.png")
    ap.add_argument("--mask", default="02_wall_mask.png")
    ap.add_argument("--preview", default="wall_mask_preview.png")
    ap.add_argument("--sat-max", type=int, default=46)
    ap.add_argument("--val-lo", type=int, default=70)
    ap.add_argument("--val-hi", type=int, default=225)
    ap.add_argument("--bottom-keep", type=float, default=0.62)
    args = ap.parse_args()

    img = cv2.imread(args.source)
    if img is None:
        raise FileNotFoundError(args.source)

    mask, parts = build(img, args.sat_max, args.val_lo, args.val_hi,
                        args.bottom_keep)
    cv2.imwrite(args.mask, (mask * 255).astype(np.uint8))

    # Preview: original | mask | overlay. The overlay is what actually gets
    # judged - it shows whether the mask touches hair, chair or screens.
    overlay = img.copy().astype(np.float32)
    tint = np.zeros_like(overlay); tint[:, :, 1] = 255      # green = will change
    overlay = overlay * (1 - mask[..., None] * 0.45) + tint * (mask[..., None] * 0.45)
    strip = np.hstack([
        img,
        cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        np.clip(overlay, 0, 255).astype(np.uint8),
    ])
    scale = 1800.0 / strip.shape[1]
    cv2.imwrite(args.preview,
                cv2.resize(strip, (1800, int(strip.shape[0] * scale))))

    print(f"[wall] mask covers {100 * (mask > 0.5).mean():.1f}% of the frame")
    print(f"[wall] mask    -> {args.mask}")
    print(f"[wall] preview -> {args.preview}  (green = region that will change)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
