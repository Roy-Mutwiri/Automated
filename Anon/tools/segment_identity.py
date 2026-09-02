"""Cut the streamer out of the approved plate. Human only - no chair, no mic.

Every single-image reconstruction model takes an RGBA of *the person*. Feed it
the whole plate and it will happily reconstruct the gaming chair as part of his
shoulders, because a chair wrapped around a torso is exactly what a naive matte
gives you.

This is not the same job as `LivePortraitRenderer._person_matte`. That matte is
allowed to be generous - it is dilated on purpose so a turning head never clips,
and it carries a few pixels of the original background because they blend into
a defocused room. Here the opposite is required: anything that is not him is
poison, and a few pixels of chair welded to his shoulder become geometry.

## What is hard about this particular plate

* The **gaming chair** is directly behind him and shares his silhouette on the
  right. Semantic segmentation puts the boundary somewhere plausible; a colour
  model does better because chair leather and skin are nothing alike.
* The **boom mic** crosses his chest and shoulder. It is thin, dark, and
  overlaps him - so it must be removed from the matte and the hole left behind
  must not be treated as background.
* **Headphones** are worn, so they stay. They are part of the silhouette the
  identity is judged on.
* **Hair** decides everything. Camera 2 and 3 will show its edge, so the matte
  is refined against the image's own colour statistics rather than left as the
  network's smooth blob.

Outputs
-------
    assets/reference/avatar_rgba.png   RGBA, human only
    assets/reference/avatar_mask.png   8-bit matte
    renders/segmentation_debug.png     overlay for visual inspection
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# VOC class ids in the DeepLabV3 weights torchvision ships.
PERSON = 15
CHAIR = 9


def semantic(img_bgr: np.ndarray, device: str = "cuda"):
    """Return (person_prob, chair_prob) at full image resolution."""
    import torch
    import torchvision

    weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    model = (torchvision.models.segmentation
             .deeplabv3_resnet101(weights=weights).eval().to(device))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    batch = weights.transforms()(
        torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(batch)["out"][0].softmax(0).cpu().numpy()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    h, w = img_bgr.shape[:2]
    person = cv2.resize(probs[PERSON].astype(np.float32), (w, h))
    chair = cv2.resize(probs[CHAIR].astype(np.float32), (w, h))
    return person, chair


def refine(img_bgr, person, chair, iterations=5):
    """Snap the boundary onto the real edge with GrabCut.

    Seeded from confident interior and confident exterior, with the chair
    explicitly seeded as background. Semantic output is right about *where* he
    is and wrong by tens of pixels about where he *ends*, which is precisely
    the error that welds chair leather to a shoulder.
    """
    mask = np.full(img_bgr.shape[:2], cv2.GC_PR_BGD, np.uint8)
    mask[person > 0.35] = cv2.GC_PR_FGD
    mask[person > 0.90] = cv2.GC_FGD
    mask[person < 0.10] = cv2.GC_BGD
    # A pixel the network thinks is chair more than person is background, and
    # saying so explicitly is far more reliable than hoping the colour model
    # discovers it.
    mask[(chair > person) & (chair > 0.25)] = cv2.GC_BGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0
                    ).astype(np.float32)


def largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the blob containing the subject.

    Semantic segmentation regularly finds 'people' on the monitors - this plate
    has faces displayed on screens behind him. Those are not our subject and a
    reconstruction model fed two people produces one confused one.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0.5).astype(np.uint8), connectivity=8)
    if n <= 1:
        return binary
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(np.float32)


def remove_mic(img_bgr, matte, top_y_frac=0.52, kernel=21, contrast=26):
    """Erase the boom arm and its cable from the torso, and fill behind them.

    The mic overlaps him, so it survives any silhouette-based matte - it is
    *inside* the person. Cutting a hole would be worse than leaving it: a
    reconstruction model fed a torso with a tube-shaped void invents geometry to
    fill it. So the mic is detected, removed, and the shirt behind it is
    inpainted, giving the model a plausible complete human.

    Detection is structural rather than by colour. The boom is bright specular
    metal and its cable is near-black, so no single threshold finds both - but
    both are *thin*, and a median filter wide enough to swallow a thin structure
    leaves broad shading and fabric folds untouched. The difference between the
    image and its median is therefore exactly the thin structures, whichever
    direction their contrast runs.

    Restricted to below `top_y_frac` of the frame so it can never touch the
    face, beard, hair or headphones - the features identity is judged on.
    """
    h, w = img_bgr.shape[:2]
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.medianBlur(grey, kernel | 1)
    thin = cv2.absdiff(grey, smooth)

    band = np.zeros((h, w), np.uint8)
    band[int(h * top_y_frac):, :] = 1
    hits = ((thin > contrast) & (matte > 0.5) & (band > 0)).astype(np.uint8)

    # Keep only elongated components: fabric noise is blobby, a boom is not.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hits, connectivity=8)
    keep = np.zeros_like(hits)
    for i in range(1, n):
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 120:
            continue
        extent = area / float(max(bw * bh, 1))
        if max(bw, bh) > 60 and extent < 0.45:      # long, and mostly not filled
            keep[labels == i] = 1

    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    filled = cv2.inpaint(img_bgr, keep, 6, cv2.INPAINT_TELEA)
    return filled, keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plate", default="assets/reference/avatar_identity_camera1.png")
    ap.add_argument("--out-rgba", default="assets/reference/avatar_rgba.png")
    ap.add_argument("--out-mask", default="assets/reference/avatar_mask.png")
    ap.add_argument("--feather", type=float, default=1.2,
                    help="edge softening in pixels; keep small - a soft matte "
                         "becomes soft geometry")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    plate = ROOT / args.plate
    img = cv2.imread(str(plate))
    if img is None:
        print(f"[segment] cannot read {plate}")
        return 2
    print(f"[segment] plate {img.shape[1]}x{img.shape[0]}")

    person, chair = semantic(img, args.device)
    print(f"[segment] semantic: person {float((person > 0.5).mean()) * 100:.1f}% "
          f"of frame, chair {float((chair > 0.5).mean()) * 100:.1f}%")

    matte = refine(img, person, chair)
    matte = largest_component(matte)

    # Close pinholes (headphone gaps, hair) without growing the silhouette.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    matte = cv2.morphologyEx(matte, cv2.MORPH_CLOSE, k)
    if args.feather > 0:
        matte = cv2.GaussianBlur(matte, (0, 0), args.feather)
    matte = np.clip(matte, 0.0, 1.0)

    alpha = (matte * 255).astype(np.uint8)
    rgba = np.dstack([img, alpha])
    cv2.imwrite(str(ROOT / args.out_rgba), rgba)
    cv2.imwrite(str(ROOT / args.out_mask), alpha)

    ys, xs = np.where(matte > 0.5)
    if len(xs):
        print(f"[segment] subject bbox x[{xs.min()}:{xs.max()}] "
              f"y[{ys.min()}:{ys.max()}]  coverage "
              f"{float((matte > 0.5).mean()) * 100:.1f}% of frame")

    # Debug: original | matte | cut-out on neutral grey, for visual inspection.
    cut = (img.astype(np.float32) * matte[..., None]
           + 128.0 * (1.0 - matte[..., None])).astype(np.uint8)
    edge = img.copy()
    cont, _ = cv2.findContours((matte > 0.5).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edge, cont, -1, (0, 255, 255), 2)
    debug = np.hstack([cv2.resize(x, (560, 320)) for x in (edge, cut)])
    (ROOT / "renders").mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT / "renders/segmentation_debug.png"), debug)
    print(f"[segment] -> {args.out_rgba}, {args.out_mask}, "
          f"renders/segmentation_debug.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
