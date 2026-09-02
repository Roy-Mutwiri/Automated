"""Score generated master-frame candidates and reject the bad ones automatically.

Separated from generation so a batch can be re-scored without re-generating, and
so the scoring model loads once for the whole batch rather than per image.

## The pose gate is the point

The previous master frame had a **27.5 degree** source yaw - the man was
addressing a point well off camera, and `neutralize_pose` can only partly
recover that before LivePortrait starts hallucinating the parts of the head the
rotation reveals. Driving a bad source pose is not a fix; rejecting it is.

Yaw here is not estimated from landmark geometry. It is read from
**LivePortrait's own motion extractor** - the exact quantity the renderer will
later drive - so the number the gate tests is the number that matters, measured
by the same model.

Gate: reject |yaw| > 10 deg, prefer < 8. Also reject large pitch or roll.

## What is and is not automatable

Automated: person present, face present and plausibly sized, head yaw/pitch/
roll, face crop size relative to LivePortrait's 512 px output.

Not automated, and left to visual inspection: whether the wall reads as wood or
textile, whether the chair is a gaming chair or an office chair, whether the
room says streamer or podcast. Those are the failures this iteration is about,
and pretending a script can judge them would be worse than admitting it cannot.

Usage
-----
    python tools/evaluate_scenes.py scenes/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Gate thresholds, from the brief.
YAW_REJECT = 10.0
YAW_PREFER = 8.0
PITCH_REJECT = 14.0
ROLL_REJECT = 10.0
FACE_MIN_PX = 120.0
PERSON_MIN = 0.03


def person_coverage(img_bgr: np.ndarray, model, weights) -> float:
    import torch

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    batch = weights.transforms()(
        torch.from_numpy(rgb).permute(2, 0, 1)
    ).unsqueeze(0).cuda()
    with torch.no_grad():
        probs = model(batch)["out"][0].softmax(0)[15].cpu().numpy()
    probs = cv2.resize(probs.astype(np.float32), (img_bgr.shape[1], img_bgr.shape[0]))
    return float((probs > 0.5).mean()), probs


def head_crop(img_rgb: np.ndarray, probs: np.ndarray):
    """Same person-guided pre-crop the renderer uses, so measurements match."""
    h, w = img_rgb.shape[:2]
    ys, xs = np.where(probs > 0.5)
    if len(xs) < 500:
        return img_rgb, (0.0, 0.0), 1.0
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    ph, pw = y1 - y0, x1 - x0
    cy, cx = y0 + ph * 0.16, x0 + pw * 0.5
    half = max(pw * 0.42, ph * 0.24, 96.0)
    cx0 = int(np.clip(cx - half, 0, w - 1)); cx1 = int(np.clip(cx + half, cx0 + 32, w))
    cy0 = int(np.clip(cy - half, 0, h - 1)); cy1 = int(np.clip(cy + half, cy0 + 32, h))
    crop = img_rgb[cy0:cy1, cx0:cx1]
    scale = 1.0
    if max(crop.shape[:2]) < 512:
        scale = 512.0 / max(crop.shape[:2])
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                          interpolation=cv2.INTER_CUBIC)
    return crop, (float(cx0), float(cy0)), scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--liveportrait-root", default="third_party/LivePortrait")
    ap.add_argument("--sheet", default="evaluation_sheet.png")
    args = ap.parse_args()

    root = Path(args.liveportrait_root).resolve()
    sys.path.insert(0, str(root))

    import torch
    import torchvision
    from src.config.inference_config import InferenceConfig
    from src.live_portrait_wrapper import LivePortraitWrapper
    from src.utils.crop import crop_image
    from src.utils.human_landmark_runner import LandmarkRunner

    files = sorted(Path(args.directory).glob("scene_*.png"))
    if not files:
        print(f"no scene_*.png in {args.directory}")
        return 1

    seg_w = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    seg = torchvision.models.segmentation.deeplabv3_resnet101(
        weights=seg_w).eval().cuda()

    weights = root / "pretrained_weights"
    cfg = InferenceConfig(
        checkpoint_F=str(weights / "liveportrait/base_models/appearance_feature_extractor.pth"),
        checkpoint_M=str(weights / "liveportrait/base_models/motion_extractor.pth"),
        checkpoint_G=str(weights / "liveportrait/base_models/spade_generator.pth"),
        checkpoint_W=str(weights / "liveportrait/base_models/warping_module.pth"),
        checkpoint_S=str(weights / "liveportrait/retargeting_models/stitching_retargeting_module.pth"),
        flag_use_half_precision=True, device_id=0,
    )
    wrapper = LivePortraitWrapper(inference_cfg=cfg)
    runner = LandmarkRunner(
        ckpt_path=str(weights / "liveportrait/landmark.onnx"),
        onnx_provider="cpu", device_id=0)
    runner.warmup()

    print(f"\n{'file':<26} {'person':>7} {'face':>7} {'crop':>6} "
          f"{'yaw':>7} {'pitch':>7} {'roll':>7}  verdict")
    print("-" * 84)

    rows = []
    for f in files:
        img = cv2.imread(str(f))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cover, probs = person_coverage(img, seg, seg_w)
        if cover < PERSON_MIN:
            print(f"{f.name:<26} {cover * 100:6.1f}% {'-':>7} {'-':>6} "
                  f"{'-':>7} {'-':>7} {'-':>7}  NO PERSON")
            rows.append((f, cover, 0, 0, 999, 999, 999, "NO PERSON"))
            continue

        crop_rgb, off, sc = head_crop(rgb, probs)
        try:
            lmk = runner.run(crop_rgb, runner.run(crop_rgb))
        except Exception:
            print(f"{f.name:<26} {cover * 100:6.1f}%  no face")
            rows.append((f, cover, 0, 0, 999, 999, 999, "NO FACE"))
            continue
        lmk = lmk.astype(np.float32) / sc
        lmk[:, 0] += off[0]; lmk[:, 1] += off[1]
        face_px = float(lmk[:, 0].max() - lmk[:, 0].min())

        # Motion extractor: the same pitch/yaw/roll the renderer will drive.
        crop = crop_image(rgb, lmk, dsize=512, scale=2.3, vx_ratio=0.0, vy_ratio=-0.125)
        crop_px = float(crop["img_crop"].shape[1])
        I = wrapper.prepare_source(cv2.resize(crop["img_crop"], (256, 256),
                                              interpolation=cv2.INTER_AREA))
        info = wrapper.get_kp_info(I)
        yaw = float(info["yaw"].item())
        pitch = float(info["pitch"].item())
        roll = float(info["roll"].item())

        if face_px < FACE_MIN_PX:
            verdict = "face too small"
        elif abs(yaw) > YAW_REJECT:
            verdict = f"REJECT yaw {abs(yaw):.0f}deg"
        elif abs(pitch) > PITCH_REJECT:
            verdict = f"REJECT pitch {abs(pitch):.0f}deg"
        elif abs(roll) > ROLL_REJECT:
            verdict = f"REJECT roll {abs(roll):.0f}deg"
        elif abs(yaw) <= YAW_PREFER:
            verdict = "PASS (preferred)"
        else:
            verdict = "pass"

        print(f"{f.name:<26} {cover * 100:6.1f}% {face_px:6.0f}p {crop_px:5.0f}p "
              f"{yaw:+7.1f} {pitch:+7.1f} {roll:+7.1f}  {verdict}")
        rows.append((f, cover, face_px, crop_px, yaw, pitch, roll, verdict))

    # Contact sheet, ordered best-first so the eye starts with the候補 worth
    # looking at. Sorting key: passing candidates by absolute yaw.
    def rank(r):
        _, cover, face, crop, yaw, pitch, roll, verdict = r
        bad = verdict.startswith(("NO", "REJECT", "face"))
        return (bad, abs(yaw))

    rows.sort(key=rank)
    cols, cw = 4, 460
    ch = int(cw * 768 / 1344)
    n = len(rows); nr = (n + cols - 1) // cols
    sheet = np.full((nr * ch, cols * cw, 3), 20, np.uint8)
    for i, r in enumerate(rows):
        f, cover, face, crop, yaw, pitch, roll, verdict = r
        im = cv2.imread(str(f))
        im = cv2.resize(im, (cw - 6, ch - 6))
        rr, cc = divmod(i, cols)
        sheet[rr * ch + 3:rr * ch + ch - 3, cc * cw + 3:cc * cw + cw - 3] = im
        good = not verdict.startswith(("NO", "REJECT", "face"))
        colour = (110, 255, 170) if good else (90, 90, 255)
        cv2.putText(sheet, f"{f.stem.replace('scene_', '')}  yaw {yaw:+.0f}",
                    (cc * cw + 12, rr * ch + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, colour, 2)
    cv2.imwrite(args.sheet, sheet)
    print(f"\nranked sheet -> {args.sheet}  (best first)")

    ok = [r for r in rows if not r[7].startswith(("NO", "REJECT", "face"))]
    print(f"{len(ok)}/{len(rows)} candidates passed the gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
