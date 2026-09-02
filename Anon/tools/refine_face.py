"""Stage 2: raise face detail in a chosen master frame without changing anything else.

Stage 1 (`generate_scene.py`) produces the whole scene - person, chair, desk,
room, lighting - in one coherent photograph. This stage takes a selected frame
and re-renders **only the head region** at much higher internal resolution, then
returns it to the frame.

## Why this exists

Room scale and face detail are in direct conflict in a single SDXL pass, and
four batches of prompt engineering could not resolve it:

* camera at 2.2 m -> the room looks superb, the head measures 94 px, unusable
* camera at 1.2 m -> the face is beautifully detailed, the room is a wall

Both are real requirements, so they have to be solved separately. The master
frame is static and generated once, so it can afford an expensive multi-stage
process that would be unthinkable per-frame.

## Why img2img at low strength, not inpainting

Inpainting *replaces* masked content. That is the wrong tool: it would invent a
new face, discarding the pose, lighting and identity that make the person belong
in the room - and belonging is the whole point of the master-frame architecture.

img2img at low denoise strength keeps the existing image as the starting point
and adds detail. At ~0.3 the geometry, expression, gaze direction, skin tone and
key light all survive; what changes is that the model resolves pores, lashes,
lip texture and individual hairs that were never present at 137 px.

**Strength is the critical parameter.** Too low does nothing; above ~0.45 it
starts inventing a different person. Default 0.30.

## Why the region is generous and the mask is soft

The crop takes hair, ears, neck and some shoulder, not a tight face box, for two
reasons: the model needs context to light the face consistently, and a soft
return mask needs somewhere to fade. A hard rectangle would leave a visible
patch - the exact "obvious compositing edge" the brief rules out.

Usage
-----
    python tools/refine_face.py assets/scene_master.png
    python tools/refine_face.py scenes/scene_03_seed903.png --strength 0.35 --out assets/scene_master.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Describes only what is being refined. No room, no composition - those already
# exist in the pixels and must not be re-imagined.
FACE_PROMPT = (
    "extreme close-up photo of a man's face, sharp detailed eyes, visible "
    "eyelashes, natural skin texture with pores and fine lines, detailed beard "
    "hair, natural lips, soft even lighting, photorealistic, raw photo"
)

FACE_NEGATIVE = (
    "plastic skin, airbrushed, beauty filter, oversmoothed, waxy, doll-like, "
    "cgi, 3d render, illustration, painting, different person, distorted face, "
    "asymmetric eyes, cross-eyed, extra teeth, blurry, low quality"
)


def head_region(img_bgr: np.ndarray, root: Path):
    """Locate a generous box around head, hair, neck and upper shoulders."""
    sys.path.insert(0, str(root))
    from src.utils.human_landmark_runner import LandmarkRunner

    import torch
    import torchvision

    # Person segmentation first, then landmarks on that crop - the same
    # two-step the renderer uses, because a bare landmark pass on a wide frame
    # latches onto furniture.
    w8 = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    seg = torchvision.models.segmentation.deeplabv3_resnet101(weights=w8).eval().cuda()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    batch = w8.transforms()(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
        probs = seg(batch)["out"][0].softmax(0)[15].cpu().numpy()
    del seg
    torch.cuda.empty_cache()
    h, w = img_bgr.shape[:2]
    probs = cv2.resize(probs.astype(np.float32), (w, h))

    ys, xs = np.where(probs > 0.5)
    if len(xs) < 500:
        raise RuntimeError("no person found in the frame")
    px0, px1, py0, py1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    ph, pw = py1 - py0, px1 - px0
    cy, cx = py0 + ph * 0.16, px0 + pw * 0.5
    half = max(pw * 0.42, ph * 0.24, 96.0)
    sx0 = int(np.clip(cx - half, 0, w - 1)); sx1 = int(np.clip(cx + half, sx0 + 32, w))
    sy0 = int(np.clip(cy - half, 0, h - 1)); sy1 = int(np.clip(cy + half, sy0 + 32, h))
    search = rgb[sy0:sy1, sx0:sx1]
    scale = 1.0
    if max(search.shape[:2]) < 512:
        scale = 512.0 / max(search.shape[:2])
        search = cv2.resize(search, (int(search.shape[1] * scale),
                                     int(search.shape[0] * scale)),
                            interpolation=cv2.INTER_CUBIC)

    runner = LandmarkRunner(
        ckpt_path=str(root / "pretrained_weights/liveportrait/landmark.onnx"),
        onnx_provider="cpu", device_id=0)
    runner.warmup()
    lmk = runner.run(search, runner.run(search)).astype(np.float32) / scale
    lmk[:, 0] += sx0; lmk[:, 1] += sy0

    fx0, fx1 = float(lmk[:, 0].min()), float(lmk[:, 0].max())
    fy0, fy1 = float(lmk[:, 1].min()), float(lmk[:, 1].max())
    fw, fh = fx1 - fx0, fy1 - fy0

    # Generous: hair above, neck and shoulder below, ears either side.
    x0 = int(np.clip(fx0 - fw * 0.70, 0, w - 1))
    x1 = int(np.clip(fx1 + fw * 0.70, x0 + 64, w))
    y0 = int(np.clip(fy0 - fh * 0.85, 0, h - 1))
    y1 = int(np.clip(fy1 + fh * 0.70, y0 + 64, h))
    return (x0, y0, x1, y1), (fw, fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--out", default="assets/scene_master.png")
    ap.add_argument("--strength", type=float, default=0.30,
                    help="denoise strength. <0.2 does nothing; >0.45 starts "
                         "inventing a different person")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=5.5)
    ap.add_argument("--work", type=int, default=1024,
                    help="internal resolution the head region is refined at")
    ap.add_argument("--liveportrait-root", default="third_party/LivePortrait")
    ap.add_argument("--compare", default="refine_compare.png")
    args = ap.parse_args()

    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline

    root = Path(args.liveportrait_root).resolve()
    img = cv2.imread(args.source)
    if img is None:
        raise FileNotFoundError(args.source)

    (x0, y0, x1, y1), (fw, fh) = head_region(img, root)
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    print(f"[refine] face {fw:.0f}x{fh:.0f} px, head region {cw}x{ch} px "
          f"-> refining at {args.work}px")

    # Square-ish working canvas at high resolution.
    work = cv2.resize(crop, (args.work, args.work), interpolation=cv2.INTER_CUBIC)
    work_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

    from PIL import Image
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                        torch_dtype=torch.float16)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL, vae=vae, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True)
    free, _ = torch.cuda.mem_get_info()
    if free / 1e9 < 9.0:
        print(f"[refine] {free / 1e9:.1f} GB free - CPU offload")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    out = pipe(
        prompt=FACE_PROMPT, prompt_2=FACE_PROMPT,
        negative_prompt=FACE_NEGATIVE, negative_prompt_2=FACE_NEGATIVE,
        image=Image.fromarray(work_rgb),
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator("cuda").manual_seed(0),
    ).images[0]

    refined = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    refined = cv2.resize(refined, (cw, ch), interpolation=cv2.INTER_AREA)

    # Soft return. The feather is wide relative to the crop so the join lands
    # in hair and shoulder rather than across the cheek, and no rectangle edge
    # survives.
    mask = np.zeros((ch, cw), np.float32)
    inset = int(min(ch, cw) * 0.13)
    cv2.rectangle(mask, (inset, inset), (cw - inset, ch - inset), 1.0, -1)
    k = max(int(min(ch, cw) * 0.16) | 1, 15)
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]

    # Match the refined crop's colour statistics back to the original so the
    # region cannot drift in exposure or white balance - the fastest way for a
    # refined patch to announce itself.
    for c in range(3):
        src, dst = refined[:, :, c].astype(np.float32), crop[:, :, c].astype(np.float32)
        s_mu, s_sd = src.mean(), src.std() + 1e-6
        d_mu, d_sd = dst.mean(), dst.std() + 1e-6
        refined[:, :, c] = np.clip((src - s_mu) * (d_sd / s_sd) + d_mu, 0, 255)

    blended = (refined.astype(np.float32) * mask
               + crop.astype(np.float32) * (1.0 - mask))
    result = img.copy()
    result[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, result)
    print(f"[refine] wrote {args.out}")

    # Before/after on the head region, for judging identity drift.
    side = 460
    a = cv2.resize(crop, (side, side)); b = cv2.resize(result[y0:y1, x0:x1], (side, side))
    cv2.putText(a, "before", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(b, "after", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imwrite(args.compare, np.hstack([a, b]))
    print(f"[refine] comparison -> {args.compare}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
