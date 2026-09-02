"""Phase 1: change the wall material to dark walnut acoustic slats, locally.

Operates on the locked master. Nothing outside `02_wall_mask.png` is allowed to
survive into the output - the person, chair, monitors, microphone and desk come
back byte-identical from the original plate.

## Method: masked img2img, not inpainting

Both were considered, as instructed. Masked img2img wins here for a practical
reason: SDXL's inpainting checkpoint is a separate ~7 GB download, and
inpainting *discards* the masked content and invents replacement from context.
For a wall we do not want invention - we want the same plane, same perspective,
same lighting, different material. img2img at moderate strength starts from the
existing pixels, so the wall's own vanishing lines and light falloff constrain
what comes back.

The compositing is what makes it surgical: the model runs on the whole frame,
but only masked pixels are kept. The rest is copied from the original, so the
approved composition is mathematically incapable of drifting.

## Grain and sharpness matching

A generated region is cleaner than a diffusion-generated photograph's own
texture, and a wall that is sharper or less noisy than the room reads as a
sticker. After compositing, the wall region's noise is matched to the
surrounding plate by measuring high-frequency energy in a ring just outside the
mask and adding matched noise if the region is too clean.

Usage
-----
    python tools/refine_wall.py --strengths 0.25 0.35 0.45
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

WALL_PROMPT = (
    "interior wall of narrow vertical dark walnut wood slats, rigid straight "
    "battens, real walnut veneer wood grain, matte low sheen finish, black "
    "acoustic felt between the slats, consistent even spacing, modern luxury "
    "studio acoustic treatment, architectural, photograph"
)

WALL_NEGATIVE = (
    "curtain, drape, fabric, cloth, velvet, pleated, rustic, cabin, barn, "
    "antique, 1970s panelling, orange wood, red wood, rough timber, "
    "horizontal boards, wavy slats, crooked, bent, melting, distorted, "
    "cartoon, illustration, text, logo"
)


def match_grain(result: np.ndarray, original: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
    """Match the refined region's high-frequency energy to the surrounding plate.

    A diffusion-generated patch comes back cleaner than the photograph it is
    being placed into. Identical colour and identical sharpness are not enough:
    a region with less sensor noise than its surroundings reads as pasted even
    when nothing else is wrong.
    """
    inside = mask > 0.5
    if inside.sum() < 100:
        return result

    ring = (cv2.dilate(inside.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)
    ring &= ~cv2.dilate(inside.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    if ring.sum() < 100:
        return result

    def hf_sigma(img, sel):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hf = g - cv2.GaussianBlur(g, (0, 0), 1.6)
        return float(hf[sel].std())

    target = hf_sigma(original, ring)
    have = hf_sigma(result, inside)
    if have >= target or target <= 0:
        return result

    add = float(np.sqrt(max(target ** 2 - have ** 2, 0.0)))
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, add, result.shape[:2]).astype(np.float32)[..., None]
    out = result.astype(np.float32) + noise * mask[..., None]
    print(f"[wall] grain: plate {target:.2f}, region {have:.2f} -> added {add:.2f}")
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_locked_original.png")
    ap.add_argument("--mask", default="02_wall_mask.png")
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0.25, 0.35, 0.45])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=6.5)
    ap.add_argument("--out-dir", default="wall_variants")
    args = ap.parse_args()

    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline
    from PIL import Image

    img = cv2.imread(args.source)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        raise FileNotFoundError("source or mask missing")
    mask = (mask.astype(np.float32) / 255.0)
    h, w = img.shape[:2]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    free, total = torch.cuda.mem_get_info()
    print(f"[wall] GPU free {free/1e9:.1f} / {total/1e9:.1f} GB")

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                        torch_dtype=torch.float16)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL, vae=vae, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True)
    if free / 1e9 < 9.0:
        print("[wall] LOW_VRAM: CPU offload")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = []
    for st in args.strengths:
        gen = pipe(
            prompt=WALL_PROMPT, prompt_2=WALL_PROMPT,
            negative_prompt=WALL_NEGATIVE, negative_prompt_2=WALL_NEGATIVE,
            image=Image.fromarray(rgb), strength=st,
            num_inference_steps=args.steps, guidance_scale=args.guidance,
            generator=torch.Generator("cuda").manual_seed(7),
        ).images[0]
        gen_bgr = cv2.cvtColor(np.array(gen), cv2.COLOR_RGB2BGR)
        gen_bgr = cv2.resize(gen_bgr, (w, h), interpolation=cv2.INTER_AREA)

        # Keep ONLY masked pixels. Everything else is the original plate, so
        # the approved composition cannot drift by construction.
        comp = (gen_bgr.astype(np.float32) * mask[..., None]
                + img.astype(np.float32) * (1.0 - mask[..., None]))
        comp = np.clip(comp, 0, 255).astype(np.uint8)
        comp = match_grain(comp, img, mask)

        path = out_dir / f"wall_s{st:.2f}.png"
        cv2.imwrite(str(path), comp)
        results.append((st, path, comp))
        print(f"[wall] strength {st:.2f} -> {path.name}")

    # Comparison strip: original first, then each strength.
    tiles = [img] + [c for _, _, c in results]
    labels = ["ORIGINAL"] + [f"s={st:.2f}" for st, _, _ in results]
    cell_w = 900
    cell_h = int(cell_w * h / w)
    shown = []
    for t, lab in zip(tiles, labels):
        r = cv2.resize(t, (cell_w, cell_h))
        cv2.putText(r, lab, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 255), 3)
        shown.append(r)
    rows = [np.hstack(shown[i:i + 2]) for i in range(0, len(shown), 2)]
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(20, 20, 20))
            for r in rows]
    cv2.imwrite("03_wall_before_after.png", np.vstack(rows))
    print("[wall] comparison -> 03_wall_before_after.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
