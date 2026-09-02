"""Generate candidate presenter portraits locally with SDXL.

## Why generate rather than pick from a dataset

The SFHQ-T2I samples are random draws from a prompt-diverse set. They happen to
contain a few Arab men, but not one matching every requirement at once - and
the full dataset is 21.8 GB to search for a single image. Generating costs a
7 GB model download and gives exact control over appearance, expression, gaze,
framing and lighting.

**The licence is unchanged.** Those dataset images were themselves produced by
SDXL; using SDXL directly lands on the same CreativeML Open RAIL++-M terms that
made the SDXL subset the commercially safe choice in the first place. This is
not a licensing shortcut, it is the same licence with better control.

## What the prompt is optimising for

Not "a good photo" - a portrait the *rest of this pipeline* can animate:

* **Front-facing with direct gaze.** Off-axis costs eye contact, and while
  `neutralize_pose` can square the head, driving far from the source pose makes
  LivePortrait hallucinate.
* **Genuinely neutral expression.** Resting expression is baked in; the
  behaviour engine only moves *relative* to it. A previous candidate was
  discarded for a furrowed brow that made every frame read as displeased.
* **No glasses.** Specular reflections and lens occlusion break eyelid
  animation.
* **Real skin texture.** The brief forbids the plastic beauty-filter look, and
  smoothing cannot be added back later.
* **Soft defocused background.** Matches the generated room and mattes cleanly.
* **Head and shoulders with room below the chin**, so the framing has material
  to work with rather than clipping at the torso.

Usage
-----
    python tools/generate_presenter.py --count 8
    python tools/generate_presenter.py --subject "a brown-skinned Arab man in his 30s"
    python tools/generate_presenter.py --count 6 --seed 100 --out candidates/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Framing, lighting and technical qualities. Kept separate from the subject so
# the subject can be changed without losing the parts that make the result
# animatable.
STYLE_STUDIO = (
    "professional studio headshot portrait, head and shoulders, "
    "looking directly into the camera lens, calm neutral relaxed expression, "
    "mouth closed, eyes open and clearly visible, symmetrical front-facing pose, "
    "soft warm key light from the front, gentle fill, "
    "85mm portrait lens, shallow depth of field, softly blurred warm background, "
    "photorealistic, natural detailed skin texture with visible pores and fine lines, "
    "sharp detailed eyes with natural catchlights, high detail, "
    "photograph, raw photo"
)

# A live streamer does not sit like a corporate headshot. The difference is
# posture and context, and both are baked into the source image - the behaviour
# engine only applies a few degrees of head delta, so it cannot lean a body
# forward or put headphones on someone.
#
# What actually reads as "live":
#   * leaning slightly toward the camera - engaged, not posed
#   * relaxed asymmetric shoulders rather than a squared-off portrait pose
#   * over-ear headphones, the single clearest streaming signifier
#   * webcam eye-level framing and a shorter lens, because a streamer sits
#     close to a webcam rather than far from an 85mm portrait lens
#
# The microphone is deliberately NOT prompted here. A boom mic is mounted to
# the desk, not to the presenter, so baking it into the portrait would make it
# swing with his head. It is added as a static foreground element in
# render/environment.py instead.
STYLE_STREAMING = (
    "seated at a desk in a home studio, leaning slightly forward toward the camera, "
    "relaxed engaged posture, shoulders relaxed and slightly asymmetric, "
    "wearing black over-ear headphones, casual clothing, "
    "looking directly into the camera lens, calm friendly neutral expression, "
    "mouth closed, eyes open and clearly visible, "
    "head and shoulders and upper chest visible, webcam at eye level, "
    "soft warm key light from the front, 50mm lens, shallow depth of field, "
    "softly blurred warm room background with bokeh lights, "
    "photorealistic, natural detailed skin texture with visible pores, "
    "sharp detailed eyes with natural catchlights, high detail, "
    "photograph, raw photo, candid livestream still"
)

STYLES = {"studio": STYLE_STUDIO, "streaming": STYLE_STREAMING}

NEGATIVE = (
    "glasses, sunglasses, eyewear, hat brim covering eyes, hands, fingers, "
    "multiple people, two faces, cropped head, top of head cut off, "
    "looking away, profile view, head turned far to the side, "
    "smiling widely, laughing, open mouth, teeth, frowning, angry, furrowed brow, "
    "exaggerated expression, closed eyes, squinting, "
    "plastic skin, airbrushed, beauty filter, oversmoothed skin, waxy, doll-like, "
    "harsh shadows, dramatic lighting, backlit, silhouette, "
    "blurry face, out of focus face, low quality, deformed, distorted, "
    "watermark, text, signature, cartoon, illustration, 3d render, cgi"
)

DEFAULT_SUBJECT = (
    "a brown-skinned Arab man in his mid 30s, warm medium-brown complexion, "
    "short neatly trimmed dark beard, short dark hair, kind intelligent face"
)


def build_prompt(subject: str, style: str = "streaming") -> str:
    return f"{subject}, {STYLES[style]}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--subject", default=DEFAULT_SUBJECT)
    ap.add_argument("--style", default="streaming", choices=sorted(STYLES),
                    help="streaming = leaning forward at a desk with headphones; "
                         "studio = formal squared-off headshot")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=34)
    ap.add_argument("--guidance", type=float, default=6.0,
                    help="lower keeps skin natural; high CFG bakes in the "
                         "over-contrasted plastic look the brief rejects")
    ap.add_argument("--out", default="candidates")
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusionXLPipeline

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(args.subject, args.style)
    print(f"[gen] subject : {args.subject}")
    print(f"[gen] {args.count} candidates, {args.steps} steps, cfg {args.guidance}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    paths = []
    for i in range(args.count):
        seed = args.seed + i
        gen = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            width=1024, height=1024,
            generator=gen,
        ).images[0]
        path = out_dir / f"candidate_{i:02d}_seed{seed}.png"
        image.save(path)
        paths.append(path)
        print(f"[gen] {path.name}")

    # Contact sheet for side-by-side comparison.
    cols = min(4, len(paths))
    cell = 320
    rows = (len(paths) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 25, np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (cell - 6, cell - 6))
        r, c = divmod(i, cols)
        sheet[r * cell + 3:r * cell + cell - 3, c * cell + 3:c * cell + cell - 3] = im
        cv2.putText(sheet, str(i), (c * cell + 10, r * cell + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    sheet_path = out_dir / "contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"[gen] contact sheet -> {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
