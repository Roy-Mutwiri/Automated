"""Generate wardrobe variants of the presenter portrait by SDXL inpainting.

## Why the outfit has to be baked into the source image

The same reason the headphones and the forward lean are (see
`assets/PROVENANCE.md`): the behaviour engine applies a few degrees of head
delta and nothing else, and LivePortrait regenerates only the *face crop*. The
torso in the output is static pixels lifted straight from the source portrait,
and the head is warped from source appearance features. So a garment or a
headdress cannot be an overlay - it has to be part of the image the whole
pipeline is built from.

The consequence is the design: **one prepared source per outfit**, generated
ahead of time, and the dropdown swaps between prepared sources rather than
compositing anything at runtime. That is also why it looks right - the folds,
the shadows and the way the cloth catches the key light are all real diffusion
output at portrait resolution, not a 2D sticker tracked onto a moving head.

## Why inpainting rather than regenerating

Regenerating from the prompt with `generate_presenter.py --subject "...in a
thobe"` produces *a different person* every time. The wardrobe has to be the
same man in different clothes, so the face, the beard, the skin and the key
light must survive untouched. Inpainting masks the face out of the edit
entirely: only the garment region is denoised, and everything the identity
depends on is copied through.

## Chaining

Clothing and headwear are separate edits applied in sequence (body first, then
head) so the two dropdowns can combine freely without generating a bespoke
prompt for every pair. Each stage re-reads the previous stage's output, so a
ghutra is drawn over the thobe that is already there and picks up its colour
bounce.

## The masks

Derived from the face landmarks rather than hard-coded fractions, so this still
works if the base portrait is replaced.

* **Garment** - everything below the chin, down to a line above the hands.
  Hands stay out of it deliberately: diffusion models are unreliable at hands
  and there is no reason to re-roll a pair that already came out correct.
* **Headwear** - the crown and the sides of the head down to shoulder height,
  *minus* the face. A ghutra is not a hat; it drapes over the ears and onto the
  shoulders, so a mask that stops at the hairline can only ever produce
  something that sits on the head like a cap.

Run `--preview` first. It writes the masks over the portrait and generates
nothing, which costs seconds instead of minutes.

Usage
-----
    python tools/generate_wardrobe.py --preview
    python tools/generate_wardrobe.py --clothing thobe_white --headwear ghutra_white
    python tools/generate_wardrobe.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Shared photographic direction. The garment has to look like it was in the
# original photograph, which means matching the lens, the key light and the
# skin rendering - not just naming the item.
LOOK = (
    "photorealistic, raw photo, natural fabric texture with visible weave, "
    "soft warm key light from the front left, gentle falloff, "
    "50mm lens, shallow depth of field, natural cloth folds and creases, "
    "consistent lighting with the rest of the photograph, high detail"
)

NEGATIVE = (
    "cartoon, illustration, 3d render, cgi, painting, flat colour, "
    "costume, cosplay, fancy dress, theatrical, plastic, shiny latex, "
    "distorted fabric, melted cloth, floating cloth, extra limbs, extra arms, "
    "hands, fingers, deformed hands, text, logo, watermark, signature, "
    "harsh shadows, blown highlights, low quality, blurry"
)

# -- the wardrobe -----------------------------------------------------------
# Terminology is the real terminology. "Arabic headscarf" would prompt a
# tourist-shop approximation; SDXL knows ghutra, shemagh, agal and taqiyah, and
# naming them is most of what separates a plausible result from a costume.

CLOTHING = {
    "tee": None,   # the base portrait, unedited
    "thobe_white": (
        "wearing a crisp white thobe, traditional Gulf Arab kandura, "
        "plain round collar with a narrow placket, long sleeves, "
        "clean pressed white cotton with soft natural folds"
    ),
    "thobe_beige": (
        "wearing a light beige thobe, traditional Gulf Arab kandura, "
        "plain round collar, long sleeves, soft matte cotton fabric"
    ),
    "polo_navy": (
        "wearing a navy blue cotton polo shirt, ribbed collar, "
        "short sleeves, casual smart, soft knit texture"
    ),
    "hoodie_charcoal": (
        "wearing a charcoal grey hoodie, hood down resting behind the neck, "
        "heavy cotton fleece, drawstrings, relaxed fit"
    ),
    "shirt_linen": (
        "wearing an oatmeal linen button-down shirt, collar open at the neck, "
        "top button undone, rolled sleeves, natural linen slub texture"
    ),
}

HEADWEAR = {
    "none": None,  # bare head, keeps the headphones
    "ghutra_white": (
        "wearing a crisp white ghutra headscarf draped over his head and "
        "shoulders, held in place by a black agal cord ring resting on the "
        "crown, traditional Gulf Arab headdress, clean starched cotton with "
        "soft vertical folds falling past the ears onto the shoulders"
    ),
    "shemagh_red": (
        "wearing a red and white checkered shemagh keffiyeh draped over his "
        "head and shoulders, held by a black agal cord ring on the crown, "
        "traditional Arab headdress, woven check pattern following the folds "
        "of the cloth"
    ),
    "ghutra_loose": (
        "wearing a white ghutra headscarf worn loosely without an agal, "
        "one end thrown back over the shoulder, traditional Arab headdress, "
        "soft cotton with natural drape"
    ),
    "taqiyah": (
        "wearing a white embroidered taqiyah kufi skullcap fitted closely to "
        "the crown of his head, fine tonal embroidery, cotton"
    ),
}


def detect_landmarks(img_bgr: np.ndarray, lp_root: Path) -> np.ndarray:
    """Two-pass landmark detection, matching the renderer's approach.

    Uses LivePortrait's own `landmark.onnx` and nothing else - see
    `LivePortraitRenderer._detect_landmarks` for why no third-party detector is
    involved.
    """
    if str(lp_root) not in sys.path:
        sys.path.insert(0, str(lp_root))
    from src.utils.human_landmark_runner import LandmarkRunner

    runner = LandmarkRunner(
        ckpt_path=str(lp_root / "pretrained_weights/liveportrait/landmark.onnx"),
        onnx_provider="cpu",
        device_id=0,
    )
    runner.warmup()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return runner.run(rgb, runner.run(rgb))


class Geometry:
    """Face measurements the masks are built from."""

    def __init__(self, lmk: np.ndarray, shape: tuple[int, int]) -> None:
        self.h, self.w = shape
        self.cx = float((lmk[:, 0].max() + lmk[:, 0].min()) * 0.5)
        self.face_w = float(lmk[:, 0].max() - lmk[:, 0].min())
        self.face_h = float(lmk[:, 1].max() - lmk[:, 1].min())
        self.brow_y = float(lmk[:, 1].min())     # top of the face, not the crown
        self.chin_y = float(lmk[:, 1].max())
        self.eye_y = float(np.mean(lmk[:48, 1]))


def _protect_head(m: np.ndarray, g: Geometry, grow: float = 1.0) -> None:
    """Punch the face back out of a mask.

    Everything the identity is carried by - eyes, nose, mouth, beard - has to
    be copied through untouched rather than re-denoised, so it is removed from
    every mask as the last step.
    """
    cv2.ellipse(
        m,
        (int(g.cx), int(g.eye_y + g.face_h * 0.30 * grow)),
        (int(g.face_w * 0.50 * grow), int(g.face_h * 0.70 * grow)),
        0, 0, 360, 0, -1,
    )


def garment_mask(g: Geometry, hands_at: float = 0.86, feather: int = 41) -> np.ndarray:
    """The body: shoulders, chest and arms, above the hands.

    Not "everything below the chin". The shoulders sit *higher* than the chin,
    so a horizontal cut at the jaw draws its seam straight across the chest and
    leaves the collar - the part of a garment that most says what it is -
    untouched. The mask starts above the shoulder line instead and has the head
    punched back out of it, which puts the boundary along the jaw and neck
    where a real collar line would fall anyway.
    """
    m = np.zeros((g.h, g.w), np.uint8)
    top = int(g.brow_y + g.face_h * 0.50)          # just above the shoulders
    cv2.rectangle(m, (0, top), (g.w, int(g.h * hands_at)), 255, -1)
    # Slightly tighter than the headwear protection: the collar has to be
    # reachable, so the neck is left in the editable region.
    _protect_head(m, g, grow=0.92)
    return cv2.GaussianBlur(m, (feather | 1, feather | 1), 0)


def headwear_mask(g: Geometry, drape: float = 0.9, feather: int = 31) -> np.ndarray:
    """Crown and ears, plus the cloth falling past them, minus the face.

    A ghutra is not a hat. It is a square of cloth over the crown whose sides
    hang past the ears and onto the shoulders, so the mask is a cap *plus two
    side panels* rather than one large blob - a single big ellipse reaches
    across the chest, where no part of the headdress actually goes, and hands
    the model a bib to fill in.

    ``drape`` is how far the side panels fall below the chin, in face heights;
    at 0 the panels vanish and the mask is a skullcap.
    """
    m = np.zeros((g.h, g.w), np.uint8)

    # The crown. The landmark set covers the face only, and the top of the
    # head sits roughly half a face-height above the brow line.
    cv2.ellipse(
        m,
        (int(g.cx), int(g.brow_y + g.face_h * 0.05)),
        (int(g.face_w * 0.92), int(g.face_h * 0.80)),
        0, 0, 360, 255, -1,
    )

    # The two side panels, outboard of the face so they cannot creep across it.
    if drape > 0:
        for side in (-1, 1):
            cv2.ellipse(
                m,
                (int(g.cx + side * g.face_w * 0.68),
                 int(g.chin_y - g.face_h * 0.15)),
                (int(g.face_w * 0.40), int(g.face_h * (0.55 + drape * 0.55))),
                0, 0, 360, 255, -1,
            )

    _protect_head(m, g)
    return cv2.GaussianBlur(m, (feather | 1, feather | 1), 0)


def preview(img: np.ndarray, masks: dict[str, np.ndarray], out: Path) -> None:
    tint = {"garment": (0, 200, 255), "headwear": (255, 120, 0)}
    tiles = []
    for name, m in masks.items():
        over = img.copy().astype(np.float32)
        a = (m.astype(np.float32) / 255.0)[..., None] * 0.55
        over = over * (1 - a) + np.array(tint[name], np.float32) * a
        over = over.astype(np.uint8)
        cv2.putText(over, name, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 3, cv2.LINE_AA)
        tiles.append(cv2.resize(over, (512, 512)))
    cv2.imwrite(str(out), np.hstack(tiles))
    print(f"[wardrobe] mask preview -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", default="assets/presenter_source.png")
    ap.add_argument("--out", default="assets/wardrobe")
    ap.add_argument("--liveportrait-root", default="third_party/LivePortrait")
    ap.add_argument("--clothing", action="append", choices=sorted(CLOTHING))
    ap.add_argument("--headwear", action="append", choices=sorted(HEADWEAR))
    ap.add_argument("--all", action="store_true",
                    help="every clothing x headwear combination")
    ap.add_argument("--preview", action="store_true",
                    help="write the masks over the portrait and stop")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--strength", type=float, default=0.97,
                    help="SDXL base has a 4-channel UNet, so masked denoising "
                         "needs to run nearly to full strength or the original "
                         "garment bleeds through as a ghost")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--force", action="store_true",
                    help="regenerate variants that already exist")
    args = ap.parse_args()

    base_path = ROOT / args.base if not Path(args.base).is_absolute() else Path(args.base)
    img = cv2.imread(str(base_path))
    if img is None:
        return print(f"[wardrobe] cannot read {base_path}") or 2

    lp_root = Path(args.liveportrait_root)
    if not lp_root.is_absolute():
        lp_root = ROOT / lp_root
    lmk = detect_landmarks(img, lp_root)
    g = Geometry(lmk, img.shape[:2])
    print(f"[wardrobe] face {g.face_w:.0f}x{g.face_h:.0f} at ({g.cx:.0f}, "
          f"{g.eye_y:.0f}), chin {g.chin_y:.0f}")

    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.preview:
        preview(img, {"garment": garment_mask(g),
                      "headwear": headwear_mask(g)},
                out_dir / "mask_preview.png")
        return 0

    if args.all:
        clothes, heads = sorted(CLOTHING), sorted(HEADWEAR)
    else:
        clothes = args.clothing or ["tee"]
        heads = args.headwear or ["none"]

    import torch
    from diffusers import AutoPipelineForInpainting

    pipe = AutoPipelineForInpainting.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    from PIL import Image

    def to_pil(bgr):
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def inpaint(bgr, mask, prompt, seed):
        result = pipe(
            prompt=f"{prompt}, {LOOK}",
            negative_prompt=NEGATIVE,
            image=to_pil(bgr),
            mask_image=Image.fromarray(mask),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            strength=args.strength,
            width=1024, height=1024,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    gmask = garment_mask(g)
    made = 0
    for ci, cloth in enumerate(clothes):
        body = img
        if CLOTHING[cloth] is not None:
            body = inpaint(img, gmask, CLOTHING[cloth], args.seed + ci * 17)
        for hi, head in enumerate(heads):
            name = f"{cloth}__{head}.png"
            path = out_dir / name
            if path.exists() and not args.force:
                print(f"[wardrobe] {name} exists, skipping")
                continue
            out = body
            if HEADWEAR[head] is not None:
                # A skullcap sits on the crown; a ghutra falls to the
                # shoulders. Same mask function, different drape.
                drape = 0.0 if head == "taqiyah" else 0.9
                out = inpaint(body, headwear_mask(g, drape=drape),
                              HEADWEAR[head], args.seed + ci * 17 + hi * 3)
            cv2.imwrite(str(path), out)
            made += 1
            print(f"[wardrobe] {name}")

    print(f"[wardrobe] {made} variant(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
