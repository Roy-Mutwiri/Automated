"""Generate the wardrobe's variant portraits by SDXL inpainting.

## Why the outfit has to be baked into the source image

The same reason the headphones and the forward lean are (see
`assets/PROVENANCE.md`): the behaviour engine applies a few degrees of head
delta and nothing else, and LivePortrait regenerates only the *face crop*. The
torso in the output is static pixels lifted straight from the source portrait,
and the head is warped from source appearance features. So a garment or a
headdress cannot be an overlay - it has to be part of the image the whole
pipeline is built from.

Hence the design: **one prepared source per outfit**, generated ahead of time,
and the dropdown swaps between prepared sources rather than compositing
anything at runtime. That is also why it looks right - the folds, the shadows
and the way the cloth takes the key light are real diffusion output at portrait
resolution, not a 2D sticker tracked onto a moving head.

## Why inpainting rather than regenerating

Regenerating from the prompt with `generate_presenter.py --subject "...in a
thobe"` produces *a different person* every time. The wardrobe has to be the
same man in different clothes, so the face, the beard, the skin and the key
light must survive untouched. Inpainting removes the face from the edit
entirely: only the masked region is denoised and everything the identity
depends on is copied through.

## Why the dedicated inpainting checkpoint

SDXL base was tried first and cannot do this. Its UNet takes 4 channels and was
never trained on mask conditioning, so a masked region is just a region it is
denoising blind. Measured across three rounds on the headwear:

* at strength 0.97 it re-imagines the skull and returns a **wound turban**;
* at 0.85 it stops generating a headdress at all and merely restyles the hair;
* with the drape mask it drops the cloth onto the **neck, as a scarf**.

The garments came out well throughout - a shirt is a plausible continuation of
a torso, so blind denoising lands on one. A ghutra is not a plausible
continuation of a scalp; putting a new object into a masked region is exactly
what mask conditioning is for, and the 9-channel inpainting UNet is trained on
it. `--model` can be pointed back at the base checkpoint to reproduce the
failure.

## Chaining

Clothing and headwear are separate edits applied in sequence (body, then head)
so the two dropdowns combine freely without a bespoke prompt per pair. The
headwear stage re-reads the clothing stage's output, so a ghutra is drawn over
the thobe that is already there and picks up its bounce light.

## The masks

Derived from face landmarks, not hard-coded fractions, so this survives the
base portrait being replaced.

* **Garment** - the shoulders, chest and arms with the head punched out, down
  to a line above the hands. Hands stay out of it deliberately: diffusion
  models are unreliable at hands and there is no reason to re-roll a pair that
  already came out correct.
* **Headwear** - the crown, plus a panel down each side of the head, *minus*
  the face. A ghutra is not a hat; it hangs past the ears onto the shoulders,
  so a mask stopping at the hairline can only ever produce a cap.

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

from presenter.render.wardrobe import Item, Wardrobe  # noqa: E402

# The 9-channel, mask-conditioned checkpoint. Same CreativeML Open RAIL++-M
# terms as SDXL base, so the licensing story in PROVENANCE.md is unchanged.
MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# CLIP truncates at 77 tokens, silently.
#
# The first version of this file paired a 53-token garment description with a
# 56-token block of photographic direction and lost 31 tokens off the end -
# which is to say, it lost the *entire* photographic direction while appearing
# to work. `check_prompts()` runs before anything is generated so a future edit
# fails loudly instead of quietly dropping the tail.
TOKEN_LIMIT = 77


def check_prompts(w: Wardrobe, model: str) -> list[str]:
    """Report every prompt CLIP would silently truncate."""
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer")
    combos = [
        ("negative", w.negative_for(headwear=False)),
        ("negative+headwear", w.negative_for(headwear=True)),
    ]
    for section, kind in ((w.clothing, "clothing"), (w.headwear, "headwear")):
        for item in section.values():
            if item.edits:
                combos.append((f"{kind}/{item.key}", w.prompt_for(item)))

    over = []
    for name, text in combos:
        n = len(tok(text).input_ids)
        if n > TOKEN_LIMIT:
            over.append(f"{name}: {n} tokens, {n - TOKEN_LIMIT} would be dropped")
    return over


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


def headwear_mask(g: Geometry, drape: float = 0.9, feather: int = 31,
                  wide: float = 1.15) -> np.ndarray:
    """The shape the cloth occupies, minus the face.

    Two shapes, because a skullcap and a ghutra are not the same object.

    ``drape == 0`` is a **cap**: the crown ellipse alone, tight to the skull.

    ``drape > 0`` is a **bell**: the crown flaring outward and downward to the
    shoulders in one contiguous region, with the centre of the chest punched
    back out because a ghutra hangs *beside* the neck rather than over it.

    An earlier version used a crown plus two separate side panels. Compared
    side by side against the bell it left the hairline and the headphone band
    partly outside the mask, and what came back was the model continuing that
    context - braided hair and reconstructed headphone cups down the sides
    instead of cloth. Covering the whole region the headdress owns, in one
    piece, is what stops it.
    """
    m = np.zeros((g.h, g.w), np.uint8)

    # The crown. The landmark set covers the face only, and the top of the head
    # sits roughly half a face-height above the brow line.
    crown_w = g.face_w * (wide if drape > 0 else 0.92)
    cv2.ellipse(
        m,
        (int(g.cx), int(g.brow_y + g.face_h * (0.02 if drape > 0 else 0.05))),
        (int(crown_w), int(g.face_h * (0.95 if drape > 0 else 0.80))),
        0, 0, 360, 255, -1,
    )

    if drape > 0:
        flare = np.array([
            [g.cx - crown_w, g.brow_y + g.face_h * 0.02],
            [g.cx + crown_w, g.brow_y + g.face_h * 0.02],
            [g.cx + g.face_w * (wide + 0.15), g.chin_y + g.face_h * drape],
            [g.cx - g.face_w * (wide + 0.15), g.chin_y + g.face_h * drape],
        ], np.int32)
        cv2.fillPoly(m, [flare], 255)
        cv2.ellipse(
            m,
            (int(g.cx), int(g.chin_y + g.face_h * 0.75)),
            (int(g.face_w * 0.46), int(g.face_h * 0.95)),
            0, 0, 360, 0, -1,
        )

    _protect_head(m, g)
    return cv2.GaussianBlur(m, (feather | 1, feather | 1), 0)


def agal_mask(g: Geometry, feather: int = 21) -> np.ndarray:
    """The band on the crown where the cord ring sits.

    A third pass, over a deliberately tiny region, because the agal never
    appeared when it was one clause inside the ghutra prompt - the model spent
    the whole mask on the cloth and the cord is a small dark detail on top of
    it. The taqiyah is the clue: it came out immediately, and the only thing
    that made it different was a small mask with the whole prompt pointed at
    one object. So the agal gets the same treatment.
    """
    m = np.zeros((g.h, g.w), np.uint8)
    cy = int(g.brow_y - g.face_h * 0.34)
    cv2.ellipse(m, (int(g.cx), cy),
                (int(g.face_w * 0.86), int(g.face_h * 0.30)),
                0, 0, 360, 255, -1)
    # Nothing below the brow: the cord sits on the crown, not on the face.
    cv2.rectangle(m, (0, int(g.brow_y)), (g.w, g.h), 0, -1)
    return cv2.GaussianBlur(m, (feather | 1, feather | 1), 0)


def preview(img: np.ndarray, masks: dict[str, np.ndarray], out: Path) -> None:
    tint = {"garment": (0, 200, 255), "headwear": (255, 120, 0)}
    tiles = []
    for name, m in masks.items():
        over = img.astype(np.float32)
        a = (m.astype(np.float32) / 255.0)[..., None] * 0.55
        over = (over * (1 - a) + np.array(tint[name], np.float32) * a).astype(np.uint8)
        cv2.putText(over, name, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 3, cv2.LINE_AA)
        tiles.append(cv2.resize(over, (512, 512)))
    cv2.imwrite(str(out), np.hstack(tiles))
    print(f"[wardrobe] mask preview -> {out}")


def contact_sheet(paths: list[Path], out: Path, cols: int = 3) -> None:
    cell = 420
    rows = (len(paths) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 24, np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (cell - 6, cell - 6))
        r, c = divmod(i, cols)
        sheet[r * cell + 3:r * cell + cell - 3, c * cell + 3:c * cell + cell - 3] = im
        cv2.putText(sheet, p.stem, (c * cell + 12, r * cell + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), sheet)
    print(f"[wardrobe] contact sheet -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wardrobe", default="config/wardrobe.yaml")
    ap.add_argument("--liveportrait-root", default="third_party/LivePortrait")
    ap.add_argument("--model", default=MODEL,
                    help=f"inpainting checkpoint. Point at {BASE_MODEL} to "
                         f"reproduce the failure described in the module "
                         f"docstring")
    ap.add_argument("--clothing", action="append")
    ap.add_argument("--headwear", action="append")
    ap.add_argument("--all", action="store_true",
                    help="every clothing x headwear combination")
    ap.add_argument("--preview", action="store_true",
                    help="write the masks over the portrait and stop")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--strength", type=float, default=0.99,
                    help="garment denoising strength")
    ap.add_argument("--head-strength", type=float, default=0.99,
                    help="headwear denoising strength, separate because the "
                         "two edits fail in different directions")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--force", action="store_true",
                    help="regenerate variants that already exist")
    ap.add_argument("--sheet", action="store_true",
                    help="also write a contact sheet of everything generated")
    args = ap.parse_args()

    w = Wardrobe.load(args.wardrobe)

    img = cv2.imread(str(w.base))
    if img is None:
        print(f"[wardrobe] cannot read base portrait {w.base}")
        return 2

    lp_root = Path(args.liveportrait_root)
    if not lp_root.is_absolute():
        lp_root = ROOT / lp_root
    lmk = detect_landmarks(img, lp_root)
    g = Geometry(lmk, img.shape[:2])
    print(f"[wardrobe] face {g.face_w:.0f}x{g.face_h:.0f} at ({g.cx:.0f}, "
          f"{g.eye_y:.0f}), chin {g.chin_y:.0f}")

    w.directory.mkdir(parents=True, exist_ok=True)

    if args.preview:
        preview(img, {"garment": garment_mask(g), "headwear": headwear_mask(g)},
                w.directory / "mask_preview.png")
        return 0

    over = check_prompts(w, args.model)
    if over:
        print("[wardrobe] prompts exceed CLIP's 77-token limit and would be "
              "truncated without warning:")
        for line in over:
            print(f"[wardrobe]   {line}")
        return 2

    if args.all:
        clothes, heads = list(w.clothing), list(w.headwear)
    else:
        clothes = args.clothing or [next(iter(w.clothing))]
        heads = args.headwear or [next(iter(w.headwear))]
    for key, table, kind in ((clothes, w.clothing, "clothing"),
                             (heads, w.headwear, "headwear")):
        unknown = [k for k in key if k not in table]
        if unknown:
            print(f"[wardrobe] unknown {kind}: {unknown}; have {sorted(table)}")
            return 2

    import torch
    from diffusers import AutoPipelineForInpainting
    from PIL import Image

    print(f"[wardrobe] loading {args.model}")
    pipe = AutoPipelineForInpainting.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[wardrobe] UNet takes {pipe.unet.config.in_channels} channels "
          f"({'mask-conditioned' if pipe.unet.config.in_channels > 4 else 'NOT mask-conditioned'})")

    def inpaint(bgr, mask, item: Item, seed: int, *, head: bool):
        result = pipe(
            prompt=w.prompt_for(item),
            negative_prompt=w.negative_for(headwear=head),
            image=Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            mask_image=Image.fromarray(mask),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            strength=args.head_strength if head else args.strength,
            width=1024, height=1024,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    gmask = garment_mask(g)
    written: list[Path] = []
    for ci, cloth in enumerate(clothes):
        body = None                      # generated lazily: skipped combos cost nothing
        for hi, head in enumerate(heads):
            path = w.path(cloth, head)
            if path == w.base:
                continue                 # this combination *is* the base portrait
            if path.exists() and not args.force:
                print(f"[wardrobe] {path.name} exists, skipping")
                continue

            if body is None:
                item = w.clothing[cloth]
                body = (inpaint(img, gmask, item, args.seed + ci * 17, head=False)
                        if item.edits else img)

            out = body
            head_item = w.headwear[head]
            if head_item.edits:
                out = inpaint(body, headwear_mask(g, drape=head_item.drape),
                              head_item, args.seed + ci * 17 + hi * 3, head=True)
            cv2.imwrite(str(path), out)
            written.append(path)
            print(f"[wardrobe] {path.name}")

    if args.sheet and written:
        contact_sheet(written, w.directory / "contact_sheet.png")
    print(f"[wardrobe] {len(written)} variant(s) -> {w.directory}")
    print(f"[wardrobe] {len(w.missing())} combination(s) still missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
