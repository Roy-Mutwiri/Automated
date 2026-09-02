"""Generate the master frame: presenter, chair, desk and streaming room in one image.

Implements `docs/environment_design.md`. Read that first - this file is the
mechanism, that file is the reasoning.

## The point

The presenter, the gaming chair, the desk and the room are generated *together*
as a single photograph. The renderer then animates only the face crop and
leaves every other pixel untouched.

This is the opposite of compositing a matted person over a background, and it
is what makes the hard requirements free rather than hand-built:

* the chair physically contains him, because they were drawn together
* the contact shadow where his body meets the seat is real
* there is exactly one light direction in the image, so nothing needs
  reconciling
* walnut, felt, leather and brushed metal are real diffusion output
* the background cannot warp, drift or flicker, because it is never regenerated
* **there is no compositing edge, because there is no composite**

## Why the prompt is shaped the way it is

Every clause is defending against a specific failure:

* **Room described before the person.** Lead with the subject and SDXL returns
  a portrait with a hint of room. Lead with the room and it builds a space and
  puts someone in it - which is the composition actually wanted.
* **No text anywhere.** Nothing in the scene may require the model to spell.
  Signage, book spines, chart labels and screen UI all come back as gibberish.
  Screens are prompted dark and out of focus for this reason alone.
* **Warm tungsten dominant, one small cool accent.** The research is
  unambiguous that cold blue/purple dominance is the cheap-bedroom signature.
* **"40-60% empty shelves", "minimal props".** The failure mode of a generated
  room is over-decoration. Restraint has to be asked for explicitly or the
  model fills every surface.
* **35-50mm.** Wider distorts the face; longer collapses the room and defeats
  the purpose of having one.

Usage
-----
    python tools/generate_scene.py --count 8
    python tools/generate_scene.py --count 4 --seed 200 --concept executive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# SDXL aspect bucket closest to 16:9. Generating at a trained bucket avoids the
# duplicated-subject artefacts that off-bucket sizes produce.
WIDTH, HEIGHT = 1344, 768

SUBJECT = (
    "a brown-skinned Arab man in his mid 30s, warm medium-brown complexion, "
    "short neatly trimmed dark beard, short dark hair, calm confident face, "
    "wearing a fitted dark charcoal crew-neck shirt, "
    "seated upright and relaxed in a premium black ergonomic gaming chair with "
    "a high back and headrest, leaning very slightly toward the camera, "
    "shoulders relaxed, looking directly into the camera lens, "
    "calm friendly neutral expression, mouth closed, eyes open and clearly visible"
)

# The room leads. See module docstring.
ROOM = (
    "interior photograph of a luxury professional streaming studio at night, "
    "back wall of vertical dark walnut acoustic slat panels over charcoal felt, "
    "visible natural wood grain, semi-matte finish, "
    "dark charcoal painted wall on one side, "
    "premium floating shelves mostly empty with only a few objects, "
    "one small plant, two books, one small sculptural object, "
    "hidden warm LED strip lighting under the shelves, "
    "one small warm tungsten lamp, "
    "dark walnut desk in the foreground with immaculate cable management, "
    "a broadcast microphone on a matte black boom arm at the lower edge of frame, "
    "a headphone stand, "
    "the edge of a dark monitor turned away at the side of frame, "
    "spacious room with real depth, the wall several feet behind the man"
)

LIGHT_AND_CAMERA = (
    "soft key light from the upper left at forty degrees, gentle fill from the "
    "right, subtle warm rim light separating his hair and shoulder from the "
    "dark wall, warm tungsten practical lights in the room, "
    "the man is lit brighter than the room behind him, "
    "warm amber and charcoal colour palette, "
    "shot on a full frame mirrorless camera at 40mm, eye level, "
    "mild shallow depth of field, face sharp, background softly defocused but "
    "clearly legible, "
    "photorealistic, natural detailed skin texture with visible pores, "
    "sharp detailed eyes with natural catchlights, "
    "professional colour grading, photograph, raw photo"
)

CONCEPTS = {
    # Selected direction: executive materials, streaming apparatus present but
    # restrained, one distant cool accent.
    "hybrid": ROOM,
    # A: no visible streaming gear at all.
    "executive": ROOM.replace(
        "a broadcast microphone on a matte black boom arm at the lower edge of frame, ",
        "",
    ).replace("a headphone stand, ", ""),
    # B: apparatus foregrounded.
    "creator": ROOM + ", a premium desktop PC with a glass side panel and a "
    "single subtle warm interior light, a small control pad on the desk",
}

NEGATIVE = (
    # Text is the single biggest generated-scene tell.
    "text, writing, letters, words, signage, labels, logo, watermark, "
    "book titles, screen text, user interface text, numbers, charts, graphs, "
    # Cheap-room signature.
    "rgb lighting, rainbow lighting, neon, purple lighting, pink lighting, "
    "saturated blue, cyberpunk, gamer bedroom, led strip visible, "
    "cluttered, messy, funko pop, figurines, posters, energy drink, "
    "foam acoustic pyramids, "
    # Geometry failures.
    "warped walls, curved wall, crooked shelves, floating objects, "
    "distorted furniture, extra limbs, extra arms, deformed hands, "
    "impossible geometry, melting objects, duplicated person, two people, "
    # Camera failures.
    "fisheye, ultra wide angle, wide angle distortion, low angle, "
    "extreme bokeh, blurry face, out of focus face, "
    # Look failures.
    "plastic skin, airbrushed, oversmoothed, waxy, doll-like, "
    "cgi, 3d render, video game screenshot, illustration, cartoon, anime, "
    "overexposed, blown highlights, flat lighting, harsh flash, "
    "cheap, low quality, jpeg artifacts"
)


# Shot-size language, and it is load-bearing.
#
# The first attempt led with the room, on the theory that describing the space
# first would make SDXL build a space and then put someone in it. It does not:
# all eight candidates came back as empty interior-design renders with no human
# in them at all. The room description simply consumed the whole prompt.
#
# The person has to be the grammatical subject to be rendered. But making him
# the subject without qualification produces a headshot, which is the failure
# in the other direction. So he leads *and* the shot size is stated explicitly
# and repeatedly - "medium shot", "waist up", a stated camera distance - which
# is the only reliable lever on framing.
SHOT = (
    "medium shot photograph, waist up, camera two metres away, "
    "the man occupies the middle of the frame with the room visible around him"
)


def build_prompt(concept: str) -> str:
    return f"{SHOT}, {SUBJECT}, {CONCEPTS[concept]}, {LIGHT_AND_CAMERA}"


def _person_coverage(img_bgr: np.ndarray) -> float:
    """Fraction of the frame occupied by a person, via segmentation."""
    import torch
    import torchvision

    weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        weights=weights
    ).eval().cuda()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    batch = weights.transforms()(
        torch.from_numpy(rgb).permute(2, 0, 1)
    ).unsqueeze(0).cuda()
    with torch.no_grad():
        probs = model(batch)["out"][0].softmax(0)[15].cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return float((probs > 0.5).mean())


def measure_scene(path: Path, root: Path, coverage: float) -> tuple[float, str]:
    """Return ``(face_width_px, verdict)`` for a generated scene.

    **This must first establish that a person exists.** An earlier version
    measured the landmark span and nothing else, and reported 600 px faces on
    eight images that contained no human being at all - the landmark model
    happily returns a full set of points for an empty room, and their span is
    meaningless. A gate that passes every candidate is worse than no gate,
    because it is trusted.

    So person segmentation runs first and decides whether there is anyone to
    measure. Only then is the face size meaningful, and it is sanity-checked
    against the frame: a "face" wider than a third of the image is the model
    fitting a face template to furniture, not a detection.
    """
    if coverage < 0.02:
        return 0.0, "NO PERSON"

    sys.path.insert(0, str(root))
    try:
        from src.utils.human_landmark_runner import LandmarkRunner
    except Exception:
        return -1.0, "n/a"

    weights = root / "pretrained_weights/liveportrait/landmark.onnx"
    if not weights.exists():
        return -1.0, "n/a"
    runner = LandmarkRunner(ckpt_path=str(weights), onnx_provider="cpu", device_id=0)
    runner.warmup()

    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        lmk = runner.run(rgb, runner.run(rgb))
    except Exception:
        return 0.0, "no face"

    face = float(lmk[:, 0].max() - lmk[:, 0].min())
    if face > w / 3.0:
        # Implausible for a room-scale frame; the detector has latched onto
        # something that is not a face.
        return face, "SUSPECT"
    if face < 120.0:
        return face, "too small"
    return face, "PASS"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--concept", default="hybrid", choices=sorted(CONCEPTS))
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=6.5)
    ap.add_argument("--out", default="scenes")
    ap.add_argument("--liveportrait-root", default="third_party/LivePortrait")
    ap.add_argument("--no-measure", action="store_true")
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusionXLPipeline

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(args.concept)
    print(f"[scene] concept {args.concept}  {WIDTH}x{HEIGHT}  "
          f"{args.count} candidates, {args.steps} steps, cfg {args.guidance}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    paths = []
    for i in range(args.count):
        seed = args.seed + i
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            width=WIDTH, height=HEIGHT,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        path = out_dir / f"scene_{i:02d}_seed{seed}.png"
        image.save(path)
        paths.append(path)
        print(f"[scene] {path.name}", flush=True)

    del pipe
    torch.cuda.empty_cache()

    # Contact sheet plus the face-size gate.
    root = Path(args.liveportrait_root).resolve()
    cols, cell_w = 4, 420
    cell_h = int(cell_w * HEIGHT / WIDTH)
    rows = (len(paths) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 22, np.uint8)

    print("\n[scene] face width (design gate: >= 120 px)")
    for i, p in enumerate(paths):
        face = -1.0 if args.no_measure else measure_face(p, root)
        verdict = "n/a" if face < 0 else ("PASS" if face >= 120 else "reject")
        print(f"  {p.name:<26} {face:6.0f} px  {verdict}")
        im = cv2.imread(str(p))
        im = cv2.resize(im, (cell_w - 6, cell_h - 6))
        r, c = divmod(i, cols)
        sheet[r * cell_h + 3:r * cell_h + cell_h - 3,
              c * cell_w + 3:c * cell_w + cell_w - 3] = im
        label = f"{i}  {face:.0f}px" if face >= 0 else str(i)
        cv2.putText(sheet, label, (c * cell_w + 12, r * cell_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if face >= 120 or face < 0 else (80, 80, 255), 2)

    sheet_path = out_dir / "contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"\n[scene] contact sheet -> {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
