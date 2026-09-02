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

* **Person first, shot size stated explicitly.** Leading with the room was
  tried and failed outright: all eight candidates came back as empty interior
  renders with no human in them. The person must be the grammatical subject to
  be rendered at all - and the shot size must then be forced with "medium
  shot" / "waist up" / a camera distance, or it collapses to a headshot.
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

# Encoder 1: who he is and how he is framed. MUST stay under 77 CLIP tokens.
# Encoder 1 (CLIP-L): who he is, framed close enough to animate.
#
# "medium close-up" and a 1.2 m camera replace the earlier "medium shot" at
# 2 m. A room-scale shot put the face at 94 px, which is below LivePortrait's
# 256 px inference input once cropped - the room looked superb and the face was
# unusable. Closer framing trades a little visible room for a face that can
# actually be driven.
# BOTH encoders begin with the man. That rule is the hard-won one.
#
# SDXL runs two text encoders (CLIP-L and OpenCLIP-G) and diffusers takes a
# separate prompt for each. Whatever leads a prompt dominates it, and OpenCLIP-G
# dominates the image. Leading encoder B with the wall produced eight of twelve
# candidates containing no human at all; leading encoder A with the man while
# encoder B described only the room produced blank walls behind a good subject.
#
# So: encoder A is WHO + COMPOSITION + CAMERA, encoder B is WHO + ENVIRONMENT +
# MATERIALS, and both open on the same subject clause. The room is context, not
# the topic.
_ANCHOR = "one adult male streamer seated close to camera"

# Encoder A: WHO + composition + camera. The chair is named here as well as in
# encoder B, because it kept disappearing when it lived in only one.
SUBJECT_PROMPT = (
    "livestream camera frame, " + _ANCHOR + ", "
    "a brown-skinned Arab man in his 30s, short dark beard, "
    "in a matte black high-back gaming chair, headrest behind his shoulders, "
    "facing directly at camera, head and upper torso large in frame, "
    "frontal face, looking into the lens, calm neutral, 40mm, natural skin"
)

# Encoder B: same subject, then a MODERN room.
#
# "walnut" alone is dangerous: SDXL reads it as "rustic wooden room" and
# returns a cabin - the last cycle produced an antique den with a bare lamp and
# a vintage camera. Walnut has to be named as a *slat panel accent* against
# charcoal, with matte black and monitors carrying the rest, or the whole room
# turns brown. Roughly charcoal 35 / walnut 25 / black tech 25 / light 15.
ROOM = (
    "photo of " + _ANCHOR + " in a modern luxury streaming room at night, "
    "charcoal grey walls, dark walnut vertical acoustic slat panel accent "
    "behind him, matte black desk, two modern monitors glowing, broadcast "
    "microphone on a black boom arm, hidden warm LED strip lighting, "
    "minimal contemporary, shallow depth of field, photograph"
)

CONCEPTS = {
    # Selected direction: executive materials, streaming apparatus present but
    # restrained, one distant cool accent.
    "hybrid": ROOM,
    # A: no visible streaming gear at all.
    "executive": ROOM.replace("broadcast microphone on a boom arm, ", ""),
    # B: apparatus foregrounded.
    "creator": ROOM.replace("one tungsten lamp, ", "a glass-side PC, "),
}

# Negatives are split across both encoders too. The first version was 218
# tokens, so two thirds of it - every geometry and look term - was discarded
# before the model saw it.
NEGATIVE = (
    "rustic, cabin, log cabin, antique, vintage, traditional study, library, "
    "country house, wooden room, all wood interior, brown room, farmhouse, "
    "bare table lamp, old camera, antique furniture, ornate frames, "
    "music studio, midi keyboard, piano, mixing console, podcast only"
)

NEGATIVE_2 = (
    "wide room shot, tiny person, distant person, empty room, no people, "
    "office chair, mesh chair, executive chair, corporate office, bedroom, "
    "rgb lighting, rainbow, neon, purple, cyberpunk, "
    "text, letters, signage, logo, watermark, screen text, "
    "cgi, 3d render, illustration, plastic skin, deformed hands, blurry face"
)


CLIP_LIMIT = 77


def check_length(text: str, label: str, tokenizer=None) -> int:
    """Fail loudly if a prompt exceeds what CLIP will actually read.

    **This guard exists because its absence cost two full generation rounds.**
    The first prompt here was 390 tokens. CLIP's limit is 77. Diffusers
    truncates silently - no error, no warning in normal use - so 313 tokens
    were discarded and the model simply never saw most of the description.

    That produced two opposite failures that looked like prompt-engineering
    problems and were not: with the room first, the *subject* fell off the end
    and eight empty rooms came back; with the subject first, the *room* fell off
    and eight blank interiors came back. Both times the visible text ended
    mid-sentence.

    A silent truncation is the worst kind of failure - it looks like the model
    disagreeing with you. Hence a hard check rather than a comment.
    """
    if tokenizer is None:
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    n = len(tokenizer(text)["input_ids"])
    if n > CLIP_LIMIT:
        raise ValueError(
            f"{label} is {n} tokens, over CLIP's {CLIP_LIMIT}. "
            f"{n - CLIP_LIMIT} tokens would be silently discarded. "
            f"Shorten it.\n  {text}"
        )
    return n


def build_prompts(concept: str) -> tuple[str, str]:
    """Return ``(prompt, prompt_2)`` - one per SDXL text encoder.

    SDXL has two encoders (CLIP-L and OpenCLIP-G) and diffusers accepts a
    separate prompt for each, giving two independent 77-token budgets whose
    embeddings are concatenated. That is what makes it possible to describe both
    a person and a room without either being truncated away.

    Split by role: who he is and how he is framed in the first, where he is and
    how it is lit in the second.
    """
    return SUBJECT_PROMPT, CONCEPTS[concept]


def vram_preflight(need_gb: float = 9.0) -> bool:
    """Report GPU memory before loading anything. Returns True if NORMAL mode.

    Exists because two batches died mid-run with no traceback. The cause was
    never the generator: LM Studio's llama-server was holding ~13.7 GB of the
    16 GB. A CUDA OOM during decode kills the process silently, so from the
    outside it looks like the tool is broken.

    **This reports and adapts. It never terminates anything.** Other people's
    applications - an LLM server, a 3D editor with unsaved work, a browser -
    are not ours to kill, and a tool that quietly closes them would be far worse
    than one that runs slowly.
    """
    import torch

    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / 1e9, total / 1e9
    used_gb = total_gb - free_gb

    print("[preflight] GPU memory")
    print(f"[preflight]   total {total_gb:5.1f} GB")
    print(f"[preflight]   used  {used_gb:5.1f} GB")
    print(f"[preflight]   free  {free_gb:5.1f} GB   (need ~{need_gb:.0f} GB for NORMAL)")

    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        rows = [r for r in out.splitlines() if r.strip()]
        if rows:
            print(f"[preflight]   {len(rows)} process(es) on the GPU:")
            for r in rows[:8]:
                print(f"[preflight]     {r}")
    except Exception:
        pass

    if free_gb >= need_gb:
        print("[preflight] mode NORMAL")
        return True

    print(f"[preflight] mode LOW_VRAM - only {free_gb:.1f} GB free.")
    print("[preflight] Other applications are holding GPU memory. Nothing will")
    print("[preflight] be terminated; using CPU offload instead (slower, safe).")
    return False


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
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=6.5)
    ap.add_argument("--out", default="scenes")
    ap.add_argument("--liveportrait-root", default="third_party/LivePortrait")
    ap.add_argument("--no-measure", action="store_true")
    ap.add_argument("--low-vram", action="store_true",
                    help="stream modules to the GPU on demand; slower but "
                         "survives a card shared with other applications")
    args = ap.parse_args()

    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    normal_mode = vram_preflight()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt, prompt_2 = build_prompts(args.concept)

    # Enforce the token budget before spending minutes on generation. Silent
    # truncation already cost two rounds; it does not get a third.
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    n1 = check_length(prompt, "subject prompt", tok)
    n2 = check_length(prompt_2, "room prompt", tok)
    n3 = check_length(NEGATIVE, "negative", tok)
    n4 = check_length(NEGATIVE_2, "negative_2", tok)

    print(f"[scene] concept {args.concept}  {WIDTH}x{HEIGHT}  "
          f"{args.count} candidates, {args.steps} steps, cfg {args.guidance}")
    print(f"[scene] tokens  subject {n1}  room {n2}  neg {n3}  neg2 {n4}  "
          f"(limit {CLIP_LIMIT} each)")

    # SDXL's stock VAE produces NaNs in fp16, so diffusers silently upcasts it
    # to float32 for decoding. At 1344x768 that fp32 decode on top of the fp16
    # UNet exceeded 16 GB and killed the first batch after one image with no
    # traceback. Tiling the decode fixed the crash but cost ~5 minutes per
    # image, which is worse than the problem.
    #
    # The fp16-fixed VAE removes the upcast entirely: it decodes natively in
    # fp16, so peak memory drops and no tiling is needed. Correct fix rather
    # than a workaround for a workaround.
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, vae=vae, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True,
    )

    # LOW_VRAM keeps one module on the GPU at a time, so peak usage is bounded
    # by the largest single module rather than the whole pipeline. Costs ~30%
    # throughput; removes the silent-OOM failure entirely. The master frame is
    # generated once, so stability beats speed here.
    if args.low_vram or not normal_mode:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Decode the latents in tiles rather than all at once.
    #
    # Without this the first 1344x768 batch died silently after one image. The
    # cause is in the log: diffusers upcasts the VAE to float32 for decoding,
    # and a full-frame fp32 decode at this resolution on top of the fp16 UNet
    # exceeds 16 GB. The process was killed with no traceback, which is exactly
    # what a CUDA OOM during decode looks like from the outside.
    #
    # Tiling and slicing cost a little decode time and bound the peak, which
    # matters more here: generation is a batch job, and an OOM twelve images in
    # wastes far more time than tiled decoding ever will.
    # Slicing only - tiling is unnecessary now the decode is fp16, and it was
    # the tiling specifically that cost the time.
    # API moved in newer diffusers: it lives on the VAE now, not the pipeline.
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    paths = []
    for i in range(args.count):
        seed = args.seed + i
        image = pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=NEGATIVE,
            negative_prompt_2=NEGATIVE_2,
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
        if args.no_measure:
            face, verdict, cover = -1.0, "n/a", -1.0
        else:
            cover = _person_coverage(cv2.imread(str(p)))
            face, verdict = measure_scene(p, root, cover)
        print(f"  {p.name:<26} person {cover * 100:5.1f}%  face {face:6.0f} px  {verdict}")
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
