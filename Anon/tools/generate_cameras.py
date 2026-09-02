"""Generate one master frame per camera position.

## Why a camera is a whole generated photograph

There is no 3D scene here. LivePortrait warps a face crop out of one image; it
cannot invent the back of a head, a ceiling, or a corner of the room the
photograph never contained. So "switch to camera 6" cannot be a transform of
camera 1 - camera 6 has to be photographed.

This is the same shape as the wardrobe: a camera, like an outfit, is a prepared
source, and switching one is `LivePortraitRenderer.set_source`.

## Which cameras can be alive

`tools/evaluate_scenes.py` rejects a master frame whose head yaw exceeds 10
degrees, because LivePortrait hallucinates past that. So the prompts for the
angled cameras have him **looking into the lens that is live** - his body turns,
his head does not. That is what a real multi-camera presenter does, and it is
also the only way those frames pass the gate. Cameras showing his back, or wide
enough that his face is a few pixels, are stills and are marked as such in
`config/cameras.yaml`.

## Where the room text comes from

Imported from `generate_scene.py` rather than restated. The room, its materials
and the negatives have one definition, and only the composition clause varies
per camera - otherwise seven cameras drift into seven different rooms by
editing accident on top of the drift diffusion gives you anyway.

**This buys stylistic consistency, not geometric consistency.** Independent
samples cannot put the same shelf in the same place. Making the room literally
the same across angles needs depth reprojection from a single master, or a real
3D set.

Usage
-----
    python tools/generate_cameras.py --list
    python tools/generate_cameras.py --camera cam6 --count 4
    python tools/generate_cameras.py --all --count 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from presenter.render.cameras import Camera, CameraRig  # noqa: E402

# One definition of the room, the negatives and the model. Imported, not
# restated - see the module docstring.
from generate_scene import (  # noqa: E402
    CONCEPTS, MODEL, NEGATIVE, NEGATIVE_2, check_length, vram_preflight,
)

WIDTH, HEIGHT = 1344, 768        # SDXL's 16:9 aspect bucket


def texts(rig: CameraRig, cam: Camera, concept: str) -> tuple[str, str, str, str]:
    """Return ``(subject, room, negative, negative_2)`` for one camera."""
    room = cam.room or rig.room or CONCEPTS[concept]
    return (
        cam.subject,
        room,
        cam.negative or NEGATIVE,
        cam.negative_2 or NEGATIVE_2,
    )


def check_all(rig: CameraRig, concept: str) -> list[str]:
    """Fail loudly on any prompt CLIP would silently truncate.

    The same guard `generate_scene.py` carries, applied to the per-camera
    prompts. It cost that file two full generation rounds to learn; there is no
    reason to pay for the lesson twice.
    """
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    problems = []
    for cam in rig.ordered():
        if cam.derive:
            continue          # no prompt: this camera reframes the master
        for label, text in zip(
            ("subject", "room", "negative", "negative_2"),
            texts(rig, cam, concept),
        ):
            try:
                check_length(text, f"{cam.key}/{label}", tok)
            except ValueError as exc:
                problems.append(str(exc).splitlines()[0])
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="config/cameras.yaml")
    ap.add_argument("--concept", default="hybrid", choices=sorted(CONCEPTS))
    ap.add_argument("--camera", action="append",
                    help="camera key, repeatable; default is all missing ones")
    ap.add_argument("--all", action="store_true",
                    help="every camera, including ones already generated")
    ap.add_argument("--list", action="store_true",
                    help="print the rig and what has been generated, then stop")
    ap.add_argument("--count", type=int, default=3,
                    help="candidates per camera; the first is kept as the "
                         "camera, the rest are written alongside for picking")
    ap.add_argument("--seed", type=int, default=1700)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=6.5)
    ap.add_argument("--low-vram", action="store_true",
                    help="force CPU offload regardless of the preflight")
    args = ap.parse_args()

    rig = CameraRig.load(args.config)

    if args.list:
        print(f"[cameras] {len(rig.cameras)} cameras, directory {rig.directory}")
        for cam in rig.ordered():
            state = "generated" if rig.exists(cam.key) else "MISSING"
            kind = "live " if cam.animated else "still"
            print(f"[cameras]   {cam.key:6s} {kind}  {state:9s}  {cam.label}")
            if cam.hint:
                print(f"[cameras]            {cam.hint}")
        return 0

    problems = check_all(rig, args.concept)
    if problems:
        print("[cameras] prompts would be silently truncated by CLIP:")
        for line in problems:
            print(f"[cameras]   {line}")
        return 2

    if args.all:
        keys = list(rig.cameras)
    elif args.camera:
        unknown = [k for k in args.camera if k not in rig.cameras]
        if unknown:
            print(f"[cameras] unknown: {unknown}; have {sorted(rig.cameras)}")
            return 2
        keys = args.camera
    else:
        keys = rig.missing()
    if not keys:
        print("[cameras] nothing to generate; --all to regenerate")
        return 0

    import torch
    from diffusers import StableDiffusionXLPipeline

    normal = vram_preflight() and not args.low_vram
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    if normal:
        pipe = pipe.to("cuda")
    else:
        # Never terminate anyone else's process to make room - see
        # generate_scene.vram_preflight. Offload instead.
        pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    rig.directory.mkdir(parents=True, exist_ok=True)
    candidates_dir = rig.directory / "candidates"
    candidates_dir.mkdir(exist_ok=True)

    for i, key in enumerate(keys):
        cam = rig.cameras[key]
        if cam.derive:
            # Nothing to generate: this camera *is* the master frame at another
            # framing, which is what keeps the man identical across the shots
            # that actually show his face.
            print(f"[cameras] {cam.key} derives from the master ({cam.framing})")
            continue
        subject, room, neg, neg2 = texts(rig, cam, args.concept)
        print(f"[cameras] {cam.key} ({'live' if cam.animated else 'still'}) "
              f"{cam.label}")
        for c in range(args.count):
            seed = args.seed + i * 100 + c
            image = pipe(
                prompt=subject, prompt_2=room,
                negative_prompt=neg, negative_prompt_2=neg2,
                num_inference_steps=args.steps, guidance_scale=args.guidance,
                width=WIDTH, height=HEIGHT,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cand = candidates_dir / f"{cam.key}_{c:02d}_seed{seed}.png"
            cv2.imwrite(str(cand), bgr)
            print(f"[cameras]   candidate {cand.name}")
            if c == 0:
                cv2.imwrite(str(rig.path(cam.key)), bgr)

    print(f"[cameras] wrote to {rig.directory}")
    print("[cameras] candidates are in ./candidates - copy a better one over "
          "the chosen frame if the first is not the best")
    print("[cameras] then gate the live ones: "
          "python tools/evaluate_scenes.py --dir assets/cameras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
