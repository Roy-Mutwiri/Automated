# Source identity provenance

## `presenter_source.jpg`

| | |
|---|---|
| Depicts | **A synthetic person. No real individual.** |
| File | `SDXL_image_0001002.jpg` (renamed) |
| Resolution | 1024 × 1024 |
| Dataset | [SFHQ-T2I](https://github.com/SelfishGene/SFHQ-T2I-dataset) — Synthetic Faces High Quality, Text2Image (122,726 curated synthetic faces) |
| Dataset licence | **MIT** |
| Generating model | **SDXL** (encoded in the filename prefix) |
| Model output licence | **CreativeML Open RAIL++-M** — permits commercial use |
| Obtained from | HF mirror `bitmind/SyntheticFacesHighQuality-T2I`, `small-sample.zip` |
| Retrieved | 2026-09-02 |

## Why this image specifically

**Licensing.** The SFHQ-T2I dataset is MIT, but that is not the whole story and
the dataset's own claim that synthetic generation means "no license issues" is
an over-simplification. The images come from five different generators with
different downstream terms:

| Generator | Output licence | Commercial use |
|---|---|---|
| **SDXL** | CreativeML Open RAIL++-M | **Yes** |
| Flux1.schnell | Apache-2.0 | Yes |
| Flux1.dev | FLUX.1 [dev] Non-Commercial | **No** |
| Flux1.pro | BFL API terms | Conditional |
| DALL-E 3 | OpenAI terms | Conditional |

The sample obtained contained 133 Flux1.dev, 69 SDXL and 63 Flux1.pro images.
**Only the SDXL subset is unambiguously safe for commercial use**, and since the
commercial question is currently open, the choice was restricted to the SDXL
subset (69 in the tiny sample, 550 in the small sample).
The generating model is recorded in each filename, so this is verifiable rather
than assumed.

**Suitability as a presenter.** Requested identity: an Arab man. Selected from
the licence-safe SDXL subset against the visual requirements:

- Gulf Arab man in white ghutra and black egal.
- Direct camera gaze with genuine eye contact — the presenter read.
- Front-facing, no glasses (specular reflections and occlusion complicate
  eyelid animation; a rejected candidate wore them).
- Natural skin: visible pores and beard detail, no beauty-filter smoothing.
- Warm indoor lighting with a defocused background, which matches the
  generated streaming room rather than fighting it.
- Head-and-shoulders framing at approximately eye level.
- The headdress is a stable, low-detail shape — it mattes cleanly and does not
  flicker the way loose windswept hair would.

**Known caveats for this source:**

- An earlier pick (`SDXL_image_0000227.jpg`) was rejected after seeing it
  animated: its resting expression carried a furrowed brow, so the neutral
  state read as stern rather than open. Resting expression is baked into the
  source - the behaviour engine can only move relative to it - so this is a
  portrait-selection problem, not a tuning one. Worth checking on any
  replacement.
- Head pose in the source is yaw −10.2°, pitch +1.5°. The renderer neutralises 85 %
  of that (`neutralize_pose`) so he addresses the lens; a little residual is
  left deliberately, since a perfectly square head is itself unnatural.

## Disclosure obligation

This is a synthetic identity. If the avatar is presented publicly, the product
must retain a clear indication that it is an AI/virtual presenter. Nothing here
depicts, imitates, or is derived from a real person.

## Replacing this image

Drop a new file in `assets/`, point `avatar.source_image` in
`config/avatar.yaml` at it, and update this file with its provenance. If the
replacement is a real person's likeness, the permission record belongs here
too.
