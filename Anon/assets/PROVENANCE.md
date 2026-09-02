# Source identity provenance

## `presenter_source.jpg`

| | |
|---|---|
| Depicts | **A synthetic person. No real individual.** |
| File | `SDXL_image_0012281.jpg` (renamed) |
| Resolution | 1024 × 1024 |
| Dataset | [SFHQ-T2I](https://github.com/SelfishGene/SFHQ-T2I-dataset) — Synthetic Faces High Quality, Text2Image (122,726 curated synthetic faces) |
| Dataset licence | **MIT** |
| Generating model | **SDXL** (encoded in the filename prefix) |
| Model output licence | **CreativeML Open RAIL++-M** — permits commercial use |
| Obtained from | HF mirror `bitmind/SyntheticFacesHighQuality-T2I`, `tiny-sample.zip` |
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
commercial question is currently open, the choice was restricted to those 69.
The generating model is recorded in each filename, so this is verifiable rather
than assumed.

**Suitability as a presenter.** Selected from the SDXL subset against the
visual requirements:

- Direct camera gaze with genuine eye contact — the presenter read.
- Neutral, calm professional expression, matching `PRESENTER_CALM`.
- Natural skin: visible pores and fine lines, no beauty-filter smoothing.
- Clean, well-formed eyes with correct catchlights and no generation artefacts
  (a rejected candidate had unnaturally saturated amber irises).
- Soft out-of-focus background — gives depth separation and is inherently
  temporally stable, since there is no sharp detail to flicker.
- Head-and-shoulders framing at approximately eye level.
- No glasses (specular reflections and occlusion complicate eyelid animation).
- Settled hair rather than windswept, which helps temporal stability.

## Disclosure obligation

This is a synthetic identity. If the avatar is presented publicly, the product
must retain a clear indication that it is an AI/virtual presenter. Nothing here
depicts, imitates, or is derived from a real person.

## Replacing this image

Drop a new file in `assets/`, point `avatar.source_image` in
`config/avatar.yaml` at it, and update this file with its provenance. If the
replacement is a real person's likeness, the permission record belongs here
too.
