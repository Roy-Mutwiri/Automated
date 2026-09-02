# Source identity provenance

## `presenter_source.png`

| | |
|---|---|
| Depicts | **A synthetic person. No real individual.** |
| Origin | **Generated locally**, not sampled from a dataset |
| Model | `stabilityai/stable-diffusion-xl-base-1.0` (SDXL 1.0) |
| Model / output licence | **CreativeML Open RAIL++-M** — permits commercial use |
| Seed | 65 |
| Resolution | 1024 × 1024 |
| Tool | `tools/generate_presenter.py --style streaming` (prompts live in that file) |
| Generated | 2026-09-02 |

Reproduce with:

```
python tools/generate_presenter.py --count 8 --seed 60 --style streaming
# the chosen frame is candidate_05_seed65
```

## Why generated rather than sampled

Earlier revisions used portraits from the [SFHQ-T2I](https://github.com/SelfishGene/SFHQ-T2I-dataset)
dataset. That worked until the requested identity became specific: across 619
licence-safe sample images, every Arab man present was light-to-medium
complexioned. The full dataset is 21.8 GB to download on the chance it holds a
closer match.

**The licence did not change by generating.** SFHQ-T2I's own images were
produced by SDXL among other models, and the SDXL subset was chosen precisely
because CreativeML Open RAIL++-M permits commercial use. Running SDXL directly
lands on the same terms - this is the same licence with control over the
result, not a shortcut around one.

## What the prompt optimises for

Not photographic quality in the abstract - a portrait the rest of the pipeline
can actually animate. Every requirement below was learned from a portrait that
failed on it:

| Requirement | Why |
|---|---|
| Front-facing, direct gaze | A dataset pick sat at yaw -10 deg, permanently addressing a point off-camera. `neutralize_pose` can correct that, but driving far from the source pose makes LivePortrait hallucinate. This one starts at yaw +2.6 deg. |
| Genuinely neutral expression | A dataset pick had a furrowed brow. Resting expression is baked into the source; the behaviour engine only moves *relative* to it, so no parameter fixes a stern face. |
| Head **and shoulders** with room below | Tightly-cropped portraits clip the torso at the image boundary, which shows as a hard vertical line once the background is replaced. This framing removes the artefact rather than hiding it behind the desk. |
| No glasses | Specular reflections and lens occlusion break eyelid animation. |
| Plain, softly-lit background | Mattes far more cleanly than a busy one, and it is replaced anyway. |
| Real skin texture, low CFG | The brief forbids the plastic beauty-filter look; high guidance bakes it in and it cannot be removed afterwards. |

## Disclosure obligation

This is a synthetic identity. If the avatar is presented publicly, the product
must retain a clear indication that it is an AI/virtual presenter. Nothing here
depicts, imitates, or is derived from a real person.

## Replacing this image

Either regenerate with a different `--subject`, or drop a file in `assets/` and
point `avatar.source_image` in `config/avatar.yaml` at it, then update this
file. If the replacement is a real person's likeness, the permission record
belongs here too.

**Never recolour a face to change apparent ethnicity.** Skin tone is not a
colour shift - it changes subsurface scattering, shadow density and highlight
response, so the result looks wrong. Generate or source a different person.

## Streaming pose

The portrait is generated with `--style streaming` rather than `studio`, because
posture is baked into the source: the behaviour engine applies only a few
degrees of head delta, so it cannot lean a body forward or put headphones on
someone. A squared-off studio headshot reads as a corporate portrait no matter
what the face does afterwards.

The streaming style asks for a forward lean, relaxed asymmetric shoulders,
over-ear headphones, and webcam-height framing on a shorter lens.

**The microphone is deliberately not in the prompt.** A boom mic is clamped to
the desk, not to the presenter - baking one into the portrait would weld it to
his head so it swung whenever he turned. It is a static foreground layer
(`render_mic_foreground`) instead.

Trade-off worth knowing: streaming framing pulls back, which shrinks the face
and costs animation resolution. Candidates were compared on face size as well
as pose for exactly this reason.
