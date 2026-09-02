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

## `wardrobe/` — outfit variants

| | |
|---|---|
| Depicts | **The same synthetic person** as `presenter_source.png`, in different clothes |
| Origin | SDXL **inpainting** of the base portrait, generated locally |
| Model | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` |
| Model / output licence | **CreativeML Open RAIL++-M** — same terms as SDXL base, commercial use permitted |
| Tool | `tools/generate_wardrobe.py` (definitions in `config/wardrobe.yaml`) |
| Naming | `<clothing>__<headwear>.png`; `tee__none` is the base portrait itself |

Reproduce with:

```
python tools/generate_wardrobe.py --preview     # check the masks first
python tools/generate_wardrobe.py --all --sheet
```

### Why inpainting rather than a fresh generation per outfit

Regenerating with `generate_presenter.py --subject "...in a thobe"` produces a
different person every time, and the wardrobe has to be one man changing
clothes. Inpainting masks the face out of the edit entirely, so the identity,
the beard, the skin texture and the key light are copied through byte-for-byte
and only the garment region is denoised. **No variant contains a re-generated
face.**

### Why a second model was downloaded

SDXL base cannot insert a headdress. Its UNet takes 4 channels and was never
trained on mask conditioning, so a masked region is denoised blind: at high
strength it re-imagines the skull into a wound turban, at low strength it only
restyles the hair, and given the drape mask it puts the cloth on the neck as a
scarf. Garments worked throughout — a shirt is a plausible continuation of a
torso — but a ghutra is not a plausible continuation of a scalp. The
inpainting checkpoint's 9-channel UNet is trained for exactly this and does it.

The licence is unchanged: that checkpoint is SDXL 1.0 fine-tuned for
inpainting and ships under the same CreativeML Open RAIL++-M terms, which is
why it was the acceptable choice rather than a better-known alternative.

### Cultural accuracy

The headwear prompts name real garments — **ghutra** and **shemagh** (the
cloth), **agal** (the black cord ring that holds it), **taqiyah**/**kufi** (the
skullcap), **thobe**/**kandura** (the robe). This is not decoration: prompting
"Arabic headscarf" returns a costume-shop approximation, and the negative
prompt has to name *turban*, *bandana* and *hijab* explicitly to stop the model
substituting them. A ghutra is laid over the crown and hangs free past the
ears; it is not wound around the head. Variants that come back wound are wrong
and should be regenerated with a different `--seed`, not accepted.

### Disclosure obligation

Unchanged and applies to every variant: this is a synthetic identity, and the
product must retain a clear indication that it is an AI/virtual presenter.
