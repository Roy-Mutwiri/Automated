# Environment design specification

This document is the specification the scene generator is built from, not a
description written after the fact. `tools/generate_scene.py` implements it.

## The architectural decision: one master frame

**The room is not composited behind the presenter. The presenter, the chair,
the desk and the room are generated together as a single photograph, and only
the face is animated.**

This is a deliberate reversal of the previous approach, which matted the person
out and composited them over a procedurally drawn room. That approach cannot
meet this brief, and it is worth being precise about why:

| Requirement | Composite approach | Master frame |
|---|---|---|
| Chair physically contains the person | Needs a hand-built occlusion mask per pose | Free - they were drawn together |
| Contact shadow where body meets chair | Must be faked | Free |
| One coherent light direction | Two lighting models must be reconciled by hand | Free |
| Real walnut, felt, leather, brushed metal | Procedural drawing cannot produce them | Free |
| Background never warps | Guaranteed, background is untouched | Guaranteed |
| Visible compositing edge | A permanent risk at every silhouette | **There is no composite** |

The last row is the important one. The brief says *"do not think: take face
image + paste onto gaming room."* The only way to actually satisfy that is to
never perform that paste.

**What still moves:** LivePortrait regenerates only the face crop and pastes it
back into the master frame. Every pixel outside that crop is byte-identical
across the entire session, which makes background drift, warp and flicker
structurally impossible rather than merely unlikely.

**What this costs:** the face is smaller in a room-scale frame than in a
headshot, so there is less resolution for the animated region. That is the real
trade and it is measured, not hand-waved - see *Face resolution budget* below.

## Three concepts

### Concept A — Executive Streamer
Dark walnut slat wall, charcoal, minimal shelving, warm practicals, essentially
no technology on show. Reads as a serious professional's private office.
*Strength:* highest perceived cost, ages well, no gamer clichés.
*Weakness:* does not read as a **streaming** room. Nothing says he broadcasts.

### Concept B — Creator Man Cave
Same materials, but the streaming apparatus is present and legible: boom mic,
headphone stand, secondary display, subtle PC glow.
*Strength:* unambiguously a streamer.
*Weakness:* drifts toward equipment-advertisement if prop density creeps up.

### Concept C — Modern Trading Streamer
Multiple displays, muted chart glow, finance cues.
*Strength:* on-identity for Trade Fix.
*Weakness:* charts are the fastest route to gibberish AI text, and the room
stops looking good to anyone who does not care about trading.

### Selected: **A + B, with a trace of C**
Weighted roughly **70 / 20 / 10** as directed. Concretely:

- The **wall and materials** come from A. They carry the premium read.
- **One microphone, one headphone stand, one partial display edge** come from
  B. Enough to establish that he streams here; not an equipment showcase.
- From C: **a single distant display with a cool cast**, no legible content.
  The cool accent exists to make the warm room read warmer by contrast, which
  is its real job. No chart shapes are prompted, because prompting charts
  produces gibberish text.

## Camera

| | |
|---|---|
| Output | 1920 × 1080, 16:9 |
| Generation | 1344 × 768 (SDXL 16:9 aspect bucket), upscaled |
| Lens | 35-50mm full-frame equivalent |
| Height | Eye level, a few degrees above |
| Distance | ~1.6-2.0 m - far enough that the room is behind him, not against him |
| Depth of field | Mild. Face sharp, chair mostly sharp, wall softer but **legible** |

Explicitly not: ultra-wide (distorts the face), long lens (collapses the room),
low angle, high angle.

## Subject

- Occupies **45-60 % of frame height** - head, shoulders, upper torso.
- Placed **off-centre**, room filling the negative space.
- Seated upright but relaxed, slight forward attention.
- Nothing growing out of his head: no shelf edge, lamp or slat line behind the
  skull. This is checked on every candidate.

## Depth layers

1. **Foreground** - boom mic arm, desk edge (nearest lens, most defocused)
2. **Subject + chair**
3. **Desk surface and near props**
4. **Slat wall, shelving, practicals**
5. **Room falling away on one side** - the layer that stops it reading as a flat backdrop

## Lighting

| Light | Spec |
|---|---|
| Key | Soft, ~40° off axis, slightly above eye level, warm-neutral |
| Fill | Opposite, clearly weaker; keeps shadow shape |
| Rim | Subtle, warm, separates hair and shoulder from the dark wall |
| Practicals | Hidden-emitter LED under shelves, one small warm lamp |
| Ambient | Overhead **off** - it fights the key and flattens the face |

Face reads about a stop brighter than the room. Practicals must not clip.

## Materials

- **Wall** - vertical dark walnut veneer slats on charcoal felt, visible grain,
  semi-matte
- **Chair** - black/graphite premium ergonomic, fabric or Alcantara-like, fine
  stitching, high back with head support
- **Desk** - dark walnut or matte black, immaculate cable management
- **Metal** - matte black, controlled reflection
- No gloss plastic anywhere

## Palette

Charcoal, graphite, matte black, dark walnut. Warm amber and tungsten for
practicals. **One** muted cool accent, distant and small. No rainbow RGB, no
saturated purple or pink.

## Props — deliberately few

Broadcast microphone on a boom; headphone stand; two or three books; one plant;
one small sculptural object; partial monitor edge. Shelves **40-60 % empty**.

Banned: Funko Pops, energy drinks, game boxes, posters, neon signage, anything
requiring legible text.

## Face resolution budget

The known cost of room-scale framing. At 1344 × 768 with the subject at ~55 %
of frame height, the face spans roughly 130-170 px. LivePortrait crops ~2.3×
face width and resizes that to 256 px for inference, so a 150 px face yields a
~345 px crop - above the 256 px input, meaning no upsampling before inference.
Generating the master frame larger and upscaling improves the *pasted* result
but not the inference input.

**Acceptance threshold: face ≥ 120 px in the generated frame.** Candidates
below it are rejected regardless of how good the room looks.

## Text

Nothing in the scene may require the model to spell. No signage, no labels, no
chart text, no book titles. Branding and monitor UI, if ever wanted, are
overlaid programmatically afterwards - which is also why monitor regions are
worth keeping maskable.

## Framing math, exactly

Asked for explicitly, because "1344x768 -> 1920x1080" hides a crop.

```
generation bucket   1344 x 768    aspect 1.7500   (SDXL's nearest 16:9 bucket)
output              1920 x 1080   aspect 1.7778   (true 16:9)
```

The bucket is very slightly squarer than 16:9, so the two cannot map one to one.
Two options existed and only one is acceptable:

* **Fit** the whole 1344x768 inside 1920x1080 - preserves every source pixel but
  pillarboxes by ~15 px per side, exposing the blurred fill as a visible border
  on an image that must be full-bleed. Rejected.
* **Crop** to the output aspect - loses a few rows, no border. Chosen.

So `framing="full"` takes the largest centred rectangle of *output* aspect that
fits the source:

```
frame_w = min(w, h * out_w/out_h) = min(1344, 768 * 1.7778) = min(1344, 1365) = 1344
frame_h = frame_w * out_h/out_w   = 1344 / 1.7778                             =  756
left    = (1344 - 1344)/2 = 0
top     = ( 768 -  756)/2 = 6
```

**Result: 6 px trimmed from the top and 6 from the bottom, full width kept, then
scaled 1344x756 -> 1920x1080 (a uniform 1.4286x on both axes).**

Verified at runtime: `frame_rect (0, 6, 1344, 756)`, `content_rect (0, 0, 1920,
1080)` - content fills the canvas exactly, so no pillarbox and no letterbox.
Both axes scale by the same factor, so there is no aspect distortion.

## Motion regions, and not painting ourselves into a corner

The brief asks that we not lock permanently into a face-only crop. We have not,
and the reason is structural rather than a promise.

The per-frame write is confined by `mask_out`, which is
`LivePortrait's crop mask x (subject alpha, when compositing)`. Nothing in the
compositor assumes that region is a face - it is simply "the region the renderer
is allowed to write this frame". Widening it later to include shoulders means
producing a larger driven region and a correspondingly larger mask; the
background-lock guarantee is unchanged, because it comes from *"everything
outside the written region is copied from a fixed plate"*, not from the region
being small.

What would need to change for upper-body motion, when we get there:

| Piece | Status |
|---|---|
| Write-region masking | Already general. No change. |
| Background lock | Already general. No change. |
| Driving signal | `AvatarPose` already carries `tx`/`ty`/`scale` and breathing |
| Renderer | Needs a body warp; LivePortrait alone drives the head |

No work is being done for this now, per the brief - this records only that the
door is open.
