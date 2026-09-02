# Surgical finishing: wall, lighting, monitors

The master frame is approved and frozen. Everything in this document is
compositing on top of it: **no diffusion runs on the plate after generation.**
Each phase edits one named defect, under an explicit mask, and proves it left
the rest of the frame byte-identical.

```
master_locked_original.png   the approved generation. Never overwritten.
  -> master_v02_wall.png     phase 1, walnut acoustic slats
  -> master_v03_lighting.png phase 2, two architectural practicals
  -> master_v04_final.png    phase 3, monitor content
```

Each version is a separate file. Nothing is edited in place, so any phase can be
re-run or reverted without regenerating anything.

## Why compositing rather than inpainting

The wall was first attempted with masked SDXL img2img at strengths 0.25 / 0.35 /
0.45. It worked mechanically - the mask held, the composition never drifted -
and produced no walnut at any strength. That was the point at which the problem
was diagnosed correctly: **changing a wall's material is not a generation
problem.** The geometry, the perspective and the light are already in the plate
and are all correct. Only the surface reflectance is wrong, and a surface
reflectance is something you can compute.

Every phase since has been deterministic. The same inputs produce the same
output bit for bit, which also means each result can be measured rather than
judged.

## Phase 1 - the wall

`tools/wall_material.py`. Procedural walnut battens on charcoal felt,
perspective-mapped and lit by the plate's own light.

**One global wall coordinate system.** The material is generated once across the
full frame width and then sampled per fragment, so every visible piece of wall
is a window onto the same surface. This is what makes the battens pass behind
the centre monitor and emerge on the far side in the correct place. Texturing
each fragment independently gives each its own phase, which is the classic tell.

**Relative illumination, not absolute.** The first version transferred the
plate's absolute luminance and the walnut came back a pale peach: it had
inherited the grey wall's ~128 luma. What should carry over is *where* the light
falls, not how bright the old material was. Normalising the illumination field
to a relative falloff (0.67-1.32 around its mean) and applying it to the new
material's own base took the wall to luma 47 while keeping the vignette, the
monitor spill and every cast shadow, so objects stay attached to the wall.

**Scale.** Three pitches were rendered and compared at 100%, 50%, 25% and
360 px mobile width. 22 px was clean everywhere but wider than real product
proportions; 11 px merged into mush below 50%. **16 px** was selected - about
65 mm equivalent, and it holds at every size.

Measured: wall luma 128.3 -> 46.7. Defocus matched at sigma 2.30. Grain measured
at 4.28 against a plate of 3.12, so none was added.

## Phase 2 - architectural practicals

`tools/scene_lighting.py`. Two emitters, each with an origin, a direction, an
inverse-square falloff and a real occlusion test.

| | |
|---|---|
| `left_wall_wash` | point, off-frame at (-70, 250), grazing the walnut left to right |
| `under_shelf_led` | line, under the walnut shelf's front edge, pointing down onto the AV gear |

Shadows are cast by resampling the occluder mask into polar coordinates around
each emitter and attenuating along the ray, so an object darkens what is behind
it and nothing in front of it. Light cannot pass through the chair, the
monitors, the speaker column or the subject.

**Multiplicative, not additive.** This is the decision the phase turns on.
Irradiance scales a surface's radiance by its own reflectance; it does not add a
constant. The first version added light in linear space and the felt gaps went
27.7 -> 49.4 luma, converging on the wood and turning the wall into grey
stripes. Multiplying preserves the batten:felt ratio exactly at any intensity,
and it *reveals* grain rather than drawing it - the absolute gap widens while
the ratio holds, which is what a lamp does to a textured surface.

**A third light was designed and cut.** It was aimed at the walnut between the
centre monitor and the chair and measured a mean delta of 0.00: the chair fills
that gap, so the only surface available to receive it was the chair, which is
off limits. Two motivated sources beat three where one lands on nothing.

Three strengths were rendered. **B** was selected: A is invisible, C begins to
read as designed. Wall luma 48.0 -> 49.8; battens 67.9 -> 70.0; felt 27.7 ->
29.1, so the gaps stay black.

## Phase 3 - monitor content

`tools/monitor_replace.py` and `config/monitor_geometry.json`. Perspective
homography, no diffusion. The centre and right panels are replaced; the left is
powered off and stays that way.

Screen quadrilaterals were located by max-gradient scans along the panel edges,
not by eye - the first estimates sat 4-7 px inside the true LCD and left a rim
of the old picture showing. Corners are allowed to fall outside the frame,
because the quad must describe the whole physical panel or the visible strip
carries the wrong horizontal scale.

The content in `assets/screens/` is drawn programmatically: panels, bars,
blocks, one restrained line, and a TRADE FIX wordmark built from line segments
rather than typeset, so no glyph can drift into gibberish. It is stored
separately from the plate so the application can swap it later without touching
the room.

### Three things that had to be got right

**Occlusion is a matte, not a mask.** Two approaches failed first. A darkness
test over the whole quad classified the *old content's* dark regions as
occluders and punched the previous picture's silhouette through the new
interface. A dilated binary mask then kept a ring of plate several pixels wide
outside the true silhouette - and that plate is the old bright blue screen, so
the frame carried a vivid cyan halo tracing the hairline, worse than the problem
it solved. What works is a known-background matte: the subject's edge pixels are
a linear blend of subject and screen, both endpoints are estimable by normalised
convolution from confident interior pixels, and alpha is the projection onto the
line between them.

**Composite by replacing the background, not by blending over it.**

```
out = plate + visibility * (new - old)
```

A pixel that is 40% screen loses 40% of the old screen's colour and gains 40% of
the new one; a pure-hair pixel is untouched. `old` must be the plate itself
wherever the screen is fully visible - using the smoothed extrapolation
everywhere computes `plate + new - blur(plate)`, which is the new content plus
the old picture's high frequencies, and the previous image ghosts straight
through.

**Exposure needs two anchor points.** A linear gain hit the muted cyan accent as
hard as the charcoal and produced a saturated blue panel. A gamma lift fixed the
accent but raised the UI's charcoal base from 19 to 64 luma, giving the screen a
milky veil. Fitting an affine curve to the plate's black point *and* mean
preserves the UI's internal contrast exactly while placing both correctly, with
a soft knee so the accent rolls off instead of clipping.

The target is 0.75x the plate's LCD luminance, not 1.0x: a charcoal dashboard on
the same monitor genuinely photographs darker than a bright picture does, and
matching exactly would mean building a UI that is not dark.

**Specular preservation is implemented and disabled.** The idea generalises - a
reflection of a room light is achromatic while screen content is not - but
measured against this plate it selected two bright blobs from the old game
content's graffiti and carried them onto the new interface as ghosts. These
panels are matte and nothing bright faces them, so there is nothing genuine to
preserve. The layer is still computed and reported so the decision stays visible
rather than silently absent.

## What is verified, and how

`tools/verify_master.py` renders neutral, blink, gaze and head-move poses
through the real pipeline and derives the dynamic region from the frames
themselves - whatever changed *is* the written region - then compares everything
else.

| | |
|---|---|
| Max diff outside the dynamic region | **0**, all three poses |
| Walnut wall, right monitor, under-shelf light, desk | **0** pixels written |
| Centre monitor LCD surface | blink max 3, gaze max 1, head-move max 43 |

The centre panel sits partly inside the animated head crop. Of the 341 LCD
pixels that change by more than 8 levels during a head move, **all 341** are
within 12 px of the subject's silhouette - the hair edge re-occluding the
screen, which is what should happen. Zero elsewhere.

## Known residuals

* **The headphones still reflect the old screen.** A faint blue highlight on the
  headphone band was cast by the blue panel that is no longer there. It is
  inside the subject and the subject is not to be modified, so it stays. Fixing
  it means relighting the human, which is a different decision.
* **The battens are slightly too regular** at 100%. Real installations have
  millimetre-scale variance. Per-batten tone and grain variation are in;
  positional jitter is deliberately not, because crooked geometry is a worse
  failure than regular geometry in a premium installation.
