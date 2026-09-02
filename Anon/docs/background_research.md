# The background: what a real streaming room looks like on camera

Research date: 2026-09-02. Written against
[`src/presenter/render/environment.py`](../src/presenter/render/environment.py),
which already generates a room procedurally. The purpose here is not "ideas for
a background" but **which of the choices in that file are supported by how
these rooms are actually built and shot, and which are guesses** — plus the
specific numbers that turn taste into something checkable.

## Why generic advice needs translating before it applies

Almost every source below is written for a person with a camera, a room and
three lights. We have none of those: we have a fixed portrait with a *fixed*
warm front-left key, and everything behind it is synthesised. Two consequences
run through this whole document:

1. **Advice about the subject's lighting is read backwards.** We cannot move
   the key. Any rule stated as "light the subject like X" becomes "the
   background must be consistent with a key that is already at X".
2. **Advice about palette loses to advice about lighting match.** The most
   repeated aesthetic recommendation — cool blues/greys/whites, avoid warm
   reds and oranges — is real advice for a real room, and it is *wrong here*,
   because our source is warm-lit. Mismatched lighting between subject and
   plate is the loudest tell in any composite and cannot be fixed downstream.
   `environment.py` already resolves this tension the right way; it is worth
   recording that the conflict is deliberate rather than an oversight.

## The numbers

These are the quantities worth holding onto. Everything else in the sources is
decoration.

| Quantity | Value | Source consensus |
|---|---|---|
| Subject → back wall | 3–6 ft (0.9–1.8 m); 2–3 m if you want independent control of background exposure | strong |
| Subject → key background objects (shelf, plant) | 2–4 ft (0.6–1.2 m) | strong |
| Separation between distinct planes | ≥ 1–1.5 m | moderate |
| **Background brightness vs face** | **1–2 stops darker; background light ⅓ to ⅕ of key** | **strong** |
| Back/rim light intensity | 10–25 % of key | strong |
| Key light | 4200–5600 K, 45° off axis, above eye level | strong |
| Background / accent light | 2700–3500 K warm, or RGB | strong |
| RGB hues in frame | 1–2 dominant, static — not cycling | strong |
| Object count in frame | 3–4 (minimalist), grouped in odd numbers (3, 5), varied heights | strong |
| Palette | 1–2 primary colours | strong |
| Eye line | on the upper third line, ~⅓-box headroom, camera at or slightly above eye level | strong |

The **1–2 stop rule is the single most actionable number in this document.**
"If the subject and the background are the same brightness, the image looks flat
regardless of how good your key light is." That is a measurable property of our
output, not an opinion, and we currently do not measure it.

## What is actually in these rooms

Ranked by how often it appears in frame, across the room-setup sources:

1. **An LED strip or two washing a wall or the back of a monitor** — near
   universal, and the cheapest thing that makes a room read as "a stream".
2. **A shelf of objects** — figurines, books, game boxes. Doubles as the depth
   cue; sources specifically recommend placing smaller objects *slightly in
   front* of the shelf to create micro-layers.
3. **Warm string / fairy lights**, usually strung high — the bokeh source.
4. **A neon or LED name sign** — brand identity; the one sharp graphic element
   people accept in an otherwise defocused plate.
5. **Plants** — the standard "this is a home, not an office" signal.
6. **Acoustic panels**, increasingly as a design element (fabric-wrapped,
   felt tiles, slatted wood) rather than grey foam.
7. **Monitor spill** on the lower wall, from the screen the person is facing.

Anti-patterns, stated repeatedly: glossy surfaces that catch hotspots, visible
cables and power bricks, and uniform "wall of stuff" clutter — the sources are
blunt that "busy equals interesting" is a myth.

Nearly all of this dissolves under a portrait lens at f/1.8, which is the
premise `environment.py` is built on, and that premise holds up: at wall
distance there is no fine structure left, only coloured light and silhouette.

## Optics: what defocused light actually does

Two effects matter and only one is currently modelled.

**Disc with a hot rim.** A defocused point light is not a gaussian blob — the
aperture projects a disc with a marginally brighter edge. Already implemented
(`_radial_sprite`), and it is the right call: this is what makes synthetic
bokeh read as glass rather than airbrush.

**Cat's-eye squashing.** Round highlights near the frame edges get clipped by
the lens barrel into lentil/cat's-eye shapes, oriented radially — optical
vignetting, most visible wide open, and *strongest at exactly the aperture we
are claiming*. Perfectly circular discs across the whole frame is a signature of
rendered bokeh, not photographed bokeh. **Not currently modelled.**

Blade shape (polygonal vs round highlights) matters mostly when stopped down;
wide open the blades are retracted, so round discs are correct for f/1.8 and we
should not add polygons.

## Compositing: the tells

- **Light wrap** is the standard fix for the hard cut-out edge: feather a
  little of the background back onto the subject's rim, imitating light
  spilling around the silhouette. Sources treat it as the default first step in
  blending a subject into a plate. **Not currently implemented — this is the
  largest single realism gap.**
- **Grain matched to the plate**, not merely present. We have grain; whether
  its magnitude matches the source portrait's own noise is unverified.
- **Lighting and colour match** — "our brains are incredibly sensitive to
  discrepancies in how light interacts with objects." Already the organising
  principle of the module.

## Motion: should the background move at all?

The ambient-loop literature converges on: hold the camera still, let one even,
directionless motion carry the frame, and make the clip read the same at any
moment so it can play for hours. That is a very tight constraint, and it argues
*against* most of what people actually put in animated stream backgrounds.

For us the safe form is: **modulate intensity, never geometry.** A ±2 % gain on
the LED wash over a ~30 s period, or a slow twinkle on a subset of bokeh discs,
cannot warp or wobble because nothing moves — which is what the brief actually
forbids. Static remains defensible; this is an option, not a defect.

## Gap analysis against `environment.py`

**Already correct and independently supported by the research** — warm-dominant
wall matched to the source key rather than the fashionable cool wash; LED accent
restrained and pushed to the frame edges; two hues, static, not cycling; bokeh
clustered along the upper wall rather than scattered; shelf drawn as a darkening
rather than as black; monitor spill low on the wall; desk as a *foreground*
plane with a weaker blur (a real focus ramp, not one flat backdrop); vignette;
grain; 2× supersample. That is most of the list, and none of it was derived from
these sources — it agrees with them anyway.

**Gaps found, and what was done about them:**

| # | Change | Status |
|---|---|---|
| 1 | **Light wrap** at the subject/plate boundary | Done — `light_wrap()`, applied per frame over the rebuilt edge |
| 2 | **Measure and enforce the 1–2 stop rule** | Done — `key_luminance()` + `fit_exposure()`; currently lands at **−1.51 stops** and is reported in the renderer's info line |
| 3 | **Cat's-eye bokeh** | Done — `_radial_sprite(cats_eye, angle)`, two-circle clip |
| 4 | **Chromatic fringing on bokeh** | Done — per-channel radial zoom on the glow layer |
| 5 | **Third depth plane** | Done — shelf *objects* at 0.78× the wall blur; see the correction below |
| 6 | **Shelf object count** | Done — 5 objects in two groups with varied heights, was ~6 spread evenly |
| 7 | ±2 % LED breathe | **Rejected.** It would mean recomposing the background every frame, and that is precisely the optimisation that took the composite from 44 ms to a fraction of it. Paying a per-frame full-frame multiply for a 2 % brightness wobble nobody is going to consciously see is a bad trade at 13.4 FPS |

Not recommended, and still not done: a synthesised rim light on the subject.
The research is clear that a rim at 10–25 % of key is the standard separation
tool, but our source portrait does not have one, and faking it from the alpha
edge is exactly the kind of thing that reads as a glowing sticker when it fails.

### Two things the research got right and I got wrong first

**The shelf board is not a separate plane.** Reading "create micro-layers" as
"blur the shelf less than the wall" produced a hard rule ruled straight across
the frame — visible immediately in a render, invisible in any of the numbers.
The board is fixed *to* the wall and is at wall distance; it is the objects
standing on it that are nearer the camera. Only they get the weaker kernel.

**"The brightness of the face" is not the median of the face.** A landmark
bounding box on this source is 60 % beard, hair and shadow, so its median reads
72 against lit skin at 140–170. Fitting the plate to that drove the entire room
two thirds of a stop too dark, and the error is worst on exactly the subjects
where separation matters most. `key_luminance` uses the 80th percentile: inside
the lit cheek, below the speculars. With that fixed, the fitted exposure lands
within 3 % of the constants that had been tuned by eye — which is the best
evidence available that both are right.

## Sources

Room and background design —
[StreamScheme](https://www.streamscheme.com/room-background-for-twitch/) ·
[StreamMentor](https://streammentor.com/room-backgrounds-for-twitch/) ·
[Riverside](https://riverside.com/blog/streaming-background-ideas) ·
[Get On Stream](https://getonstream.com/stream-room-background-ideas/) ·
[Eureka: background aesthetics](https://eurekaergonomic.com/blogs/eureka-ergonomic-blog/streaming-background-aesthetic-design-guide) ·
[Eureka: wall decor arrangement](https://eurekaergonomic.com/blogs/eureka-ergonomic-blog/arrange-wall-decor-streaming-backdrop)

Depth and defocus —
[StreamYard: depth of field](https://streamyard.com/blog/ai-depth-of-field-video-background) ·
[StreamYard: bokeh](https://streamyard.com/blog/ai-bokeh-background-for-video) ·
[Fstoppers: depth tricks](https://fstoppers.com/education/depth-tricks-beat-bunch-bokeh-every-time-718236) ·
[Live Streaming Pros](https://livestreamingpros.com/how-to-get-a-blurry-background-for-talking-head-videos-for-live-stream-and-recorded-video/)

Lighting —
[Switcher Studio](https://www.switcherstudio.com/blog/lighting-setup-for-live-streaming) ·
[MACCAM: how to light an interview](https://www.maccam.tv/blogs/lighting-guides/how-to-light-an-interview) ·
[OneStream: 3-point](https://onestream.live/blog/how-to-set-up-a-3-point-lighting-system-for-live-streaming/) ·
[Lume Cube](https://lumecube.com/blogs/news/how-to-get-good-lighting-for-streaming) ·
[StreamHub: key/fill/back](https://streamhub.world/streamer-blog/equipment/925-essential-streaming-lighting-setup-key-light-fill-light-and-backlight-techniques/) ·
[Wikipedia: background light](https://en.wikipedia.org/wiki/Background_light) ·
[GVM: podcast background lighting](https://gvmled.com/podcast-studio-lighting-guide/)

Studio set design —
[The Podcast Consultant](https://thepodcastconsultant.com/blog/podcast-background-ideas) ·
[Fame](https://www.fame.so/post/podcast-studio-design) ·
[Castmagic](https://castmagic.io/post/podcast-set-up-ideas)

Optics —
[Jakub Trávník: On Bokeh](https://jtra.cz/stuff/essays/bokeh/index.html) ·
[Photography Life: vignetting](https://photographylife.com/what-is-vignetting) ·
[Fstoppers: understanding bokeh](https://fstoppers.com/education/what-bokeh-and-what-actually-makes-it-good-or-bad-903056) ·
[35mmc: bokeh terminology](https://www.35mmc.com/13/06/2015/understanding-lens-terminology-bokeh/)

Compositing —
[PremiumBeat: light wrapping](https://www.premiumbeat.com/blog/what-is-light-wrapping-tips-tutorials/) ·
[Foundry/Nuke docs](https://learn.foundry.com/nuke/content/comp_environment/effects/bg_reflections_fg_elements.html) ·
[Filmbaker: compositing hacks](https://www.filmbaker.com/blog/blend-reality-7-vfx-compositing-hacks-for-filmmakers)

Framing and ambient motion —
[Wikipedia: headroom](https://en.wikipedia.org/wiki/Headroom_%28photographic_framing%29) ·
[Westport Studios: rule of thirds](https://www.westportstudiosllc.com/post/the-rule-of-thirds-why-professional-framing-is-the-foundation-of-b2b-video) ·
[Morphic: ambient loops](https://morphic.com/resources/videos/ambient-loop-videos)
