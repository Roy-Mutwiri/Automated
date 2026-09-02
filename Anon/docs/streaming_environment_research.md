# What makes a streaming room read as expensive

Research for the environment phase. Companion to `background_research.md`,
which covers the *optics* of a defocused background; this one covers the
**room itself** - what is in it, how it is lit, and why some rooms read as a
$20k studio and others as a bedroom with LEDs.

## The single biggest differentiator: the wall is architecture, not decoration

Every premium setup found treats the back wall as a built element. Every cheap
one treats it as a surface to stick things on.

**Vertical walnut acoustic slats over dark felt** are the current signature of
the high-end look, and the reasons are practical rather than fashionable:

- Walnut is the darkest common slat finish and **hides monitor glare and RGB
  bleed** far better than light oak - which matters enormously for someone
  sitting under constant screen and key-light exposure.
- Real veneer over MDF with felt backing actually absorbs mid frequencies, so
  it is functional, not decorative. Foam pyramids look cheap *because* they
  read as an afterthought stuck to a wall.
- Vertical orientation adds perceived height, which compensates for the 8-foot
  ceilings most home studios have.

Sources: [walnut acoustic panels for home studios](https://akuwoodpanel.com/blogs/articles/best-walnut-acoustic-panels-for-home-studios),
[slat panels for podcast studios](https://akuwoodpanel.com/blogs/articles/best-acoustic-slat-panels-for-podcast-studio-walls),
[acoustic panels for gaming rooms](https://akuwoodpanel.uk/blogs/articles/acoustic-wood-panels-for-gaming-and-esports-rooms).

**Design consequence:** the wall behind our presenter is vertical dark walnut
slats on a charcoal ground, warm-lit. That single decision does more for the
"expensive" read than any prop.

## Lighting: the subject and the room are lit separately

The consistent professional pattern is three-point on the subject plus a
*fourth, separate* treatment for the background.

| Light | Placement | Purpose |
|---|---|---|
| Key | ~45° off camera axis, above eye level, soft | Shape. Not flat frontal. |
| Fill | Opposite side, weaker | Lifts shadow without killing dimension |
| Back / rim | Behind, high | Separates hair and shoulder from a dark room |
| Practicals | In the room itself | Depth, atmosphere, and a *motivation* for the ambience |

The instruction that recurs everywhere: **light the face first, then light the
background.** Exposing both equally is the flat-webcam look.

Also consistent: **kill the overhead room light.** Ceiling downlights fight the
key and produce the raccoon-eye shadows that read as amateur.

Sources: [3-point lighting for streaming](https://www.colborlight.com/blogs/articles/set-up-3-point-lighting-for-streaming),
[live streaming lighting guide 2026](https://www.switcherstudio.com/blog/lighting-setup-for-live-streaming),
[3-point for live streaming](https://onestream.live/blog/how-to-set-up-a-3-point-lighting-system-for-live-streaming/),
[streaming background aesthetics](https://eurekaergonomic.com/blogs/eureka-ergonomic-blog/streaming-background-aesthetic-design-guide).

## Expensive vs. cheap, as a table

The distinction is almost never *cost of objects*. It is restraint, negative
space, and whether light has a visible source.

| Reads expensive | Reads cheap |
|---|---|
| One or two warm practicals, emitter hidden | RGB strip on every edge, emitter visible |
| 40-60% empty shelf | Every shelf full |
| 2-3 accent colours total | Rainbow |
| Materials differ (wood / matte metal / fabric) | Everything the same plastic |
| Deep room, subject well off the wall | Subject 40cm from a flat wall |
| Cable management invisible | Visible spaghetti |
| Objects of varied scale, deliberately placed | Uniform clutter of same-size items |
| Warm tungsten dominant | Cold blue/purple dominant |

Varied object scale is worth calling out: it creates visual rhythm and makes a
space read as curated rather than stocked.

## Composition patterns from real streamer cameras

- Subject occupies roughly **45-60% of frame height** - head, shoulders, upper
  torso. Not a headshot.
- Subject placed **off-centre**, with the room filling the negative space.
- The camera sits **at or just above eye level**, never low.
- **35-50mm full-frame equivalent.** Wider distorts the face; longer collapses
  the room and defeats the point of having one.
- Background is **softer but still legible**. The room is the flex; destroying
  it with f/1.2 bokeh wastes it.
- Chair is visible and matters - it is one of the strongest "this is a real
  seated person" signals available.

## What must be avoided, mechanically

Generated rooms fail in recognisable ways. Watching for these specifically:

- Warped or non-parallel slats; slats that change pitch across the wall
- Shelves that do not meet the wall, or float
- Gibberish text on monitors, spines, and signage
- Impossible chair geometry (extra legs, armrests that pass through the body)
- Objects intersecting each other
- Shadows inconsistent with the visible practicals
- Cables that connect to nothing

## Implications for this project

1. **Walnut slat wall, charcoal ground, warm practicals** is the design.
2. **Subject lighting is warm-neutral key from ~40° with a rim**; room is warmer
   and about a stop under. Our `fit_exposure` already enforces the stop
   relationship - the research validates the number rather than changing it.
3. **Restraint is the whole game.** The failure mode for a generated room is
   over-decoration, not under-decoration.
4. **No text anywhere** that the model has to spell.
