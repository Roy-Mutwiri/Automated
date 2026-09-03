# Expression calibration: what a viewer can actually see

Layer B of the two-layer problem, for the face. The behaviour engine can decide
the presenter is amused; this is the measurement of whether the renderer shows
it, in pixels at 1920x1080, and at what intensity the change crosses the
threshold of being visible at all.

Tool: `tools/expression_salience.py`. Sheet: `expression_salience_sheet.png`.
Raw numbers: `expression_salience.json`.

## Two measurement mistakes, both corrected here

Both are worth recording because both produced confident, precise, wrong
numbers, and neither raised an error.

**1. The median over all 203 landmarks.** The earlier conclusion was that a
smile moves the face 0.3-0.6 px and is therefore perceptually absent. That is
what a smile measures like when you average the whole head: a smile moves the
mouth, and roughly 130 of the 203 landmarks are not on the mouth. Measuring per
region and reporting p90 rather than the mean gives 5.1 px for the same smile -
an order of magnitude apart, and the difference between "the renderer cannot do
expressions" and "the renderer does expressions fine".

**2. Regions defined as fractions of a guessed head box.** The first region
boxes gave the brow band **zero landmarks**, silently, and reported 0.00 px for
every brow-driven expression. The landmark cloud occupies only x 0.12-0.66 of
the crop that was being carved up, so every fraction was wrong. This is the same
failure as the expression-index scan that once ranked latents by measuring the
forehead.

The fix in both cases is the same: anchor the measurement to the landmarks
themselves, never to a hand-picked box. Regions are now bands of the landmark
bounding box, which runs brow-top to chin whatever the crop does.

## Visibility thresholds at 1080p

Measured as p90 landmark displacement within each region, against neutral.

| displacement | verdict |
|---|---|
| under ~1.5 px | invisible; the frame reads as unchanged |
| 1.5 - 3 px | subliminal - present but not legible as an expression |
| over ~3.5 px | clearly visible |

## Measured salience (peak px, after the fixes below)

| expression | 20% | 40% | 60% | 80% | drives |
|---|---|---|---|---|---|
| SMALL_SMILE | 1.0 | 2.4 | 4.1 | 5.9 | mouth corners |
| AMUSED | 1.8 | 4.4 | 7.4 | 10.1 | mouth + cheek + tilt |
| SKEPTICAL | 1.7 | 3.4 | 5.0 | 6.5 | one brow + tilt |
| SURPRISED | 1.6 | 3.6 | 5.7 | 7.7 | brows |
| FOCUSED | 1.0 | 2.2 | 3.4 | 4.4 | brow lower + furrow |

Every expression now clears 3.5 px at the intensity the state machine actually
triggers it with (`STATE_EXPRESSION` in `behavior/engine.py`).

## Two defects this found

**SKEPTICAL never rendered.** Its spec had `brow=0.0` alongside
`brow_split=0.55`. The split is a *proportion of the brow raise*, so with a zero
brow it multiplied nothing, and the single raised eyebrow that defines
scepticism was absent from every frame the system has ever produced. The whole
expression moved the face 0.9 px at full intensity - the mouth alone, below
threshold. With a real brow (`0.34`, split `0.90`) it moves 6.5 px and reads as
scepticism.

**The brow split always favoured the left.** The renderer's two brow channels
are not equally strong (`brow_l` gain 0.012, `brow_r` gain 0.036, from
`render/calibration.py`), so a permanently left-leading split shipped the weaker
channel every time. The split now follows the same coin as the head tilt, so the
same eyebrow does not rise on every sceptical moment and both channels get used.

**FOCUSED was too quiet to see.** Concentration is genuinely a subtle
expression, but at the 0.55 the THINKING state triggers it with, it moved the
face 1.8 px - below threshold. Strengthened (`squint` 0.30 to 0.38, `brow` -0.22
to -0.34, `furrow` 0.30 to 0.46) until both its trigger levels clear 3 px, which
is still far short of a scowl.

## What the measurement does not cover

`head_tilt` is now included, because it is part of an expression's visible
signature and leaving it out under-reported everything that leans on it -
scepticism most of all. Not included: the temporal envelope. These are static
peaks. A 6 px expression that arrives over four frames and a 6 px expression
that arrives over forty are the same number here and completely different on
screen.
