# Architecture

## Pipeline

```
                    ┌─────────────────────────────────────────┐
                    │           BEHAVIOUR ENGINE              │
                    │  (no I/O, no GPU, steppable faster      │
                    │   than real time for analysis)          │
                    │                                         │
   profile ────────▶│  arousal (OU, ~26 s)                    │
   state   ────────▶│  motion budget (leaky, ~3 s half-life)  │
                    │      │                                  │
                    │      ├── blinking     log-normal + burst│
                    │      ├── gaze         fixation/saccade/ │
                    │      │                microsaccade/drift│
                    │      ├── head         min-jerk + OU sway│
                    │      ├── breathing    quasi-periodic    │
                    │      ├── expression   rare, subtle      │
                    │      └── posture      very slow drift   │
                    └────────────────┬────────────────────────┘
                                     │  AvatarPose  (the seam)
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │              RENDERER                   │
                    │  SchematicRenderer  (diagnostic, now)   │
                    │  LivePortraitRenderer (photoreal, next) │
                    └────────────────┬────────────────────────┘
                                     ▼
                       temporal stabilisation ─▶ compose ─▶ display

   later:  TTS audio ─▶ features ─▶ mouth model ─▶ AvatarPose.mouth_* / jaw
```

## The one interface that matters

`AvatarPose` (`types.py`) is the entire contract between behaviour and
rendering. The behaviour engine knows nothing about warping fields or implicit
keypoints; the renderer knows nothing about blink probability.

This split is load-bearing, not tidiness:

- **The rendering decision is reversible.** `docs/avatar_model_research.md`
  compares several viable backends and explicitly declines to commit before
  benchmarking. Swapping backends touches one file.
- **Behaviour is testable without a GPU.** `tools/behavior_timeline.py`
  simulates 30 minutes in about a second. That is what made it possible to find
  and fix a rate bug that would have been nearly invisible by eye.
- **Lip-sync becomes additive.** An audio-driven mouth model writes
  `mouth_open` / `jaw` on the same pose object.

Poses are **absolute, never incremental**. A dropped or late frame therefore
cannot accumulate error — incremental updates drift, and drift on a face reads
as the identity slowly changing.

## Time, not frames

Everything is driven by measured elapsed seconds. The OU processes use exact
discretisation rather than Euler–Maruyama specifically so their variance is
correct under a fluctuating `dt`.

Verified: at 25 / 30 / 60 Hz the same seed produces blink rates of 15.20 /
14.65 / 15.55 per minute and breath rates of 13.45 / 13.50 / 13.75. Behaviour
is frame-rate invariant.

## Stillness: the mechanism

The brief's hardest requirement — knowing when *not* to move — is implemented
as a three-part split:

1. **Arousal** (OU, ~26 s) modulates every rate at once, producing genuinely
   quiet minutes and livelier ones instead of a constant statistical density.
2. **Motion budget** — a leaky accumulator charged by each voluntary movement.
   While charged it gates new discretionary behaviour at *fire time*.
3. **Involuntary floor** — breathing, ocular drift, microsaccades and head sway
   are never gated.

Point 3 is the whole distinction between *still* and *frozen*.

> **Design note.** The budget was initially folded into the sampled interval.
> That was wrong: intervals are always sampled immediately after a movement,
> which is exactly when suppression peaks, so every interval baked in peak
> suppression and stayed stretched — dragging the head-move rate to a third of
> intent (25 s observed median against a 7.5 s base). Transient state must gate
> at fire time; only persistent traits belong in the interval.

## Module map

| Path | Role |
|---|---|
| `types.py` | `AvatarPose`, `BehaviorEvent` — the seam |
| `behavior/curves.py` | min-jerk, easing, asymmetric blink profile |
| `behavior/randomness.py` | seeded RNG, log-normal/gamma intervals, OU process, cooldowns |
| `behavior/state.py` | `BehaviorState`, per-state modulation, the three profiles |
| `behavior/context.py` | `Drives` — per-frame resolved modulation + fire-time gate |
| `behavior/blinking.py` | log-normal intervals, doubles, partials, asymmetry |
| `behavior/gaze.py` | fixation / saccade / microsaccade / drift |
| `behavior/head.py` | min-jerk adjustments over OU sway |
| `behavior/breathing.py` | quasi-periodic, re-drawn per breath |
| `behavior/expression.py` | rare, subtle, no immediate repeats |
| `behavior/posture.py` | slow drift + weight shifts |
| `behavior/engine.py` | arousal, motion budget, composition |
| `render/base.py` | `Renderer` protocol |
| `render/schematic.py` | diagnostic rig preview (**not** the deliverable) |
| `app.py` | real-time clock, debug overlay, failure tolerance |
| `tools/behavior_timeline.py` | long-run audit; exits non-zero on failure |

## Failure handling

A render exception keeps the last good frame and increments a counter shown in
the debug overlay. No black frames, no crash. For a system intended to run
unattended for hours this is the difference between a glitch and a dead stream.

## Future audio interface (not implemented)

```python
avatar.set_state(BehaviorState.SPEAKING)   # implemented
avatar.set_profile("PRESENTER_CALM")       # implemented
avatar.set_gaze(x, y)                      # to add
avatar.set_expression(name, intensity)     # to add
avatar.push_audio(pcm_chunk)               # phase 2
```

States `PRE_SPEECH` / `SPEAKING` / `POST_SPEECH` already exist with correct
rate modulation, so the pipeline can be exercised before audio arrives.
