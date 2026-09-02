# Human behaviour architecture

The system that decides what the presenter does. Renderer-agnostic, free of I/O,
and steppable faster than real time — a thirty-minute behavioural analysis costs
about a second.

## The rig we do not have

**This has to be said first, because it determines what is buildable today.**

The current renderer is LivePortrait driving a 2D master frame. It consumes
exactly eight channels — `yaw`, `pitch`, `roll`, `tx`, `ty`, `scale`,
`eye_open_l/r` — plus an expression latent that this session calibrated for
gaze and brows. It regenerates a face crop and pastes it into a static plate.

**There is no skeleton.** No shoulders, no torso, no arms, no hands, no fingers,
no pelvis, no feet. No IK, no joint hierarchy, no collision primitives, no chair
or desk contact.

So the layers the brief specifies divide into two groups:

| Layer | Status |
|---|---|
| 1 Skeletal base pose | **not renderable** — no skeleton |
| 2 Breathing | renderable, via `scale`/`ty` only |
| 3 Posture / weight | partially — reads as head and framing drift |
| 4 Co-speech gesture | **not renderable** |
| 5 Head / neck | renderable |
| 6 Gaze | renderable (as of this session's calibration) |
| 7 Blink | renderable, via a trained retargeting network |
| 8 Facial speech | out of scope for milestone 1 |
| 9 Facial emotion | partially — brows calibrated, no cheek/mouth channels |
| 10 Micro-expression | partially |
| 11 Hand / finger detail | **not renderable** |
| 12 Physics / correctives | **not renderable** |

Six of twelve layers have nowhere to go. Chair contact, desk contact, foot
contact, gesture space, collision, IK release and two-handed gestures are all
specified against a rig that does not exist in this repository.

That is not a reason to stop, and it does change the order of work. Milestone 1
— the silent human — is defined entirely over channels we *do* have: sitting,
breathing, blinking, looking, occasional adjustment, stillness. It is the gate
the brief puts before everything else, and it is achievable now. The body layers
are designed here as interfaces, simulated and testable in the behaviour engine,
and rendered when a rig arrives.

**The decision that unblocks the other six layers is not mine alone**: whether
the 2D master-frame pipeline is replaced by a MetaHuman rig, or the two coexist
with the 2D path as a fallback. See `human_motion_research.md` — MetaHuman is
the only rig option with a commercial licence.

## Components

```
                         (future) CONTENT / LLM
                                   │
                          semantics + emotion
                                   │
                                   ▼
      ┌─────────────────── BehaviorEngine ───────────────────┐
      │                                                       │
      │  StateScheduler ──── semi-Markov, duration-aware      │
      │       │              min/median/max + cooldowns       │
      │       ▼                                               │
      │  Drives  ← arousal (OU, ~26 s) + motion budget        │
      │       │                                               │
      │       ├── AttentionSystem   world-space targets       │
      │       │        │            eye/head division         │
      │       │        ▼                                      │
      │       ├── GazeSystem        microsaccades, drift      │
      │       ├── BlinkSystem       hazard, demand-modulated  │
      │       ├── HeadSystem        idle sway, voluntary moves│
      │       ├── BreathingSystem   involuntary floor         │
      │       ├── PostureSystem     slow comfort shifts       │
      │       ├── ExpressionSystem  brow / micro-expression   │
      │       │                                               │
      │       ▼                                               │
      │  constraints.apply()  ← anatomy wins, always last     │
      │       │                                               │
      │  BehaviorMemory  ← recent actions, n-gram repetition  │
      └───────┼───────────────────────────────────────────────┘
              ▼
          AvatarPose (one canonical pose per timestamp)
              ▼
        renderer → cameras 1-7
```

### BehaviorDirector

Not yet a separate class. Its responsibilities currently live in
`StateScheduler` (what state, for how long) and `AttentionSystem` (what he is
attending to). It becomes its own component when content arrives and the state
stops being self-directed; the seam already exists, because `set_state()` makes
an external caller the owner and the scheduler stands down.

### AttentionSystem — `attention.py`

The component that answers "why did he look there?". Targets are directions in
the **room**, not the screen: LENS, MAIN_DISPLAY, SECOND_DISPLAY, CHAT, DESK,
MIDDLE_DISTANCE.

The camera is not an input to this file. That is what makes the hard
requirement — a camera cut must not change the gaze — hold structurally rather
than by discipline.

Each shift is divided between eyes and head as a function of the target's
eccentricity: below 11°, eyes only; beyond it the head takes a growing share
capped at 0.62, the eyes lead by 20–60 ms, and the eyes counter-roll as the head
arrives so the combined gaze stays on target.

### GazeSystem — `gaze.py`

Keeps everything involuntary: microsaccades (Poisson, ~1.6 Hz, amplitude at
perceptual threshold) and slow drift. When an attention system is supplied it no
longer chooses targets. That split is deliberate — the involuntary floor must
keep running whatever attention is doing, because it is the difference between
eyes that hold a target and eyes that are dead.

### BlinkSystem — `blinking.py`

Log-normal intervals with a hazard that rises with fixation age, modulated by
**visual demand** from the attention target. Reading a display suppresses
blinking; the middle distance releases it. Asymmetric kinematics, frame-rate
accommodation so a fast blink is never a single-frame flash, brow coupling.

### StateScheduler — `scheduler.py`

Semi-Markov. Duration drawn once on entry from a log-normal clamped to
[min, max]; successor chosen only on expiry; re-entry blocked by a cooldown.

### ConstraintSolver — `constraints.py`

The last stage, over everything. Soft saturation at real cervical range,
distinct from the profile's stylistic head limits. When a rig arrives this is
where joint limits, IK, chair/desk contact and self-collision attach; the
call site does not change.

### BehaviorMemory — `scheduler.py`

Recent voluntary actions, recency-weighted suppression, and n-gram repetition
detection over **voluntary events only**.

## Principles, and where each one lives

| Principle | Mechanism |
|---|---|
| Every movement has a reason | `AttentionSystem` — gaze targets are objects, not displacements |
| The commonest valid action is nothing | motion budget + `allow_voluntary()` fire-time gate |
| Stillness is not frozen | involuntary floor: breathing, drift, microsaccades never suppressed |
| The human has memory | `BehaviorMemory` recency decay; `_recent` in attention |
| No single model owns the body | authority table below |
| Physics overrides generated motion | `constraints.apply()` runs last, unconditionally |
| A camera switch never changes state | the camera is not an input to any behaviour module |

## Model authority

Enforced by construction — each channel has exactly one owner, and a system that
does not own a channel cannot write it.

| Channel | Owner | Others |
|---|---|---|
| eyes (gaze direction) | AttentionSystem | gaze adds microsaccades/drift only |
| eyelids | BlinkSystem | expression may couple the brow, never the lid |
| head yaw/pitch | AttentionSystem (share) **+** HeadSystem (idle) | additive, then constrained |
| breathing | BreathingSystem | nothing else touches `scale` |
| brows | ExpressionSystem | blink couples at 0.18 |
| arms / hands | *unowned — no rig* | future co-speech layer, high authority |

When a learned co-speech model arrives it gets high authority over arms and
hands, medium over upper torso, low over head, and **zero over eyes, eyelids and
breathing**. Those three are the ones a viewer reads for life, they are cheap to
compute correctly, and they must keep working when every model is offline.

## Interfaces for the content pipeline

Stable now, so the audio terminal can build against them:

```python
engine.set_state(BehaviorState.SPEAKING)   # caller takes ownership of state
engine.set_profile("PRESENTER_ENERGETIC")
pose = engine.update(dt)                   # one canonical pose per timestamp
events = engine.drain_events()             # for logging and replay
```

Still to add: `push_audio_features(...)`, `set_emotion(valence, arousal)`,
`set_attention(target)`. Deliberately not stubbed — an empty method that silently
does nothing is worse than an absent one.

## Determinism

`BehaviorEngine(seed=N)` reproduces a behaviour schedule exactly. This is what
lets the camera terminal render the same performance from all seven cameras and
get the same human in each.

## Verification

`tests/test_behavior.py`, 36 tests. The ones that matter are statistical, not
functional: interval coefficients of variation, rate ordering across states,
anatomical bounds over five simulated minutes per profile, and absence of
repeated n-grams in voluntary behaviour.

Metrics cannot pass Milestone 1. Only watching five minutes can.
