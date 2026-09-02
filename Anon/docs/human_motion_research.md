# Human motion: model survey and selection

Research pass for the human-behaviour terminal, September 2026. Written to
decide what to build, not to catalogue the field.

## The finding that decides most of this

**The three strongest co-speech motion systems are licensed for research only,
and so is the dataset almost all of them are trained on.**

| System | Licence | Usable in a branded commercial stream? |
|---|---|---|
| SentiAvatar (2026) | CC BY-NC-SA 4.0 | **No** |
| EMAGE / PantoMatrix | code Apache-2.0, **BEAT2 dataset CC BY-NC-SA** | **No** — the weights are derived from BEAT2 |
| DIDiffGes | AAAI-25 paper, trained on BEAT/BEAT2 | **No**, same reason |
| MECo | SIGGRAPH-25, BEAT2-derived | **No**, same reason |
| LiveGesture (2026) | stated CC-BY 4.0 | **Possibly** — needs the training-data provenance verified |
| MetaHuman toolset | Unreal EULA; free under $1M/yr revenue, usable in any engine since 2025 | **Yes** |

This is not a footnote. "Trade Fix" is a brand and this is a commercial
product. A non-commercial licence on the gesture model contaminates every frame
it touches, and it is the kind of problem that is discovered late and expensively.

The BEAT2 licence propagates: a model whose weights were trained on
CC BY-NC-SA data inherits the restriction regardless of how the *code* is
licensed. EMAGE's Apache-2.0 code does not rescue EMAGE's checkpoints.

**Consequence for the plan:** these systems are legitimate *references* — for
architecture, for motion statistics, for validating our own numbers — and are
not deployable as-is. The route to a commercially clean co-speech layer is our
own motion library from a consenting performer, which the brief already
proposes and which is now the load-bearing part of the strategy rather than a
nice-to-have.

## The systems, and what each is actually for

### SentiAvatar — the architecture worth stealing

*Not the weights. The shape.*

Plan-Then-Infill, concretely: a fine-tuned Qwen-0.5B planner consumes a motion
label plus **sparsely sampled** audio tokens and emits keyframe motion tokens
every 4 frames. A separate 38.5M-parameter Audio-aware Infill Transformer then
fills the 3 intermediate frames from the boundary keyframes plus frame-level
HuBERT features at 20 FPS. Motion is 6D rotation over 63 joints (25 body +
19×2 hands); face is 51 ARKit blendshapes on a **separate pathway that bypasses
the LLM entirely** and runs in parallel from audio alone. Reported 6 s of
output in 0.3 s, ~20× real time, with streaming continuation by prepending the
last two keyframe pairs.

Three things here are directly applicable and cost nothing to adopt:

1. **Meaning picks the gesture; prosody places it.** The planner sees sparse
   audio and semantics; the infiller sees dense audio. That is exactly the
   separation the brief asks for, and it is the reason the output is not a
   reflex of amplitude.
2. **Face and body are separate pathways.** Independently arrived at, this is
   the same conclusion as our "no single model owns the whole body" principle.
3. **Sparse keyframes + dense infill is how you get real-time on a budget.**
   The expensive semantic model runs at 5 Hz; the cheap model runs at 20 Hz;
   the rig interpolates at render rate. Our GPU headroom (see below) makes this
   the only viable shape.

### EMAGE / BEAT2 — the measuring stick

Holistic: face, local body, hands, global translation, on a SMPL-X body with
FLAME head. BEAT2 is the community-standard mesh-level co-speech dataset and
the reason it is worth caring about is that **its statistics are a target we
can measure ourselves against** — gesture frequency, amplitude distributions,
head-motion velocity — without shipping a byte of it.

A caveat the brief already anticipates and which the numbers confirm: BEAT2 is
recorded from speakers performing *for a capture session*. Its motion is
broader and more animated than a calm seated streamer. Any amplitude taken from
it needs normalising down, not up.

### DIDiffGes — the latency proof

Decouples body and hand distributions, uses a GAN to model the marginal
implicitly so diffusion can take large steps: **10 sampling steps instead of
~1000**, a 100× reduction. Relevant less as a candidate than as evidence that
real-time diffusion gesture generation is now a solved shape, so we should not
architect around the assumption that gesture generation must be offline.

### MECo — the personality mechanism

Fine-tunes an LLM to read speech audio *and a motion example together*, placing
the example as an explicit query context in the prompt rather than
pseudo-labelling a style class. Accepts motion clips, static poses, human video,
or text. Supports per-body-part control.

This is the closest published match to what the brief wants from a recorded
performer: not "style token = calm", but "move like *this* clip". If we build
our own motion library, this is the mechanism that turns it into a personality
rather than a lookup table.

### LiveGesture (2026) — the one that might be licensable

Streaming transformer, 0.5 s audio chunks, sub-500 ms per-chunk latency, stated
CC-BY 4.0. **Confidence in the details below is low** — the numbers I could
extract for parameter count and VRAM read as inferred rather than measured, and
the training-data provenance is exactly the thing that determines whether the
CC-BY licence is meaningful. Flagged for a proper read, not adopted.

### MetaHuman — the only commercially clean rig

Since 2025 the toolset is usable with any engine, MetaHumans can be sold, and
it is free under $1M/yr revenue. Audio-driven animation is now in-engine
(UE 5.6). It supplies the things a from-scratch rig would cost months to reach:
Control Rig, facial rig, IK Rig, body correctives, neck correctives, and
expression-driven wrinkle/skin deformation.

One caution from the documentation: audio-driven animation "delivers animation
for all MetaHuman facial controls, including inference of upper face gestures."
That is more than we want. Our blink and gaze systems are better than a
speech-driven inference of them, and the brief is right that the speech solver
should be a **mouth/jaw layer** with the upper face masked out.

## Physiological numbers we are actually going to use

Sourced, not invented. These become the defaults in `config/human_behaviour.yaml`.

| Quantity | Value | Note |
|---|---|---|
| Blink rate, conversation | 10.5–32.5 /min (one study: 32.4 ± 12.4) | Highest of the three regimes |
| Blink rate, primary gaze | 8.0–21.0 /min | Our idle baseline |
| Blink rate, reading | 1.4–14.4 /min (one study: 10.7 ± 9.7) | Reading **suppresses** blinking |
| Saccade duration | ~20 ms (small) to >100 ms (largest) | Main sequence: duration rises with amplitude |
| Fixation duration | 200–600 ms typical | Long right tail |
| Microsaccade rate | ~1–3 /s during sustained fixation | Amplitude 0.01°–0.3° (some report to 2°) |
| Microsaccade duration | 10–100 ms | Must stay sub-threshold visually |

The blink figures matter more than they look. They say the rate is **not a
personality constant** — it is a function of what the eyes are doing, varying
by a factor of three between reading and conversation. A single
`blink_median_interval` cannot represent that, which is why the blink system
needs a hazard modulated by visual demand rather than a scheduled interval.

## What our hardware allows

Measured on this machine, not assumed: the renderer runs at **3.7–5.3 FPS at
1920×1080**, GPU utilisation 88–96%, SM clock falling 1995 → 1867 MHz against a
~79 W ceiling. There is no spare GPU.

That rules out running a neural motion model synchronously in the render loop,
and it is the strongest argument for the SentiAvatar shape: plan sparsely and
cheaply, infill at low rate, interpolate on the rig. It also argues for
generating motion **ahead** of the renderer into a lookahead buffer, and for
keeping the autonomic layer — blink, gaze, breath, posture — entirely
procedural and on the CPU, where it costs microseconds and cannot fail when a
model does.

## Selection

| Layer | Choice | Why |
|---|---|---|
| Autonomic (blink, gaze, breath, posture, micro-motion) | **Ours, procedural, CPU** | Must survive every model being offline. Costs nothing. Already better than a speech-driven inference of the same channels. |
| Facial speech | **MetaHuman audio-driven, mouth/jaw masked** | Commercially clean; upper face stays ours. |
| Co-speech body/hands | **Our motion library + MECo-style example control** | The only commercially clean route. SentiAvatar/EMAGE/DIDiffGes are references. |
| Planner architecture | **Plan-Then-Infill, after SentiAvatar** | Semantics choose the gesture, prosody places it. Fits the GPU budget. |
| Canonical rig | **MetaHuman**, SMPL-X as motion interchange | Only rig with a commercial licence and the deformation quality this needs. |

## What has to be verified before any of this is committed

1. LiveGesture's training-data provenance. A CC-BY paper trained on a
   CC BY-NC-SA dataset is still non-commercial in effect.
2. Whether MetaHuman audio-driven animation can be **masked** to mouth and jaw
   at runtime, or only post-hoc on a baked track.
3. Whether the current renderer is replaced by a MetaHuman pipeline at all, or
   whether the 2D master-frame architecture stands. This is the largest open
   question in the project and it is not mine alone to answer — see
   `human_behaviour_architecture.md`, "The rig we do not have".

## Sources

- [SentiAvatar: Towards Expressive and Interactive Digital Humans](https://arxiv.org/html/2604.02908v1)
- [SentiAvatar open-source announcement](https://www.prnewswire.com/apac/news-releases/sentiavatar-the-first-interactive-3d-digital-human-framework-from-sentipulse-and-gsai-now-open-source-302738047.html)
- [EMAGE: Unified Holistic Co-Speech Gesture Generation](https://arxiv.org/abs/2401.00374)
- [BEAT2 dataset overview](https://www.emergentmind.com/topics/beat2-dataset)
- [DIDiffGes: Decoupled Semi-Implicit Diffusion Models for Real-time Gesture Generation](https://arxiv.org/pdf/2503.17059)
- [MECo: Motion-example-controlled Co-speech Gesture Generation](https://arxiv.org/abs/2507.20220)
- [MECo project page](https://robinwitch.github.io/MECo-Page/)
- [LiveGesture: Streamable Co-Speech Gesture Generation](https://arxiv.org/pdf/2604.10927)
- [MetaHuman 5.6 release notes](https://www.metahuman.com/news/metahuman-leaves-early-access-with-a-feature-packed-new-release)
- [MetaHuman Audio Driven Animation documentation](https://dev.epicgames.com/documentation/en-us/metahuman/audio-driven-animation)
- [MetaHuman licensing change, CG Channel](https://www.cgchannel.com/2025/06/you-can-now-sell-metahumans-or-use-them-in-unity-or-godot/)
- [Three types of spontaneous eyeblink activity: reading, primary gaze, conversation](https://www.researchgate.net/publication/11653609_Consideration_of_Three_Types_of_Spontaneous_Eyeblink_Activity_in_Normal_Humans_during_Reading_and_Video_Display_Terminal_Use_in_Primary_Gaze_and_while_in_Conversation)
- [Blink rate decreases while reading, regardless of task duration or difficulty](https://pubmed.ncbi.nlm.nih.gov/36763349/)
- [The timing of spontaneous eye blinks in text reading suggests a cognitive role](https://www.nature.com/articles/s41598-025-04839-y)
- [Human saccadic eye movements — Scholarpedia](http://www.scholarpedia.org/article/Human_saccadic_eye_movements)
- [The saccade main sequence revised](https://link.springer.com/article/10.3758/s13428-020-01388-2)
