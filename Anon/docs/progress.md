# Development log

## [Phase 1] Repository inspection — 2026-09-02

Greenfield. No existing code, models, checkpoints, avatar assets, images or
config anywhere in the project. Nothing to preserve, nothing to avoid breaking.

Environment before any work:

- **No Python installed at all** (Store alias stub only), no nvcc, no ffmpeg,
  no conda, no cmake. Only git and gh.
- GPU: RTX 5080 Laptop 16 GB, **compute capability 12.0 (Blackwell / sm_120)**.
- CPU: Ryzen AI 9 HX 375, 12C/24T, 31.3 GB RAM.

**Assessment:** CURRENT STATE — empty. USEFUL — nothing. BROKEN — nothing.
KEEP — nothing. REPLACE — nothing. MISSING — everything.

Installed uv → Python 3.11.16 → torch 2.11.0+cu128. Verified sm_120 is in
`torch.cuda.get_arch_list()` and benchmarked: 38 TFLOPS fp16 matmul, conv2d
512² in 0.195 ms, 14.7 GB VRAM free.

**sm_120 is the dominant environmental constraint.** Most avatar repos pin
torch 2.0–2.3 on cu118/cu121, which install cleanly and then fail at the first
kernel launch on Blackwell. Pinned requirements must be overridden.

Added `.gitignore` rules for weights/venv **before** downloading anything — the
repo has a file watcher that auto-commits and pushes, and a multi-GB checkpoint
would have been pushed to GitHub.

## [Phase 2] Research — 2026-09-02

Compared LivePortrait, FasterLivePortrait, PersonaLive (CVPR 2026),
EmbodiedHead, StreamAvatar, VASA-3D, SyncTalk++, MuseTalk, LatentSync,
Wav2Lip, SadTalker, AniPortrait, Hallo, EchoMimic. Full table and sources in
`avatar_model_research.md`.

**LivePortrait selected.** The decisive factor was not speed — it is that
LivePortrait is driven by *explicit motion parameters*
(`x_d = s_d·(x_c·R_d + δ_d) + t_d`) rather than by a driving video or audio.
Every other real-time candidate needs one or the other. This project needs a
face that is alive while silent and unprompted for hours, and any finite
driving clip loops. Procedurally synthesising the motion parameters makes
loop-freedom structural.

**Licensing hazard found:** LivePortrait code is MIT, but the InsightFace
models it uses for detection are **non-commercial research only**. MediaPipe
(Apache-2.0) is a documented drop-in replacement and is mandatory before any
commercial deployment. Recorded rather than discovered later.

## [Phase 3–10] Behaviour engine — 2026-09-02

Built the full engine before any renderer, on the reasoning that behaviour is
the hard part and it is testable without a GPU. Subsystems: blinking, gaze,
head, breathing, expression, posture, plus arousal and a motion budget in the
scheduler.

`tools/behavior_timeline.py` simulates 30 minutes in ~1 s and audits rates,
interval variability (CV), stillness gaps and repeated n-grams.

**Two real defects found and fixed by the audit, not by eye:**

1. **Suppression baked into intervals.** The motion budget was folded into the
   sampled interval. Intervals are always sampled right after a movement —
   exactly when suppression peaks — so each one locked in peak suppression for
   its whole duration. Head-move median interval measured 25 s against a 7.5 s
   base; the rate check failed at 1.87/min. Fixed by gating at *fire time*
   instead; only persistent state traits remain in the interval.

2. **Loop detector dominated by blinks.** Blinks were >50 % of all events, so
   `blink>blink>blink>blink` trivially topped every n-gram count and flagged as
   suspicious while telling us nothing. Restricted to voluntary behaviours,
   which is what a viewer actually perceives as a loop.

**One tuning error caught by tightening the test.** The first passing
configuration produced a voluntary movement every 2.4 s (~22/min) — every
measurement inside its published range, and visibly fidgety. The stillness
check was too lenient. Rates were reduced and the check tightened to median gap
≥ 3.0 s, ≥ 50 % of gaps over 3 s, ≥ 20 % over 5 s, a gap ≥ 10 s, and ≤ 17
voluntary moves/min.

A second mistake worth recording: I briefly scaled those thresholds by the
profile's `activity` field. That was wrong — `activity` is one input among
several and does not predict the observed rate, so the scaled thresholds failed
correctly-tuned profiles. Reverted to one uniform bar.

**Current results — all 9 profile × seed combinations pass:**

| Metric | PRESENTER_CALM |
|---|---|
| Blink rate | 14.7–15.5/min |
| Voluntary saccades | 7.0/min |
| Head moves | 2.8/min |
| Microsaccades | 1.37/s |
| Breaths | 13.7/min |
| Blink interval CV | 0.63 |
| Median stillness gap | 3.8 s |
| Longest gap | 22 s |
| Voluntary movement rate | 12.4/min |
| Repeated 4-grams | none above chance |

State modulation validated against the literature: `FOCUSED` → 9.6 blinks/min,
`SPEAKING` → 25.3/min, matching the reported halving under focus and doubling
in conversation.

**Frame-rate invariance verified:** same seed at 25/30/60 Hz gives blink rates
15.20/14.65/15.55 per minute and breath rates 13.45/13.50/13.75.

## [Phase 4-partial] Schematic renderer — 2026-09-02

Built `render/schematic.py`, a wireframe rig preview, plus `app.py` with a real
elapsed-time loop, debug overlay, frame capture and last-good-frame failure
tolerance. Ran 30 s headless: **29.7 FPS, 0 render failures**, frames written
and visually inspected — head pose, gaze and framing vary correctly between
non-consecutive frames.

This is a **diagnostic tool, not the deliverable.** Blink kinematics and
saccade amplitude are far easier to judge against clean geometry than against a
photograph. The photoreal backend consumes the same validated pose stream.

## BLOCKED — source portrait needed

The photoreal renderer cannot proceed without a source identity, and the brief
requires either a synthetic identity created for the project or a real person's
likeness with explicit permission. Neither exists in the repo and this is not a
decision to make unilaterally — it is the face of the product.

## Next

1. Obtain/agree the source portrait (blocking).
2. Install LivePortrait, override its pinned torch for sm_120.
3. Write `render/liveportrait.py` mapping `AvatarPose` → `(R, δ, s, t)`.
4. Benchmark FPS / VRAM / latency; add identity-drift and flicker checks.
5. Long-duration visual run and tuning against the real face.

## [Phase 4] Photoreal renderer — 2026-09-02

Source portrait resolved (see `assets/PROVENANCE.md`): a synthetic face from
SFHQ-T2I, restricted to the **SDXL** subset because that is the only generator
in that dataset whose output licence (CreativeML Open RAIL++-M) unambiguously
permits commercial use. The sample also contained Flux1.dev images, which are
explicitly non-commercial — the generating model is encoded in each filename,
so this was verifiable rather than assumed.

`render/liveportrait.py` implemented. Renders correctly on the first working
run: identity preserved, skin texture intact (pores and fine lines survive),
background bokeh stable, and **blinks are anatomically correct** — proper lid
crease and lash line, because closure goes through LivePortrait's trained
`retarget_eye` network rather than hand-nudged keypoints.

**Removed every non-commercial component from the runtime.** LivePortrait's
stock cropper needs InsightFace (non-commercial models). MediaPipe was the
planned substitute but its current release has dropped the `solutions` API.
Better solution found: LivePortrait's own `landmark.onnx` bootstraps itself in
two passes — coarse pass on the whole image, then refine on the resulting crop.
Detection runs once at startup so the second pass is free, and the pipeline now
depends on neither InsightFace nor MediaPipe.

### Performance: below target, root cause identified

| Stage | Time |
|---|---|
| keypoints + stitching | 2.8 ms |
| **warp_decode** | **66 ms** |
| parse_output | 3.3 ms |
| compositing (original) | 44 ms |
| **total (original)** | **116 ms → 8.4 FPS** |
| total (after compositing fix) | 101 ms → 9.9 FPS |

**Target is 25 FPS minimum. This does not meet it.**

*Compositing fix.* The naive path pasted the crop into the full 1024×1024
source, then letterboxed that into 1280×720 — 44 ms of recomputing a composite
whose background never changes, then discarding most of the pixels as bars.
Replaced with one precomputed affine from crop space directly to output space,
a static background composed once, and a per-frame blend. This also fixed the
16:9 composition requirement, which the letterbox had been violating.

*The real bottleneck is not compute.* Sampling the GPU under load:
**14–18 % utilisation, 1065–1297 MHz, 30 W.** The GPU is idle most of each
frame. This is a **launch-bound** workload — many small kernels the CPU cannot
dispatch fast enough — not a compute limit. That reframes the fix entirely:

- fp32 → true fp16 weights: 68.9 → 65.0 ms. Negligible, as expected for a
  launch-bound workload.
- `cudnn.benchmark`: negligible for the same reason.
- **CUDA graphs / `torch.compile`** attack the actual problem. `triton` is not
  installed by default on Windows; `triton-windows` provides it.

Attempting to hit 25 FPS by reducing quality would be the wrong trade — the
brief explicitly ranks a stable realistic 30 FPS above an unstable 60, and
16 % utilisation means there is a large amount of performance being left on the
table for free.

### Known visual defects

1. **Framing is a tight close-up, not head-and-shoulders.** Cropping 16:9 from
   a 1024×1024 square source cannot retain shoulders. This is a source-image
   limitation, not a code defect — fix by using a source portrait framed wider
   or in landscape.
2. **Gaze and brow mapping are uncalibrated.** LivePortrait has no native gaze
   control and its `exp` dimensions are undocumented. Current indices are a
   hypothesis borrowed from community tooling, marked `verified=False` in
   `render/calibration.py` with deliberately tiny gains so a wrong guess is
   ineffective rather than face-distorting.

## [Correction] GPU measurement was wrong — 2026-09-02

The performance section above originally reported **14-18 % GPU utilisation at
30 W** and concluded the workload was almost entirely launch-bound.

**That measurement was invalid.** It sampled 12 s after launching a process
whose first frame alone takes 31 s (cudnn autotune of the 3D convolution
shapes). It measured the autotune phase — CPU-side algorithm search — not
rendering.

Re-measured in steady state with the UI running: **58-73 % utilisation,
1755-2002 MHz, 76-78 W, 67-70 °C.**

Consequences:

- The GPU is working hard, not idling. Headroom is ~30-40 %, not ~85 %.
- The realistic win from TensorRT / CUDA graphs is **1.5-2x, not 5x**.
  Reaching 25 FPS from 13.4 needs ~1.9x — the optimistic end of that band, and
  not guaranteed.
- CPU-side compositing (~20 ms of the 74.7 ms budget) is proportionally more
  significant than the earlier framing implied, and moving it to GPU rises up
  the priority list.

Lesson worth keeping: never sample GPU counters without first confirming the
process has reached steady state. A warm-up phase that long makes any early
sample meaningless.

## Background: researched, then measured — 2026-09-02

The generated room was built from first principles and it held up: warm-dominant
wall matched to the source key, restrained LED accent at the frame edges, two
static hues, clustered bokeh with an aperture rim, desk as a separate foreground
plane. Surveying how streaming and podcast rooms are actually built and shot
(`docs/background_research.md`) agreed with all of it independently — including
the one place the generic advice says the opposite. Nearly every source
recommends cool blues and greys and warns off warm tones; that advice is for a
room whose lighting you control, and here the source portrait's warm key is
fixed, so lighting match wins and the warm room stays.

Five things were genuinely missing, and are now in:

- **Light wrap.** The subject was not lit by this room, so its outline was a
  clean algebraic cut. `light_wrap()` bleeds the plate's own colour back onto
  the inside of the silhouette — magenta on the left of the head, teal on the
  right, because it samples the actual plate rather than a uniform glow.
- **Exposure fitted to the face instead of tuned by eye.** `fit_exposure()`
  scales the plate to sit the conventional 1-2 stops under the key-lit skin.
  Measured: **-1.51 stops**, reported in the renderer's info line.
- **Cat's-eye bokeh.** Optical vignetting clips off-axis highlights into
  tangentially-elongated lens shapes. Modelled as the intersection of two
  offset circles, which is what physically happens and gets the orientation
  right without a separate rotation.
- **Chromatic fringing** on the glow layer, as a per-channel radial zoom.
- **A third focus plane**: objects on the shelf, blurred less than the wall.

### Two corrections, both caught by looking at a render

**Blurring the shelf board less than the wall was wrong.** It seemed to follow
from "create micro-layers", and it put a hard rule straight across the frame.
The board is attached to the wall, so it is at wall distance; only the objects
standing on it are nearer. Nothing in the numbers flagged this — the room's mean
luminance, determinism and shape were all unchanged. It took one look at the
output.

**The median is the wrong statistic for a face.** A landmark box on this source
is 60 % beard, hair and shadow, so its median reads 72 against lit skin at
140-170, and the plate came out two thirds of a stop too dark. `key_luminance()`
uses the 80th percentile — inside the lit cheek, below the speculars. Once
corrected, the fitted exposure landed within 3 % of the constants that had been
tuned by eye, which is the strongest available evidence that both are right.

### Cost

The wrap is the only addition on the per-frame path. Indexed by boolean mask it
measured **0.89 ms/frame**; addressing the same ~5 300 pixels by integer index
instead measured **0.11 ms**, which is what shipped — the boolean form makes
numpy scan the whole blend box to find a perimeter's worth of pixels.

A slow LED "breathe" was considered and rejected. The ambient-loop literature
supports it (intensity-only modulation cannot warp or wobble, so it does not
violate the stability requirement), but it would force the background to be
recomposed every frame, undoing the optimisation that took compositing from
44 ms down. A 2 % brightness wobble is not worth that at 13.4 FPS.

### Not fixed, and not a background problem

The dark wedge visible at the presenter's lower left is the office chair's frame
in the source photograph, pulled in by the DeepLabV3 person matte. It predates
this work and belongs to matting, not to the environment.

## Wardrobe: clothing and head attire — 2026-09-02

Two dropdowns in the preview window, and the machinery behind them.

### The constraint that decided the design

An outfit cannot be a layer. The torso in the output is static pixels lifted
from the source portrait, the head is warped from appearance features extracted
from that same image at startup, and `AvatarPose` carries head angles and eyelid
openness — not a body. Nothing downstream *can* put a garment on someone. So
changing clothes means changing the image the whole pipeline was prepared from,
which is `set_source()`: swap every per-source array at once, cache the prepared
result, and leave the model weights and the room style alone.

Measured: **6.2 s** for a first switch, **0.00 s** for one already visited.
Returning to a previous outfit reproduces its frame byte-for-byte, and a cached
outfit is byte-identical to a freshly prepared one — both asserted directly
rather than eyeballed.

`_SOURCE_STATE` lists the 36 attributes that belong to a source rather than to
the renderer. It is checked against the code by a test that walks the AST of the
three methods that populate them, because the failure mode of a missing entry is
not a crash — it is the new face rendered against the previous outfit's mask.

### What the generation actually took

`tools/generate_wardrobe.py` inpaints the base portrait with the face masked out
of the edit, so identity, beard, skin and key light are copied through and only
the garment is denoised. Four things had to be found the hard way:

**CLIP truncates at 77 tokens, silently.** The first version paired a 53-token
garment description with 56 tokens of photographic direction and lost 31 off the
end — which is to say it lost the *entire* photographic direction while
appearing to work, and the results looked like costume-shop stock photos with no
error anywhere. `check_prompts()` now runs before anything is generated and it
caught a 1-token overrun on its very first run.

**SDXL base cannot insert a headdress.** Its UNet takes 4 channels and was never
trained on mask conditioning. Across three rounds: at strength 0.97 it
re-imagined the skull into a wound turban, at 0.85 it stopped producing a
headdress at all and merely restyled the hair, and with the drape mask it put
the cloth on the neck as a scarf. Garments worked throughout — a shirt is a
plausible continuation of a torso — but a ghutra is not a plausible continuation
of a scalp. This is what the 9-channel inpainting checkpoint exists for, and it
is the one thing here that needed a download rather than a better prompt.

**The mask has to match the garment's actual shape.** A horizontal cut at the
chin leaves the collar untouched, because shoulders sit *higher* than the chin —
the garment mask starts above the shoulder line with the head punched out
instead. And a crown-plus-two-side-panels headwear mask left the hairline and
headphone band partly outside it, so the model continued that context and
returned braided hair and reconstructed headphone cups. One contiguous bell from
crown to shoulders fixed it.

**Naming the garment is not enough.** Three prompt shapes were generated side by
side. "A white ghutra draped over his head, Gulf Arab headdress" returns braided
hair. A literal description without the culture returns the same. Only
establishing the wearer and then *defining* the object — "a Saudi man wearing a
white ghutra and black agal, the ghutra is a large plain white cotton cloth
covering his head..." — produces cloth.

### Not delivered, and labelled as such

**The agal does not render.** The black cord ring never appeared as a clause
inside the ghutra prompt — the model spends the whole mask on the cloth. A
dedicated third pass over the crown, on the reasoning that the taqiyah worked
precisely because it was a small mask with one object in the prompt, was tried
and was *worse*: it repainted the top of the head as bare scalp. The menu labels
were changed from "White ghutra + agal" to "White ghutra" rather than claiming
something the image does not contain.

### Incidental fix

`_person_matte` built and destroyed a ResNet-101 on every call. That was
defensible when a matte happened once per process; it now happens twice per
source prepare, and a source is prepared every time the presenter changes
clothes. The segmenter is kept resident (~230 MB against 14.7 GB free) and
released in `close()`.

### One item is named for what it renders, not what was asked for

The white thobe comes out mid-grey, consistently. It is genuinely grey rather
than white under-exposed: in the very same frames the ghutra renders bright
white under the same key light. The likely reason is that the garment mask is
large and its entire border is the grey tee underneath, so the model follows the
surrounding colour, while the ghutra's mask borders hair and background instead.

Pushing the prompt harder made it worse rather than whiter — "bright pure white,
brilliant white cotton" turned the garment into a grey waistcoat over a beige
kurta, twice. (That experiment was initially run with a guidance change at the
same time, which proved nothing; separating them showed the prompt was the
cause.) The original wording produces a clean thobe with a correct collar and
placket, so it stays and the entry is called `thobe_grey` / "Grey thobe".

Same principle as the agal: the menu says what is in the picture.
