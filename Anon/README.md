# Anon — Real-time AI Human Presenter (visual system)

The **visual half** of a real-time AI presenter: a photorealistic human that
looks alive while sitting in front of a camera, whether or not it is speaking.

This project handles appearance and behaviour only. Dialogue, LLM, TTS and
voice are Developer A's scope and are deliberately absent here. No webcam, no
microphone, no face tracking — the avatar behaves autonomously.

> **Current status: photoreal renderer working; behaviour engine complete and
> validated. Frame rate is 13.4 FPS against a 25 FPS target — the cause is
> identified and is not a quality trade-off.** See
> [Performance](#performance-honest-numbers) and [Status](#status).

## The actual problem

Not "how do I animate a picture", but *how do I reproduce the tiny unconscious
behaviours that make a person look alive* — and, harder, **when not to move at
all.** An avatar that moves continuously reads as artificial no matter how good
each individual motion is. Real humans are still most of the time.

## Quick start

```bash
# Python 3.11 + PyTorch for Blackwell (sm_120) - see GPU notes below
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -e .

# photoreal avatar
.venv/Scripts/python.exe -m presenter.app --renderer liveportrait --debug --compile

# behaviour preview (schematic rig, no GPU needed)
.venv/Scripts/python.exe -m presenter.app --debug

# 30-minute behavioural audit, ~1 second to run
.venv/Scripts/python.exe tools/behavior_timeline.py --minutes 30

# tests
.venv/Scripts/python.exe -m pytest tests/ -q
```

Controls: `1`–`9` switch behaviour state · `d` debug overlay · `s` screenshot ·
`q` quit.

## GPU requirements — read this before installing anything

This machine is an **RTX 5080 Laptop, compute capability 12.0 (Blackwell,
sm_120)**. That constraint invalidates most avatar-repo install instructions:

- Requires **CUDA 12.8+ and PyTorch built with sm_120 kernels**. Verified
  working: `torch 2.11.0+cu128`.
- Repos pinning `torch==2.0.1` / `2.3.0` on cu118/cu121/cu124 will install and
  import fine, then fail at the first kernel launch with
  `no kernel image is available for execution on the device`. **Override their
  pins.**
- Check with: `python -c "import torch; print(torch.cuda.get_arch_list())"` —
  `sm_120` must be listed.

Measured on this GPU: 38 TFLOPS fp16, conv2d 512² in 0.195 ms, 14.7 GB free.

## Architecture

```
profile + state ─▶ BEHAVIOUR ENGINE ─▶ AvatarPose ─▶ RENDERER ─▶ display
                   (no GPU, no I/O)      (the seam)
```

`AvatarPose` is the entire contract between behaviour and rendering. The engine
knows nothing about the renderer and vice versa, which is what lets the
rendering backend be swapped or benchmarked without disturbing months of
behaviour tuning — and what makes lip-sync additive later rather than a
rewrite.

Full detail: [`docs/architecture.md`](docs/architecture.md).

### How stillness works

Three mechanisms, and the third is the important one:

1. **Arousal** — a slow (~26 s) Ornstein–Uhlenbeck signal modulating every rate
   at once, so the avatar has genuinely quiet minutes and livelier ones.
2. **Motion budget** — a leaky accumulator charged by each voluntary movement,
   gating new discretionary behaviour while charged.
3. **Involuntary floor** — breathing, ocular drift, microsaccades and head sway
   are *never* suppressed.

Point 3 is the whole difference between **still** and **frozen**.

## Measured behaviour

`PRESENTER_CALM` / `IDLE_ATTENTIVE`, 30-minute simulation, all profile × seed
combinations passing:

| Metric | Measured | Human reference |
|---|---|---|
| Blink rate | 14.7–15.5/min | 12–20/min resting |
| Blink interval CV | 0.63 | 0 = metronome, 1 = memoryless |
| Voluntary saccades | 7.0/min | — |
| Microsaccades | 1.37/s | 1–2/s |
| Head moves | 2.8/min | — |
| Breaths | 13.7/min | 12–18/min quiet |
| Median stillness gap | 3.8 s | — |
| Longest stillness gap | 22 s | — |
| Voluntary movement rate | 12.4/min | — |
| Repeated 4-grams | none above chance | — |

State modulation matches the literature: `FOCUSED` → 9.6 blinks/min,
`SPEAKING` → 25.3/min (reported conversation rate is 32.4 ± 12.4).

**Frame-rate invariant** — same seed at 25/30/60 Hz gives 15.20/14.65/15.55
blinks per minute. Animation is driven by elapsed time, never frame count.

Sources for every number: [`docs/human_behavior.md`](docs/human_behavior.md).

## Performance: honest numbers

Measured on the RTX 5080 Laptop, 1280×720 output, 150–200 frame runs.

| Configuration | Frame time | p95 | FPS |
|---|---|---|---|
| Initial working version | 116 ms | — | 8.4 |
| + compositing rewrite | 94.5 ms | 107 ms | 10.6 |
| + `torch.compile` | **74.7 ms** | 82.7 ms | **13.4** |

VRAM: 1.25 GB reserved. **Target is 25 FPS minimum. This does not meet it.**

### Why — and why it is not a quality problem

Sampling the GPU under load: **14–18 % utilisation, 1065–1297 MHz, 30 W.** The
GPU sits idle most of every frame. This is a **launch-bound** workload — many
small kernels the CPU cannot dispatch fast enough — not a compute limit. The
evidence is consistent: converting weights fp32→fp16 moved the needle only
68.9→65.0 ms, exactly what you expect when the GPU is not the constraint.

So the remaining performance is available *for free*, without touching visual
quality. Reducing resolution or model size to buy frames would be the wrong
trade: the brief ranks a stable realistic 30 FPS above an unstable 60, and
there is a large amount of headroom being wasted.

**Routes to 25–30 FPS, in order of expected return:**

1. **CUDA graphs.** Directly eliminates launch overhead.
   `torch.compile(mode="reduce-overhead")` currently fails with
   `accessing tensor output of CUDAGraphs that has been overwritten` inside
   `warping_network.forward` — the fix is cloning the module outputs.
2. **TensorRT.** [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait)
   reports 30+ FPS on an RTX 3090 including pre/post-processing. It targets
   TensorRT 8.x and CUDA 12.2, so Blackwell/CUDA 12.8 needs work.
3. **GPU-side compositing.** The warp and blend still run on CPU.

Also worth checking: 30 W draw suggests the laptop may be on a power-saving
profile. Worth confirming before attributing everything to software.

## Configuration

[`config/avatar.yaml`](config/avatar.yaml). All timing values are **medians**,
never fixed intervals. Full parameter set in
`src/presenter/behavior/state.py`.

Three profiles: `PRESENTER_CALM` (default, most tuned), `PRESENTER_ENERGETIC`,
`PRESENTER_FOCUSED`.

## Identity and disclosure

The source portrait must be **either a synthetic identity created for this
project, or a real person's likeness used with explicit documented
permission** — recorded in `config/avatar.yaml` alongside the image. The system
is not designed around impersonating an uninvolved real person.

If this is ever presented publicly as a live presenter, the product must retain
a way to represent it clearly as an AI/virtual presenter rather than implying
the depicted person is physically live. This constraint costs nothing in visual
quality.

## Licensing

- This project's code: see repository.
- **LivePortrait: MIT** (code).
- **InsightFace models: non-commercial research only.** LivePortrait uses them
  for detection by default. **MediaPipe (Apache-2.0) is a documented drop-in
  replacement and is mandatory before any commercial use.**

## Status

**Done and verified**
- Behaviour engine: blinking, gaze (fixation/saccade/microsaccade/drift), head
  motion, breathing, micro-expressions, posture, arousal, motion budget.
- 30-minute audit tool with rate, variability, stillness and loop checks;
  exits non-zero on failure.
- 14 regression tests, all passing.
- Schematic rig preview at 29.7 FPS, 0 render failures over 30 s.
- Renderer chosen (LivePortrait) with reasoning and licence review.

**Blocked**
- Photoreal renderer needs a source portrait. None exists in the repo and the
  choice of face is a product decision, not an engineering one.

**Not started (deliberately)**
- Lip sync, audio, LLM, TTS, OBS, virtual camera, streaming. Out of scope until
  the visual milestone is signed off.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `no kernel image is available` | PyTorch without sm_120. Reinstall from the cu128 index. |
| `ModuleNotFoundError: presenter` | Run `uv pip install -e .` |
| Avatar looks fidgety | Raise `saccade_median_interval` / `head_median_interval`, or lower `activity`. Re-run the audit. |
| Avatar looks frozen | Check `head_sway_amplitude` and `microsaccade_rate` are non-zero. |
| Breathing visible as breathing | `breath_scale_amount` is too high. |
