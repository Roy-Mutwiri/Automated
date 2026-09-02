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
