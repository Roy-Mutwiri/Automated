# Human presenter behaviour: the measurements the engine is built on

Every default in `behavior/state.py` traces to something here. Where the
literature and the requirement disagree, the disagreement is noted rather than
silently resolved.

## Blinking

| Condition | Rate (blinks/min) | Source |
|---|---|---|
| Normal population, resting | 20–30 (mean 26.1, wide inter-subject spread) | [J. Neurosci 31(31)](https://www.jneurosci.org/content/31/31/11256), [Sci Direct](https://www.sciencedirect.com/science/article/abs/pii/S0892896799000164) |
| Active conversation | 32.4 ± 12.4 | [PubMed 24413278](https://pubmed.ncbi.nlm.nih.gov/24413278/) |
| Normal visual conditions | ~12 | as above |
| Reading (screen or print) | 5–11 (14.9 screen / 13.6 hard copy in one controlled study) | [PubMed 36763349](https://pubmed.ncbi.nlm.nih.gov/36763349/) |
| Cognitive load / focus | reduced | [Front. Psychol.](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2022.788231/full) |

Key finding: **task and cognitive demand drive blink rate far more than the
display medium.** Screen vs. print made little difference; conversation vs.
reading made a threefold one.

**Chosen baseline: 4.0 s median interval (~15/min) for `PRESENTER_CALM` in
`IDLE_ATTENTIVE`.** This is deliberately *not* the 26/min resting figure. A
presenter attending to a camera is closer to the focused end than to
unconstrained rest, and an avatar blinking every 2.3 s reads as nervous.
Measured output: 14.7–15.5/min. `SPEAKING` multiplies by 1.9 → 25.3/min,
landing inside the reported conversation range; `FOCUSED` halves it → 9.6/min.

**Interval distribution: log-normal.** Inter-blink intervals are positively
skewed, sometimes bimodal, never uniform ([PMC6207319](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6207319/),
[PubMed 20944934](https://pubmed.ncbi.nlm.nih.gov/20944934/)). A log-normal
reproduces the skew with one intuitive parameter (the median) and one shape
parameter. Measured CV ≈ 0.63–0.73 — well clear of a metronome (CV 0) and below
a memoryless Poisson process (CV 1), which is where behaviour with a refractory
period belongs.

**Kinematics.** Closing is much faster than reopening; the fast phase is an
orbicularis twitch, the slow return is levator re-engagement. Implemented as
`close_fraction = 0.36` with different curves per phase
(`curves.blink_profile`). Total duration ~145 ms ± 25 ms.

## Fixational eye movement

| Component | Amplitude | Rate / speed | Source |
|---|---|---|---|
| Microsaccades | < 1° visual angle | 1–2/s (range 1–3/s), 6–30 ms duration | [Front. Psychol. mini-review](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1364939/full), [ScienceDirect](https://www.sciencedirect.com/topics/neuroscience/microsaccade) |
| Drift | < 0.13° | < 0.5°/s | as above |
| Microsaccade peak velocity | — | 6–120°/s | as above |

**This is why the engine models three separate ocular processes.** Microsaccades
at 1.4 Hz (measured 1.33–1.42/s) are individually invisible; their *absence* is
not. A mathematically fixed pupil is the clearest single sign of a dead face,
so microsaccades and drift run continuously and are never suppressed by the
stillness logic.

Saccades are ballistic — tens of milliseconds, 1–3 frames at 30 FPS. Animating
a gaze shift as a visible glide across the eye is wrong.

## Head motion

No single authoritative rate exists for seated presenters, so the defaults here
are constrained by anatomy and by the brief rather than by a citation, and were
tuned against the audit tool:

- Yaw held within ±6°, pitch ±4.5°, roll ±3.5° for `PRESENTER_CALM`. The
  intuitive setting is far larger and produces the "bobbing" the brief lists as
  a bug.
- Deliberate adjustments at a 15 s median → measured 2.8–3.3/min.
- Continuous involuntary sway (σ ≈ 0.22°) always present underneath.

**Movement trajectory: minimum jerk.** Human point-to-point movement closely
follows the minimum-jerk quintic (Flash & Hogan 1985, *J. Neuroscience* 5(7)).
Used for every voluntary transition.

## Breathing

Quiet seated respiration is 12–18 breaths/min → 3.3–5.0 s per cycle, with a
shorter active inhale and longer passive exhale plus a brief end-expiratory
pause. Default period 4.1 s ± 0.45; measured 13.5–13.8/min.

Amplitude is the parameter most likely to be set too high. On a head-and-
shoulders crop the perceptible cue is a sub-percent scale change and a fraction
of a degree of pitch — not visible chest movement.

## Distribution choices, and why

| Behaviour | Distribution | Reason |
|---|---|---|
| Blink interval | Log-normal | Matches the reported positive skew; refractory period rules out exponential |
| Gaze shift interval | Log-normal | Same reasoning; long tail gives genuine fixations |
| Head / expression / posture interval | Log-normal | Skew plus a hard floor from the refractory gate |
| Microsaccades | Exponential (Poisson) | Genuinely memoryless — no strong refractory structure |
| Head sway, ocular drift, posture drift, arousal | Ornstein–Uhlenbeck | Needs to be correlated, bounded and **aperiodic**; a sine becomes visible once watched longer than its period |
| Movement amplitudes | Truncated Gaussian | Natural centre with a hard anatomical limit |

**Uniform distributions are used nowhere for timing.** They produce intervals
that are too evenly spread, which is precisely the artificial texture the brief
rejects.

## Where the brief overrides the literature

Resting blink rate (~26/min) and voluntary gaze-shift rates taken at face value
produce an avatar that moves noticeably more than the brief permits. An early
tuning pass sat at ~22 voluntary movements/min — every measurement inside its
published range, and visibly fidgety.

The engine therefore targets the quiet end of every range, and
`tools/behavior_timeline.py` enforces a hard stillness floor: median gap ≥ 3.0 s
between voluntary movements, ≥ 50 % of gaps over 3 s, ≥ 20 % over 5 s, at least
one gap over 10 s, and ≤ 17 voluntary movements/min. Current `PRESENTER_CALM`:
median gap 3.8 s, longest 22 s, 12.4/min.
