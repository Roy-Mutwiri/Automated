"""Does the presenter behave the same at 30 fps and 60 fps?

Frame rate is a rendering decision. It must not be a behavioural one. If the
man blinks more often, shifts posture sooner or looks around faster because the
clip is being rendered at 60, then the frame rate is secretly a personality
setting and every measurement taken at one rate is worthless at the other.

One such bug has already been found and fixed here: the voluntary-action gate
was a per-*frame* probability, so 60 fps rolled the dice twice as often and
produced 12.7% more blinks than 25 fps. That is exactly what this looks for.

## Why this is a paired test with a standard error

The first version compared three seeds and called anything outside the
seed-to-seed spread rate-dependent. It flagged four measures, and all four were
noise. The reason is worth writing down: the posture drift is a mean-reverting
process with a 42-second correlation time, so a six-minute window contains
about nine independent samples and its standard deviation is uncertain to a
quarter of its own value. Three seeds cannot see through that.

So each seed is now run at *both* rates and the **paired difference** is what
gets tested, against the standard error of those differences across seeds. That
cancels the seed-to-seed variation, which is the dominant term, and leaves only
the rate effect. Verified separately: the drift process alone, over 240 runs per
rate, gives 0.1176 at 30 fps and 0.1168 at 60 - a 0.6-sigma difference.

## What equivalence can and cannot mean

The random stream is consumed per frame, so 60 fps draws roughly twice as many
numbers as 30 and the two runs cannot produce an identical event list. Demanding
that would require a time-quantised RNG, which is a real option but a large
change.

So this checks the property that actually matters, which is stronger than it
sounds: **rates, durations and pose statistics must agree**. If the gates are
rate-independent, a blink every 4.1 s at 30 fps is a blink every 4.1 s at 60. If
any of them is not, the number moves, and it moves by a lot - the bug above
showed up as a 12.7% shift, far outside the seed-to-seed spread this reports as
its own noise floor.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(seed: float, fps: float, minutes: float) -> dict:
    from presenter.behavior.engine import BehaviorEngine

    e = BehaviorEngine(seed=seed)
    dt = 1.0 / fps
    n = int(minutes * 60 * fps)

    blinks = 0
    was_closed = False
    shifts = 0
    subfix = 0
    last_target = None
    intentions = 0
    last_state = None
    yaw, pitch, eng = [], [], []
    fam_time: dict[str, float] = {}

    for _ in range(n):
        e.update(dt)
        m = e.motion
        closed = m.face.eye_open_l < 0.35
        if closed and not was_closed:
            blinks += 1
        was_closed = closed
        if m.attention.target != last_target:
            if last_target is not None:
                shifts += 1
            last_target = m.attention.target
        if m.behavior_state != last_state:
            if last_state is not None:
                intentions += 1
            last_state = m.behavior_state
        fam = e.attention.targets[e.attention.current].family
        fam_time[fam] = fam_time.get(fam, 0.0) + dt
        yaw.append(m.head_world_yaw())
        pitch.append(m.head_world_pitch())
        eng.append(m.posture.engagement)

    subfix = getattr(e.attention, "subfix_count", 0)
    total = minutes
    return dict(
        blinks_per_min=blinks / total,
        shifts_per_min=shifts / total,
        subfix_per_min=subfix / total,
        intentions_per_min=intentions / total,
        yaw_mean=statistics.fmean(yaw), yaw_sd=statistics.pstdev(yaw),
        pitch_mean=statistics.fmean(pitch), pitch_sd=statistics.pstdev(pitch),
        eng_sd=statistics.pstdev(eng),
        camera_share=100.0 * fam_time.get("CAMERA", 0.0) / (minutes * 60),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[20260902, 7, 33, 101, 202, 303, 404, 505, 606,
                             707, 808, 909])
    args = ap.parse_args()

    keys = ["blinks_per_min", "shifts_per_min", "subfix_per_min",
            "intentions_per_min", "yaw_mean", "yaw_sd", "pitch_mean",
            "pitch_sd", "eng_sd", "camera_share"]

    at30: dict[str, list[float]] = {k: [] for k in keys}
    at60: dict[str, list[float]] = {k: [] for k in keys}
    for s in args.seeds:
        a, b = run(s, 30.0, args.minutes), run(s, 60.0, args.minutes)
        for k in keys:
            at30[k].append(a[k])
            at60[k].append(b[k])
        print(f"[fps] seed {s} done", flush=True)

    print(f"\n[fps] {args.minutes:.0f} min x {len(args.seeds)} seeds, paired\n")
    print(f"[fps] {'measure':<20}{'30 fps':>10}{'60 fps':>10}{'delta':>9}"
          f"{'SE':>8}{'t':>7}   verdict")
    bad = 0
    n = len(args.seeds)
    for k in keys:
        diffs = [b - a for a, b in zip(at30[k], at60[k])]
        md = statistics.fmean(diffs)
        se = (statistics.stdev(diffs) / (n ** 0.5)) if n > 1 else 0.0
        # Two standard errors on the *paired* difference. Not the spread across
        # seeds: that is dominated by seed-to-seed variation, which pairing
        # cancels, and which made the first version of this test cry wolf four
        # times out of ten.
        t = md / se if se > 1e-12 else 0.0
        ok = abs(t) < 2.0
        bad += 0 if ok else 1
        print(f"[fps] {k:<20}{statistics.fmean(at30[k]):>10.3f}"
              f"{statistics.fmean(at60[k]):>10.3f}{md:>+9.3f}{se:>8.3f}"
              f"{t:>+7.2f}   {'ok' if ok else 'RATE-DEPENDENT'}")
    print(f"\n[fps] {'PASS' if not bad else str(bad) + ' rate-dependent measures'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
