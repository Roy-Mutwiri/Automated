"""Thirty-minute background stability test.

Answers one question: over a long session, do room pixels that do not belong to
the human change at all?

## Why this samples rather than renders everything

Thirty minutes at 30 fps is 54,000 frames, which at this renderer's four frames
a second is nearly four hours of GPU. Sampling one frame every few seconds
across the *full* thirty minutes of simulated behaviour costs minutes and tests
the same thing better, because cumulative drift is a function of how far the
behaviour has travelled, not of how many frames were written to disk. The
engine is stepped at the full rate throughout; only the rendering is sampled.

## What counts as background

Everything outside the renderer's dynamic-human region - the segmented
silhouette grown to cover the head's whole excursion. That region is where the
generator is allowed to write; outside it, every pixel should equal the
reference frame exactly, for the entire session.

Reported: max drift, mean drift over changed pixels, and the longest run of
consecutive samples with zero drift.

Usage
-----
    python tools/background_stability.py --minutes 30 --interval 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_v04_final.png")
    ap.add_argument("--root", default="third_party/LivePortrait")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--interval", type=float, default=3.0,
                    help="seconds of simulated time between rendered samples")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out", default="background_stability.json")
    args = ap.parse_args()

    from presenter.behavior.engine import BehaviorEngine
    from presenter.render.liveportrait import LivePortraitRenderer

    r = LivePortraitRenderer(
        source_image=args.source, liveportrait_root=args.root,
        output_size=(1920, 1080), framing="full", environment="source",
        neutralize_pose=0.0,
    )

    region = getattr(r, "dynamic_human_out", None)
    if region is None:
        print("[bg] WARNING: no dynamic-human region; the whole crop is writable")
        static = None
    else:
        static = region[:, :, 0] <= 0.0
        print(f"[bg] static background is {100 * static.mean():.1f}% of the frame")

    engine = BehaviorEngine(seed=args.seed)
    dt = 1.0 / args.fps
    step = max(int(args.interval * args.fps), 1)
    n = int(args.minutes * 60.0 * args.fps)

    ref = None
    samples = []
    zero_run = best_run = 0
    for i in range(n):
        pose = engine.update(dt)
        if i % step:
            continue
        frame = r.render(pose)
        if static is None:
            static = np.ones(frame.shape[:2], bool)
        if ref is None:
            ref = frame.copy()
            continue
        d = cv2.absdiff(frame, ref).max(axis=2)
        vals = d[static]
        mx = int(vals.max())
        changed = int((vals > 0).sum())
        samples.append(dict(t=i * dt, max=mx, changed=changed,
                            mean_changed=float(vals[vals > 0].mean()) if changed else 0.0))
        zero_run = zero_run + 1 if mx == 0 else 0
        best_run = max(best_run, zero_run)
        if len(samples) % 50 == 0:
            print(f"[bg] {i * dt / 60:5.1f} min  samples {len(samples)}  "
                  f"max so far {max(s['max'] for s in samples)}", flush=True)

    overall_max = max(s["max"] for s in samples)
    overall_mean = float(np.mean([s["mean_changed"] for s in samples]))
    total_changed = int(np.mean([s["changed"] for s in samples]))
    report = dict(
        minutes=args.minutes, samples=len(samples),
        static_fraction=float(static.mean()),
        max_drift=overall_max, mean_drift_over_changed=overall_mean,
        mean_changed_pixels=total_changed,
        longest_zero_run_samples=best_run,
        longest_zero_run_seconds=best_run * args.interval,
        per_sample=samples[:400],
    )
    Path(args.out).write_text(json.dumps(report, indent=2))

    print()
    print(f"[bg] MAX STATIC PIXEL DRIFT      : {overall_max}")
    print(f"[bg] MEAN DRIFT (changed pixels) : {overall_mean:.3f}")
    print(f"[bg] MEAN CHANGED PIXELS/SAMPLE  : {total_changed} of "
          f"{int(static.sum())} static")
    print(f"[bg] LONGEST ZERO-DRIFT RUN      : {best_run} samples "
          f"({best_run * args.interval:.0f} s of simulated time)")
    print(f"[bg] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
