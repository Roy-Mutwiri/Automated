"""Milestone 1: five minutes of a silent human, plus the evidence to judge it.

No audio, no speech, no lip sync, no LLM. The presenter sits, breathes, blinks,
looks at things and occasionally shifts, and the test is whether five minutes of
that can be watched without identifying a loop, a timer, or a repeated
movement.

Produces, in one pass:

* `silent_human_5min.mp4` - the actual test. Metrics cannot pass this milestone;
  only watching it can.
* `human_behavior_timeline.csv` - every event with its time, so a suspicious
  moment in the video can be traced to the decision that caused it.
* `human_behavior_metrics.json` - distributions, not just counts. A mean blink
  rate proves nothing; the *coefficient of variation* of the intervals is what
  separates a human from a timer.
* screenshots at fixed timestamps.

The engine is stepped at a fixed 1/30 s regardless of how long a frame takes to
render, so the behaviour is the behaviour of a real five minutes even though the
render takes considerably longer than five minutes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SHOT_TIMES = (7.0, 42.0, 96.0, 155.0, 211.0, 268.0)


def summarise(events, minutes: float, poses) -> dict:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for e in events:
        by_kind[e.kind].append(e.time)

    def dist(times):
        gaps = [b - a for a, b in zip(times, times[1:])]
        if len(gaps) < 3:
            return None
        mean = statistics.fmean(gaps)
        sd = statistics.pstdev(gaps)
        return dict(
            count=len(times), per_minute=len(times) / minutes,
            mean_interval=mean, median_interval=statistics.median(gaps),
            sd_interval=sd,
            # The number that matters. A fixed timer has cv ~ 0. Poisson-like
            # arrivals sit near 1. Anything under ~0.35 will be *seen* as a
            # rhythm however carefully the mean was chosen.
            cv=sd / mean if mean > 0 else 0.0,
            min_interval=min(gaps), max_interval=max(gaps),
        )

    # Families, because the engine emits head_yaw / head_pitch / head_roll
    # separately and "how often does he move his head" is one question, not
    # three. Reported alongside the raw kinds rather than instead of them.
    FAMILIES = {
        "blink_all": ("blink", "blink_partial", "double_blink_second"),
        "head_all": ("head_yaw", "head_pitch", "head_roll"),
        "voluntary_all": ("attention", "head_yaw", "head_pitch", "head_roll",
                          "posture_shift", "expression"),
    }
    for fam, kinds in FAMILIES.items():
        times = sorted(t for k in kinds for t in by_kind.get(k, []))
        if times:
            by_kind[fam] = times

    out: dict = {"minutes": minutes, "events": {}}
    for kind, times in sorted(by_kind.items()):
        d = dist(times)
        if d:
            out["events"][kind] = d
        else:
            out["events"][kind] = dict(count=len(times),
                                       per_minute=len(times) / minutes)

    targets = Counter(e.metadata.get("target") for e in events
                      if e.kind == "attention" and e.metadata)
    total = sum(targets.values()) or 1
    out["attention_share"] = {k: v / total for k, v in targets.most_common()}

    states = [e.detail.split(" -> ")[-1] for e in events if e.kind == "state_change"]
    out["state_sequence"] = states
    out["state_visits"] = dict(Counter(states).most_common())

    # Stillness: fraction of time with no voluntary event inside a window.
    vol = sorted(e.time for e in events
                 if e.kind in ("attention", "head_move", "posture", "expression"))
    quiet = []
    for a, b in zip(vol, vol[1:]):
        quiet.append(b - a)
    if quiet:
        out["stillness"] = dict(
            longest_quiet_s=max(quiet),
            median_quiet_s=statistics.median(quiet),
            fraction_over_3s=sum(1 for q in quiet if q > 3.0) / len(quiet),
            fraction_over_8s=sum(1 for q in quiet if q > 8.0) / len(quiet),
        )

    yaw = [p.yaw for p in poses]
    gx = [p.gaze_x for p in poses]
    out["pose"] = dict(
        yaw_max=max(map(abs, yaw)), yaw_sd=statistics.pstdev(yaw),
        gaze_x_max=max(map(abs, gx)), gaze_x_sd=statistics.pstdev(gx),
        # Frame-to-frame velocity. A large max here is a pop.
        yaw_max_step=max(abs(b - a) for a, b in zip(yaw, yaw[1:])),
        gaze_max_step=max(abs(b - a) for a, b in zip(gx, gx[1:])),
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_v04_final.png")
    ap.add_argument("--root", default="third_party/LivePortrait")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--profile", default="PRESENTER_CALM")
    ap.add_argument("--video", default="silent_human_5min.mp4")
    ap.add_argument("--csv", default="human_behavior_timeline.csv")
    ap.add_argument("--metrics", default="human_behavior_metrics.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="behaviour only, no renderer; for fast metric checks")
    args = ap.parse_args()

    from presenter.behavior.engine import BehaviorEngine
    from presenter.types import AvatarPose

    engine = BehaviorEngine(profile=args.profile, seed=args.seed)
    dt = 1.0 / args.fps
    n = int(args.minutes * 60.0 * args.fps)

    renderer = None
    writer = None
    if not args.dry_run:
        from presenter.render.liveportrait import LivePortraitRenderer
        renderer = LivePortraitRenderer(
            source_image=args.source, liveportrait_root=args.root,
            output_size=(args.width, args.height),
            framing="full", environment="source", neutralize_pose=0.0,
        )
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open {args.video} for writing")

    events = []
    poses = []
    shots = {t: None for t in SHOT_TIMES}
    t0 = time.perf_counter()

    for i in range(n):
        pose = engine.update(dt)
        poses.append(AvatarPose(**vars(pose)))
        events.extend(engine.drain_events())

        if renderer is not None:
            frame = renderer.render(pose)
            writer.write(frame)
            now = i * dt
            for st in shots:
                if shots[st] is None and now >= st:
                    shots[st] = frame.copy()

            if i % 300 == 0 and i:
                el = time.perf_counter() - t0
                print(f"[silent] {i}/{n} frames  {i / el:.2f} fps  "
                      f"eta {(n - i) / max(i / el, 1e-3) / 60:.1f} min",
                      flush=True)

    if writer is not None:
        writer.release()
        for st, frame in shots.items():
            if frame is not None:
                cv2.imwrite(f"silent_human_t{int(st):03d}s.png", frame)
        print(f"[silent] video -> {args.video}")

    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_seconds", "timestamp", "kind", "detail", "magnitude"])
        for e in events:
            w.writerow([f"{e.time:.3f}", e.timestamp(), e.kind, e.detail,
                        f"{e.magnitude:.4f}"])
    print(f"[silent] timeline -> {args.csv}  ({len(events)} events)")

    metrics = summarise(events, args.minutes, poses)
    metrics["seed"] = args.seed
    metrics["profile"] = args.profile
    metrics["repeated_ngrams"] = [
        {"gram": list(g), "count": c}
        for g, c in engine.memory.repeated_ngrams(n=3, min_repeats=3)
    ]
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    print(f"[silent] metrics -> {args.metrics}")

    ev = metrics["events"]
    for kind in ("blink_all", "attention", "head_all", "posture_shift",
                 "expression", "state_change", "voluntary_all"):
        d = ev.get(kind)
        if d and "cv" in d:
            print(f"[silent] {kind:10} {d['per_minute']:6.2f}/min  "
                  f"median {d['median_interval']:6.2f}s  cv {d['cv']:.2f}")
    if "stillness" in metrics:
        s = metrics["stillness"]
        print(f"[silent] stillness: longest quiet {s['longest_quiet_s']:.1f}s, "
              f"median {s['median_quiet_s']:.1f}s, "
              f"{100 * s['fraction_over_3s']:.0f}% of gaps > 3s")
    print(f"[silent] pose: max|yaw| {metrics['pose']['yaw_max']:.2f} deg, "
          f"max step {metrics['pose']['yaw_max_step']:.3f} deg/frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
