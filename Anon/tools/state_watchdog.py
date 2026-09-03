"""Run the behaviour engine long and report anything that looks inhuman.

Warnings first, numbers second, deliberately: a wall of green metrics is
exactly what was on screen the last time the presenter looked like a
photograph.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[20260902, 7, 33, 101, 202])
    args = ap.parse_args()

    from presenter.behavior.engine import BehaviorEngine
    from presenter.behavior.watchdog import Watchdog

    total = 0
    for seed in args.seeds:
        engine = BehaviorEngine(seed=seed)
        watch = Watchdog()
        dt = 1.0 / args.fps
        for _ in range(int(args.minutes * 60 * args.fps)):
            engine.update(dt)
            watch.update(dt, engine.motion, engine.attention)

        warnings = watch.check()
        total += len(warnings)
        s = watch.summary()
        print(f"\n[watch] seed {seed}  {args.minutes:.0f} min")
        if warnings:
            for m in warnings:
                print(f"[watch]   WARN  {m}")
        else:
            print("[watch]   no warnings")
        fam = "  ".join(f"{k} {100 * v:.0f}%"
                        for k, v in s["family_share"].items())
        print(f"[watch]   yaw sd {s['yaw_sd']:.2f}  "
              f"abs yaw p95 {s['yaw_p95']:.1f}  "
              f"pitch mean {s['pitch_mean']:+.1f}")
        print(f"[watch]   longest fixation {s['longest_fixation']:.0f}s  "
              f"max camera gap {s['max_camera_gap']:.0f}s  "
              f"engagement {s['engagement_mean']:+.2f} "
              f"sd {s['engagement_sd']:.2f}")
        print(f"[watch]   {fam}")

    print(f"\n[watch] {'PASS - no warnings' if not total else str(total) + ' warnings'}")
    return 0 if not total else 1


if __name__ == "__main__":
    raise SystemExit(main())
