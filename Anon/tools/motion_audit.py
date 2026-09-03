"""Measure stillness in the rendered video, not in the engine.

The judgement that mattered most in this project was made by a person watching
a clip: "an image that sometimes moves". Every engine-side metric was green at
the time. So this measures the same property the viewer did, on the same
artefact the viewer saw - the encoded pixels - and it is deliberately blind to
what the behaviour engine believes it was doing.

## What it measures

Per frame, the mean absolute difference from the previous frame inside the head
region, in 8-bit levels. That single number separates the three regimes that
matter:

* **dead** - below `--dead`. Nothing is moving. A run of these is a photograph.
* **micro** - breathing, micro-saccades, the eyelid at rest. Alive, not
  legible as an action.
* **overt** - a head turn, a blink, an expression: something a viewer will see
  and be able to name.

The headline number is the **longest dead run**, because that is what the
failure actually felt like. A clip can have a perfectly good mean and still
contain a nine-second window where the man is a still photograph, and the mean
will never show it.

## What it does not measure

Whether the motion is *plausible*. A head that jerks between two poses thirty
times a second scores wonderfully here. This is a floor, not a rubric: it can
prove the presenter is not a photograph, and it cannot prove he is a person.
That judgement is still made by watching, and this tool exists to point at the
seconds most worth watching - it writes out the stillest and the busiest
windows as frame strips so the eye goes straight to them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--head", type=int, nargs=4, default=(790, 60, 1330, 620),
                    metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--dead", type=float, default=0.35,
                    help="mean abs level change below which a frame is dead")
    ap.add_argument("--overt", type=float, default=1.6,
                    help="above which the change is legible as an action")
    ap.add_argument("--out", default=None, help="JSON summary path")
    ap.add_argument("--strips", action="store_true",
                    help="write the stillest and busiest windows as strips")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[audit] cannot open {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    x0, y0, x1, y1 = args.head

    diffs: list[float] = []
    prev = None
    frames_kept: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        head = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(head - prev).mean()))
        prev = head
        if args.strips:
            frames_kept.append(frame)
    cap.release()

    if not diffs:
        print("[audit] no frames")
        return 1

    d = np.asarray(diffs)
    n = len(d)
    dead = d < args.dead
    overt = d > args.overt

    # Longest consecutive run of dead frames.
    longest = cur = 0
    longest_end = 0
    for i, isdead in enumerate(dead):
        if isdead:
            cur += 1
            if cur > longest:
                longest, longest_end = cur, i
        else:
            cur = 0
    longest_s = longest / fps
    still_start = (longest_end - longest + 1) / fps

    # Busiest one-second window.
    win = max(int(fps), 1)
    if n >= win:
        sums = np.convolve(d, np.ones(win), "valid")
        busy_at = int(np.argmax(sums)) / fps
        quiet_at = int(np.argmin(sums)) / fps
    else:
        busy_at = quiet_at = 0.0

    summary = dict(
        video=args.video, frames=n, seconds=n / fps, fps=fps,
        mean_change=float(d.mean()), median_change=float(np.median(d)),
        p95_change=float(np.quantile(d, 0.95)), max_change=float(d.max()),
        dead_fraction=float(dead.mean()), overt_fraction=float(overt.mean()),
        longest_dead_run_s=longest_s, longest_dead_run_at_s=still_start,
        busiest_second_at_s=busy_at, quietest_second_at_s=quiet_at,
    )

    print(f"\n[audit] {Path(args.video).name}  "
          f"{n / fps:.0f}s at {fps:.0f} fps")
    print(f"[audit]   frame-to-frame change in head region (8-bit levels)")
    print(f"[audit]     mean {d.mean():.2f}   median {np.median(d):.2f}   "
          f"p95 {np.quantile(d, 0.95):.2f}   max {d.max():.2f}")
    print(f"[audit]   dead frames (<{args.dead})  "
          f"{100 * dead.mean():.1f}% of the clip")
    print(f"[audit]   overt frames (>{args.overt}) "
          f"{100 * overt.mean():.1f}% of the clip")
    print(f"[audit]   LONGEST DEAD RUN  {longest_s:.2f}s  "
          f"starting at {still_start:.1f}s")
    print(f"[audit]   quietest second at {quiet_at:.1f}s, "
          f"busiest at {busy_at:.1f}s")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))

    # A change-over-time plot: a picture of the whole clip's liveliness.
    h, w = 240, min(1800, max(600, n // 4))
    plot = np.full((h, w, 3), 22, np.uint8)
    scale = max(float(np.quantile(d, 0.995)), 1e-3)
    xs = (np.arange(n) / n * (w - 1)).astype(int)
    ys = (h - 1 - np.clip(d / scale, 0, 1) * (h - 20)).astype(int)
    for i in range(n):
        cv2.line(plot, (xs[i], h - 1), (xs[i], ys[i]), (90, 190, 90), 1)
    for lvl, col, lab in ((args.dead, (80, 80, 220), "dead"),
                          (args.overt, (220, 180, 80), "overt")):
        yy = int(h - 1 - min(lvl / scale, 1.0) * (h - 20))
        cv2.line(plot, (0, yy), (w - 1, yy), col, 1)
        cv2.putText(plot, lab, (4, max(yy - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    out_plot = Path(args.video).with_suffix("").name + "_motion.png"
    cv2.imwrite(out_plot, plot)
    print(f"[audit]   plot -> {out_plot}")

    if args.strips and frames_kept:
        def strip(name, t0):
            i0 = int(t0 * fps)
            picks = [frames_kept[min(i0 + int(k * fps * 0.5), n - 1)]
                     for k in range(6)]
            cells = []
            for k, f in enumerate(picks):
                c = f[y0:y1, x0:x1]
                c = cv2.resize(c, (300, int(300 * c.shape[0] / c.shape[1])))
                c = cv2.copyMakeBorder(c, 22, 2, 2, 2, cv2.BORDER_CONSTANT,
                                       value=(20, 20, 20))
                cv2.putText(c, f"t={t0 + k * 0.5:.1f}s", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 235, 255), 1)
                cells.append(c)
            p = Path(args.video).with_suffix("").name + f"_{name}.png"
            cv2.imwrite(p, np.hstack(cells))
            print(f"[audit]   {name} strip -> {p}")

        strip("stillest", max(still_start, 0.0))
        strip("busiest", max(busy_at - 0.5, 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
