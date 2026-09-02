"""Pull frames out of the rendered clip at moments the timeline says matter.

The brief is right that metrics are not enough and the video is the test. This
does not replace watching it; it makes the frame-by-frame pass possible, which
is where kinematic faults actually live. A blink that is wrong is wrong over
about six frames, and no summary statistic will ever show it.

Because the behaviour engine is deterministic and the timeline records the
exact second of every event, frames can be pulled at the interesting moments
rather than at arbitrary intervals: the frames *around a blink*, the frames
*around a gaze shift*, the frames around a head turn.

Usage
-----
    python tools/inspect_clip.py --event blink --n 3
    python tools/inspect_clip.py --contact-sheet
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def load_events(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def frames_at(cap, fps, t0, count, step=1):
    out = []
    start = max(int(round(t0 * fps)), 0)
    for k in range(count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + k * step)
        ok, fr = cap.read()
        if not ok:
            break
        out.append((start + k * step, fr))
    return out


def crop_face(fr, zoom=2.0, box=(0.36, 0.06, 0.70, 0.52)):
    h, w = fr.shape[:2]
    x0, y0, x1, y1 = box
    c = fr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    return cv2.resize(c, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)


def strip(frames, labels, zoom, box):
    tiles = []
    for (idx, fr), lab in zip(frames, labels):
        t = crop_face(fr, zoom, box)
        t = cv2.copyMakeBorder(t, 26, 4, 3, 3, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(t, lab, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 235, 255), 1)
        tiles.append(t)
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(20, 20, 20))
             for t in tiles]
    return np.hstack(tiles)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="silent_human_5min.mp4")
    ap.add_argument("--timeline", default="human_behavior_timeline.csv")
    ap.add_argument("--event", default=None,
                    help="event kind to sample around, e.g. blink / attention")
    ap.add_argument("--n", type=int, default=3, help="how many instances")
    ap.add_argument("--frames", type=int, default=7)
    ap.add_argument("--lead", type=float, default=0.07, help="seconds before")
    ap.add_argument("--zoom", type=float, default=2.2)
    ap.add_argument("--contact-sheet", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[inspect] {args.video}: {total} frames at {fps:.2f} fps "
          f"({total / fps / 60:.2f} min)")

    if args.contact_sheet:
        rows = []
        n = 12
        for i in range(n):
            t = (i + 0.5) * (total / fps) / n
            fr = frames_at(cap, fps, t, 1)
            if fr:
                tile = cv2.resize(fr[0][1], (0, 0), fx=0.30, fy=0.30)
                tile = cv2.copyMakeBorder(tile, 24, 4, 3, 3,
                                          cv2.BORDER_CONSTANT, value=(20, 20, 20))
                cv2.putText(tile, f"t={t:6.1f}s", (6, 17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 235, 255), 1)
                rows.append(tile)
        grid = np.vstack([np.hstack(rows[i:i + 4]) for i in range(0, len(rows), 4)])
        out = args.out or "clip_contact_sheet.png"
        cv2.imwrite(out, grid)
        print(f"[inspect] contact sheet -> {out}")
        return 0

    events = load_events(args.timeline)
    picked = [e for e in events if e["kind"] == args.event]
    if not picked:
        kinds = sorted({e["kind"] for e in events})
        raise SystemExit(f"no {args.event!r} events; have {kinds}")

    # Spread the samples across the clip rather than taking the first n, so a
    # fault that only appears late is not missed.
    step = max(len(picked) // args.n, 1)
    chosen = picked[::step][:args.n]

    rows = []
    for e in chosen:
        t = float(e["t_seconds"]) - args.lead
        frames = frames_at(cap, fps, t, args.frames)
        labels = [f"+{(i / fps - args.lead) * 1000:+.0f}ms" for i in range(len(frames))]
        labels[0] = f"t={float(e['t_seconds']):.2f}s {e['kind']}"
        row = strip(frames, labels, args.zoom, (0.38, 0.10, 0.66, 0.42))
        if row is not None:
            rows.append(row)

    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(20, 20, 20))
            for r in rows]
    out = args.out or f"clip_{args.event}.png"
    cv2.imwrite(out, np.vstack(rows))
    print(f"[inspect] {len(rows)} x {args.frames} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
