"""Measure what the camera side needs to know before wiring the face in.

Three questions, asked because they were asked of me and because guessing at
any of them would waste someone else's afternoon:

* how much VRAM does the face pipeline hold, and how much does it need free;
* how fast is it, and can it run alongside an EEVEE render at ~9 fps or does it
  need throttling;
* are the registration anchors stable enough to composite against, or do they
  swim under expression and pose.

The third is the one that decides whether this is usable at all. A face that
lands two pixels off from frame to frame will read as jitter no matter how good
the render is, so the anchors are measured across a real behaviour sequence
rather than a still.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def vram() -> tuple[float, float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 2**30, total / 2**30
    except Exception:
        return 0.0, 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_v04_final.png")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out", default="face_cutout_sheet.png")
    args = ap.parse_args()

    from presenter.behavior.engine import BehaviorEngine
    from presenter.render.face_source import ANCHOR_NAMES, FaceSource

    used0, total = vram()
    print(f"[face] GPU total {total:.2f} GiB, in use before load {used0:.2f} GiB")

    t0 = time.perf_counter()
    src = FaceSource(args.source)
    load_s = time.perf_counter() - t0
    used1, _ = vram()
    print(f"[face] load {load_s:.1f}s, VRAM after load {used1:.2f} GiB "
          f"(+{used1 - used0:.2f})")

    engine = BehaviorEngine(seed=args.seed)
    dt = 1.0 / 30.0
    times, confs, sizes = [], [], []
    anchors = {k: [] for k in ANCHOR_NAMES}
    peak = used1
    tiles = []

    for i in range(args.frames):
        engine.update(dt)
        f = src.frame(engine.pose)
        times.append(f.render_ms)
        confs.append(f.confidence)
        sizes.append(f.rgba.shape[:2])
        for k in ANCHOR_NAMES:
            # In plate coordinates, so the numbers describe where the face
            # actually is rather than where the crop happened to start.
            ax, ay = f.anchors[k]
            anchors[k].append((ax + f.origin[0], ay + f.origin[1]))
        u, _ = vram()
        peak = max(peak, u)
        if i % max(args.frames // 6, 1) == 0 and len(tiles) < 6:
            tiles.append(f)

    ms = statistics.fmean(times)
    print(f"\n[face] throughput over {args.frames} frames")
    print(f"[face]   mean {ms:6.1f} ms  = {1000 / ms:5.2f} fps alone")
    print(f"[face]   p95  {sorted(times)[int(0.95 * len(times))]:6.1f} ms   "
          f"worst {max(times):6.1f} ms")
    print(f"[face]   VRAM peak {peak:.2f} GiB of {total:.2f} "
          f"({total - peak:.2f} free for EEVEE)")
    eevee = 1000.0 / 9.0
    print(f"[face]   serial with EEVEE at 9 fps: "
          f"{1000 / (ms + eevee):.2f} fps combined")

    print(f"\n[face] cutout size  "
          f"{statistics.fmean(h for h, w in sizes):.0f} x "
          f"{statistics.fmean(w for h, w in sizes):.0f} px  "
          f"(varies {min(sizes)} to {max(sizes)})")
    print(f"[face] confidence   mean {statistics.fmean(confs):.2f}  "
          f"min {min(confs):.2f}")

    print(f"\n[face] anchor stability, frame to frame, in plate pixels")
    print(f"[face]   {'anchor':<12}{'median step':>12}{'p95 step':>10}{'max':>8}")
    for k in ANCHOR_NAMES:
        pts = np.asarray(anchors[k])
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        print(f"[face]   {k:<12}{np.median(step):>12.2f}"
              f"{np.quantile(step, 0.95):>10.2f}{step.max():>8.2f}")

    # Interocular distance is the scale the composite is registered at; if it
    # breathes, the face will pulse.
    el = np.asarray(anchors["eye_l"])
    er = np.asarray(anchors["eye_r"])
    iod = np.linalg.norm(el - er, axis=1)
    print(f"\n[face] interocular  mean {iod.mean():.2f} px  "
          f"sd {iod.std():.2f}  range {iod.max() - iod.min():.2f}")

    if tiles:
        import cv2
        cells = []
        for f in tiles:
            c = f.rgba
            checker = np.zeros((*c.shape[:2], 3), np.uint8)
            checker[::16, :] = 60
            checker[:, ::16] = 60
            checker[:] = np.maximum(checker, 25)
            a = (c[:, :, 3:4].astype(np.float32) / 255.0)
            comp = (c[:, :, :3] * a + checker * (1 - a)).astype(np.uint8)
            comp = cv2.resize(comp, (240, int(240 * comp.shape[0] / comp.shape[1])))
            cells.append(cv2.copyMakeBorder(comp, 2, 2, 2, 2,
                                            cv2.BORDER_CONSTANT, value=(0, 200, 255)))
        h = max(c.shape[0] for c in cells)
        cells = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 0,
                                    cv2.BORDER_CONSTANT, value=(20, 20, 20))
                 for c in cells]
        cv2.imwrite(args.out, np.hstack(cells))
        print(f"[face] cutouts over a checker -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
