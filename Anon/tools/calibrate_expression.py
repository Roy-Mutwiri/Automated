"""Measure what LivePortrait's expression dimensions actually do to this face.

`render/calibration.py` maps semantic channels - gaze, brows - onto indices in
LivePortrait's 21x3 expression latent. Every entry in that map is marked
`verified=False`, the gains are set deliberately low so a wrong guess is
ineffective rather than destructive, and the tool the docstring tells you to run
was never written. This is that tool.

The consequence of leaving it unwritten is not cosmetic. The gaze system
computes saccades, fixations and microsaccades to a standard the brief calls the
highest-priority realism target, and then writes them into channels whose gain
is 0.004 against an unverified index. If those indices are wrong, the eyes do
not move at all and no amount of behavioural sophistication upstream is visible.

## Method

The latent is undocumented and has no published semantic index map, so the only
honest way to read it is to drive it and look at the pixels.

For each channel the renderer is driven at a sweep of values and each frame is
differenced against the neutral render. Three numbers come out:

* **effect** - mean absolute change inside the region the channel is *supposed*
  to move (eye boxes for gaze, brow boxes for brows).
* **collateral** - the same measure over the rest of the face. A gaze channel
  that moves the jaw is wrong regardless of how well it moves the iris.
* **selectivity** - effect / collateral. This is the number that matters. A
  channel with high effect and high collateral is not a gaze control, it is a
  face wobble.

A contact sheet of eye and brow crops is written alongside, because these
numbers can only rank candidates - whether the iris is moving *the right way*
is a question for a person looking at the image.

Usage
-----
    python tools/calibrate_expression.py --channel gaze_x
    python tools/calibrate_expression.py --scan-indices --axis 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Regions of the *head crop*, in fractions of its width/height. The renderer's
# generated crop is 512x512 and roughly face-centred, so these are stable.
# Read off a gridded render of the actual 512x512 generated crop, not assumed.
# The first version of this table put "eyes" at y 0.30-0.46, which on this
# source is the brow and forehead - so the first index scan ranked candidates on
# a region that contains no eyes at all. Measuring the wrong box produces
# confident, precise, wrong numbers.
REGIONS = {
    "brows": (0.26, 0.36, 0.66, 0.44),
    "eyes":  (0.26, 0.42, 0.66, 0.53),
    "mouth": (0.36, 0.62, 0.60, 0.74),
    "jaw":   (0.28, 0.74, 0.68, 0.90),
}


def region_box(shape, key):
    h, w = shape[:2]
    x0, y0, x1, y1 = REGIONS[key]
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def region_mean(diff, shape, key):
    x0, y0, x1, y1 = region_box(shape, key)
    return float(diff[y0:y1, x0:x1].mean())


class Probe:
    """Renders the raw generated crop, before compositing into the room.

    Compositing would confine the visible change to the paste mask and add the
    plate's own pixels to every difference, both of which corrupt the
    measurement. What is being calibrated is the model's expression space, so
    the model's own output is what gets measured.
    """

    def __init__(self, source, root):
        import torch
        from presenter.render.liveportrait import LivePortraitRenderer

        self.torch = torch
        self.r = LivePortraitRenderer(
            source_image=source, liveportrait_root=root,
            output_size=(1920, 1080), framing="full", environment="source",
            neutralize_pose=0.0,
        )

    def render_crop(self, pose):
        import cv2 as _cv2
        with self.torch.no_grad():
            x_d = self.r._build_driving_keypoints(pose)
            x_d = self.r._apply_blink(x_d, pose)
            if self.r.cfg.flag_stitching:
                x_d = self.r.wrapper.stitching(self.r.x_s, x_d)
            out = self.r.wrapper.warp_decode(self.r.f_s, self.r.x_s, x_d)
            rgb = self.r.wrapper.parse_output(out["out"])[0]
        return _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)

    def render_raw_delta(self, index, axis, amount):
        """Drive one latent dimension directly, bypassing the semantic map."""
        from presenter.types import AvatarPose
        pose = AvatarPose()
        with self.torch.no_grad():
            saved = self.r.src_exp.clone()
            self.r.src_exp[0, index, axis] += amount
            try:
                frame = self.render_crop(pose)
            finally:
                self.r.src_exp = saved
        return frame


def sweep_channel(probe, channel_name, values, out_dir):
    from presenter.render.calibration import CHANNELS
    from presenter.types import AvatarPose

    ch = CHANNELS[channel_name]
    neutral = probe.render_crop(AvatarPose())
    ng = cv2.cvtColor(neutral, cv2.COLOR_BGR2GRAY).astype(np.float32)

    intended = "brows" if channel_name.startswith("brow") else (
        "mouth" if channel_name == "mouth_open" else "eyes")

    rows, tiles = [], []
    for v in values:
        pose = AvatarPose(**{channel_name: v})
        frame = probe.render_crop(pose)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(g - ng)

        eff = region_mean(diff, frame.shape, intended)
        coll = np.mean([region_mean(diff, frame.shape, k)
                        for k in REGIONS if k != intended])
        rows.append(dict(value=v, effect=eff, collateral=coll,
                         selectivity=eff / max(coll, 1e-4),
                         max_diff=float(diff.max())))

        x0, y0, x1, y1 = region_box(frame.shape, intended)
        crop = cv2.resize(frame[y0:y1, x0:x1], (0, 0), fx=2.0, fy=2.0,
                          interpolation=cv2.INTER_NEAREST)
        crop = cv2.copyMakeBorder(crop, 26, 4, 4, 4, cv2.BORDER_CONSTANT,
                                  value=(20, 20, 20))
        cv2.putText(crop, f"{v:+.2f}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 235, 255), 1)
        tiles.append(crop)

    print(f"\n[calib] {channel_name}  gain={ch.gain}  targets={ch.targets}")
    print(f"[calib] {'value':>8} {'effect':>9} {'collateral':>11} "
          f"{'selectivity':>12} {'maxdiff':>8}")
    for r in rows:
        print(f"[calib] {r['value']:>+8.2f} {r['effect']:>9.3f} "
              f"{r['collateral']:>11.3f} {r['selectivity']:>12.2f} "
              f"{r['max_diff']:>8.0f}")

    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(20, 20, 20))
             for t in tiles]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(out_dir) / f"sweep_{channel_name}.png"),
                np.hstack(tiles))
    return rows


def scan_indices(probe, axis, amount, out_dir):
    """Drive every latent dimension on one axis and see which region reacts.

    This is the measurement that decides whether the inherited index map is
    right. It makes no assumption about which keypoint is an eye - it drives all
    21 and reports where the face moved.
    """
    from presenter.types import AvatarPose

    neutral = probe.render_crop(AvatarPose())
    ng = cv2.cvtColor(neutral, cv2.COLOR_BGR2GRAY).astype(np.float32)

    print(f"\n[calib] index scan, axis {axis}, amount {amount:+.3f}")
    print(f"[calib] {'idx':>4} " + " ".join(f"{k:>8}" for k in REGIONS)
          + f" {'best':>8}")
    results = []
    for i in range(21):
        frame = probe.render_raw_delta(i, axis, amount)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(g - ng)
        vals = {k: region_mean(diff, frame.shape, k) for k in REGIONS}
        best = max(vals, key=vals.get)
        results.append(dict(index=i, axis=axis, **vals, best=best))
        print(f"[calib] {i:>4} " + " ".join(f"{vals[k]:>8.3f}" for k in REGIONS)
              + f" {best:>8}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"index_scan_axis{axis}.json").write_text(
        json.dumps(results, indent=2))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_v04_final.png")
    ap.add_argument("--root", default="third_party/LivePortrait")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--values", type=float, nargs="+",
                    default=[-1.0, -0.5, -0.25, 0.25, 0.5, 1.0])
    ap.add_argument("--scan-indices", action="store_true")
    ap.add_argument("--axis", type=int, default=0)
    ap.add_argument("--amount", type=float, default=0.02)
    ap.add_argument("--out-dir", default="calibration")
    args = ap.parse_args()

    probe = Probe(args.source, args.root)

    if args.scan_indices:
        scan_indices(probe, args.axis, args.amount, args.out_dir)
        return 0

    from presenter.render.calibration import CHANNELS
    names = [args.channel] if args.channel else list(CHANNELS)
    for n in names:
        sweep_channel(probe, n, args.values, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
