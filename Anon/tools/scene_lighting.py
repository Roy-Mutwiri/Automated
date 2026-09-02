"""Phase 2: add restrained architectural practical lighting to the background.

Deterministic compositing. No diffusion, no generative relight, no global
exposure change. The human is protected by an explicit mask and is not touched.

## The physical model

Each practical is a real emitter with an origin, a direction, a falloff and an
occlusion test - not an orange gradient painted onto the plate.

    reflected = E(x,y) * albedo(x,y) * colour

`E` is irradiance: intensity x distance falloff x directional lobe x shadow
transmittance. It is added in **linear light** (gamma-decoded), because that is
what adding a lamp to a room actually does. Adding in sRGB space lifts blacks
far too much and is the usual reason composited light looks like a wash.

### Falloff
Inverse-square-ish, `1 / (1 + (d/r)^2)`, so the surface immediately in front of
the emitter is brightest and illumination decreases with distance. Line
emitters (an LED strip under a shelf) use distance-to-segment, which is why a
strip produces a band rather than a hot spot.

### Directional lobe
`clip(dot(unit(p - o), direction), 0, 1) ** power`. This is what makes the
emitter *hidden*: a strip under a shelf pointing down puts no light on the
shelf face above it, so the viewer sees the glow and not the LED.

### Occlusion
Shadows are cast for real. The occluder mask is resampled into polar
coordinates around the emitter and transmittance falls as
`exp(-k * cumulative occluder along the ray)`. Light therefore cannot pass
through the chair, the monitors, the speaker or the person - it stops at their
silhouette and leaves a shadow behind them.

### Multiplicative, not additive - and why the felt stays black
The light is applied as `lin_out = lin * (1 + E * colour)`, not `lin + E`.

This is the single most important decision in the file. Irradiance does not add
to a surface's radiance, it *scales* it: reflected = E x albedo, so under a
brighter room every surface gains in proportion to its own reflectance. A first
attempt added light in linear space and the felt gaps went 27.7 -> 49.4 luma,
converging on the wood and turning the wall into grey stripes - the exact
failure section 8 forbids. Adding a constant in linear space nearly doubles a
near-black pixel while barely touching a mid-tone.

Multiplying preserves the batten:felt ratio exactly, whatever the intensity.
The gaps can never catch the wood up. It also *reveals* grain rather than
drawing it: the absolute gap between light and dark grain widens with E while
the ratio holds, which is what a real lamp does to a textured surface, and what
section 7 asks for.

A small additive term (`AMBIENT_ADD`) remains, because a genuinely black
surface under a practical is not perfectly black. It is deliberately tiny.

## What is allowed to change

`05_lighting_mask.png` - the wall mask plus the under-shelf zone, minus the
dilated human. Outside it the output is copied from the input, so the byte lock
is structural.

Usage
-----
    python tools/scene_lighting.py --variants A B C
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# --- Scene geometry, read off the locked master (1344 x 768) ----------------
#
# Hard-coded on purpose, as with the wall mask: this runs once against one
# frozen plate. Values are checked visually via the preview, not guessed.

# Under the walnut shelf on the right. The shelf's front edge runs from about
# (1120, 572) to (1344, 590); the AV gear sits directly beneath it.
SHELF_P0 = (1120.0, 594.0)
SHELF_P1 = (1344.0, 610.0)
UNDER_SHELF_ZONE = (1105, 590, 1344, 706)      # x0, y0, x1, y1

# Solid objects that stop light.
OCCLUDER_RECTS = [
    (0, 10, 385, 208),        # left monitor
    (418, 24, 888, 302),      # centre monitor
    (1040, 8, 1152, 474),     # speaker / rack column
    (1168, 20, 1330, 304),    # right monitor
    (884, 190, 1132, 768),    # chair
    (0, 330, 500, 768),       # desk gear cluster, foreground left
]

TUNGSTEN_3000K = np.array([107.0, 180.0, 255.0]) / 255.0   # BGR, sRGB primaries

# --- Lights -----------------------------------------------------------------
#
# Two. Not one per shelf, not a strip around the room. This room only offers
# two surfaces that can plausibly receive an architectural practical without
# touching the subject, the chair or a screen; see the note at the end of the
# list for the third that was designed, measured and removed.

LIGHTS = [
    # A vertical wash grazing the walnut from off-frame left. This is the main
    # architectural gesture: it gives the left third of the room a light
    # direction and reveals the batten edges. Far from the face by design.
    dict(name="left_wall_wash", kind="point",
         origin=(-70.0, 250.0), radius=430.0,
         direction=(1.0, 0.22), lobe_power=1.1,
         intensity=0.34),

    # Hidden warm LED under the walnut shelf, pointing down onto the AV gear.
    # The emitter itself is never visible - only what it lands on.
    dict(name="under_shelf_led", kind="line",
         p0=SHELF_P0, p1=SHELF_P1, radius=105.0,
         direction=(0.05, 1.0), lobe_power=1.9,
         intensity=0.50),

    # A third practical was designed for the walnut gap between the centre
    # monitor and the chair, and cut after measuring it: mean delta 0.00. The
    # chair fills that gap almost entirely, so the only surface available to
    # receive the light was the chair itself - which section 9 puts off limits.
    # Two motivated sources beat three where one lands on nothing.
]

VARIANTS = {
    "A": dict(scale=0.55, warm=0.48, haze=0.06),   # very subtle
    "B": dict(scale=1.00, warm=0.60, haze=0.10),   # premium balanced
    "C": dict(scale=1.55, warm=0.68, haze=0.15),   # maximum acceptable
}

# A black surface under a practical is not perfectly black. Tiny on purpose:
# this is the only term that can lift the felt independently of the wood, so it
# is the only term that can turn the gaps grey. At 0.05 it did exactly that -
# felt +27% against battens +6% - because 0.05 x E is larger than the felt's own
# linear value. At 0.0025 both move together and the ratio survives.
AMBIENT_ADD = 0.0025


# --- Fields -----------------------------------------------------------------

def _grid(h: int, w: int):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs, ys


def _dist_point(xs, ys, origin):
    return np.hypot(xs - origin[0], ys - origin[1]), (xs - origin[0], ys - origin[1])


def _dist_segment(xs, ys, p0, p1):
    """Distance to a line segment, plus the vector from the closest point."""
    ax, ay = p0
    bx, by = p1
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / max(L2, 1e-6), 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    vx, vy = xs - cx, ys - cy
    return np.hypot(vx, vy), (vx, vy)


def transmittance(occluder: np.ndarray, origin, k: float = 5.5) -> np.ndarray:
    """Shadow field: how much light survives the trip from `origin` to each pixel.

    Computed in polar coordinates around the emitter, so the attenuation
    accumulates *along the ray*. An object therefore darkens everything behind
    it and nothing in front of it - a real shadow with the occluder's own
    silhouette, rather than a soft blob.
    """
    h, w = occluder.shape
    max_r = float(np.hypot(max(origin[0], w - origin[0]),
                           max(origin[1], h - origin[1]))) + 8.0
    size = (768, 1024)                                   # radii, angles
    pol = cv2.warpPolar(occluder, size, (float(origin[0]), float(origin[1])),
                        max_r, cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
    step = max_r / size[0]
    acc = np.cumsum(pol, axis=1) * (step / max_r)
    tr_pol = np.exp(-k * acc).astype(np.float32)
    tr = cv2.warpPolar(tr_pol, (w, h), (float(origin[0]), float(origin[1])),
                       max_r,
                       cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)
    return cv2.GaussianBlur(np.clip(tr, 0.0, 1.0), (0, 0), 3.0)


def irradiance(light: dict, shape, occluder: np.ndarray) -> np.ndarray:
    h, w = shape
    xs, ys = _grid(h, w)
    if light["kind"] == "line":
        d, (vx, vy) = _dist_segment(xs, ys, light["p0"], light["p1"])
        origin = ((light["p0"][0] + light["p1"][0]) * 0.5,
                  (light["p0"][1] + light["p1"][1]) * 0.5)
    else:
        d, (vx, vy) = _dist_point(xs, ys, light["origin"])
        origin = light["origin"]

    fall = 1.0 / (1.0 + (d / light["radius"]) ** 2)

    dx, dy = light["direction"]
    n = np.hypot(dx, dy) or 1.0
    dx, dy = dx / n, dy / n
    vn = np.maximum(np.hypot(vx, vy), 1e-3)
    lobe = np.clip((vx * dx + vy * dy) / vn, 0.0, 1.0) ** light["lobe_power"]

    tr = transmittance(occluder, origin)
    return (light["intensity"] * fall * lobe * tr).astype(np.float32)


# --- Masks ------------------------------------------------------------------

def human_mask(img: np.ndarray, grow: float = 0.026) -> np.ndarray:
    import torch
    import torchvision

    w8 = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights=w8).eval().cuda()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    batch = w8.transforms()(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
        p = model(batch)["out"][0].softmax(0)[15].cpu().numpy()
    del model
    torch.cuda.empty_cache()
    h, w = img.shape[:2]
    p = cv2.resize(p.astype(np.float32), (w, h))
    g = max(int(grow * min(h, w)) | 1, 9)
    hard = cv2.dilate((p > 0.4).astype(np.uint8), np.ones((g, g), np.uint8))
    return cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 5.0)


def build_masks(img: np.ndarray, wall_mask: np.ndarray):
    h, w = img.shape[:2]
    protect = human_mask(img)

    allow = wall_mask.copy()
    x0, y0, x1, y1 = UNDER_SHELF_ZONE
    zone = np.zeros((h, w), np.float32)
    zone[y0:y1, x0:x1] = 1.0
    zone = cv2.GaussianBlur(zone, (0, 0), 9.0)
    allow = np.maximum(allow, zone)
    allow *= (1.0 - protect)                       # never touch the human
    allow = np.clip(allow, 0.0, 1.0)

    occl = np.zeros((h, w), np.float32)
    for (a, b, c, d) in OCCLUDER_RECTS:
        occl[b:d, a:c] = 1.0
    occl = np.maximum(occl, (protect > 0.35).astype(np.float32))
    occl = cv2.GaussianBlur(occl, (0, 0), 2.0)
    return allow, protect, occl


# --- Composite --------------------------------------------------------------

def luma(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def relight(img: np.ndarray, allow: np.ndarray, occl: np.ndarray,
            scale: float, warm: float, haze: float):
    h, w = img.shape[:2]

    E = np.zeros((h, w), np.float32)
    per_light = {}
    for lg in LIGHTS:
        e = irradiance(lg, (h, w), occl) * scale
        per_light[lg["name"]] = float((e * (allow > 0.5)).max())
        E += e
    if haze > 0:
        E = E + cv2.GaussianBlur(E, (0, 0), 42.0) * haze

    colour = (1.0 - warm) + warm * TUNGSTEN_3000K       # lerp white -> tungsten
    colour = colour / colour.max()

    lin = np.power(img.astype(np.float32) / 255.0, 2.2)
    gain = 1.0 + (E * allow)[..., None] * colour[None, None, :]
    amb = AMBIENT_ADD * (E * allow)[..., None] * colour[None, None, :]
    out = np.power(np.clip(lin * gain + amb, 0.0, 1.0), 1.0 / 2.2) * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Structural byte lock: anything the mask does not reach comes back
    # unmodified from the plate.
    outside = allow <= 0.0
    out[outside] = img[outside]
    return out, per_light, E


def stats(before: np.ndarray, after: np.ndarray, wall: np.ndarray) -> dict:
    sel = wall > 0.5
    lb, la = luma(before), luma(after)
    if not sel.any():
        return {}
    wb, wa = lb[sel], la[sel]
    # Split wall pixels into battens and felt by their own median, so the felt
    # separation can be reported rather than assumed.
    med = np.median(wb)
    felt, batten = wb < med, wb >= med
    return dict(
        wall_before=float(wb.mean()), wall_after=float(wa.mean()),
        batten_before=float(wb[batten].mean()), batten_after=float(wa[batten].mean()),
        felt_before=float(wb[felt].mean()), felt_after=float(wa[felt].mean()),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_v02_wall.png")
    ap.add_argument("--wall-mask", default="02_wall_mask.png")
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--mask-out", default="05_lighting_mask.png")
    ap.add_argument("--protect-out", default="human_protection_mask.png")
    args = ap.parse_args()

    img = cv2.imread(args.source)
    wall = cv2.imread(args.wall_mask, cv2.IMREAD_GRAYSCALE)
    if img is None or wall is None:
        raise FileNotFoundError("source or wall mask missing")
    wall = wall.astype(np.float32) / 255.0

    allow, protect, occl = build_masks(img, wall)
    cv2.imwrite(args.mask_out, (allow * 255).astype(np.uint8))
    cv2.imwrite(args.protect_out, (protect * 255).astype(np.uint8))
    print(f"[light] lighting mask {100 * (allow > 0.5).mean():.1f}% of frame "
          f"-> {args.mask_out}")
    print(f"[light] human protection mask -> {args.protect_out}")

    report = {}
    outs = {}
    for v in args.variants:
        p = VARIANTS[v]
        out, per_light, E = relight(img, allow, occl, **p)
        path = f"lighting_{v}.png"
        cv2.imwrite(path, out)
        outs[v] = out

        outside = allow <= 0.0
        d = int(np.abs(out[outside].astype(int) - img[outside].astype(int)).max())
        s = stats(img, out, wall)
        report[v] = dict(params=p, peak_irradiance=per_light, byte_lock=d, **s)
        print(f"[light] {v}: scale {p['scale']:.2f} warm {p['warm']:.2f}  "
              f"wall luma {s['wall_before']:.1f} -> {s['wall_after']:.1f}  "
              f"batten {s['batten_before']:.1f} -> {s['batten_after']:.1f}  "
              f"felt {s['felt_before']:.1f} -> {s['felt_after']:.1f}  "
              f"byte lock outside mask: max diff {d}")

    Path("lighting_report.json").write_text(json.dumps(report, indent=2))

    # Comparison sheet: plate, then each variant.
    tiles = [("v02 wall (no practicals)", img)] + \
            [(f"LIGHT_{v}", outs[v]) for v in args.variants]
    cw = 900
    ch = int(cw * img.shape[0] / img.shape[1])
    cells = []
    for lab, t in tiles:
        r = cv2.resize(t, (cw, ch))
        cv2.rectangle(r, (0, 0), (cw, 40), (16, 16, 16), -1)
        cv2.putText(r, lab, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 235, 255), 2)
        cells.append(r)
    rows = [np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)]
    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(18, 18, 18)) for r in rows]
    cv2.imwrite("lighting_compare.png", np.vstack(rows))
    print("[light] comparison -> lighting_compare.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
