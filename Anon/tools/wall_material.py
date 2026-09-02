"""Deterministic walnut-slat material swap on the locked master. No diffusion.

The generation phase is over. The master already contains geometry, perspective,
lighting, occlusion and composition; none of that needs rediscovering. Only the
**material** on one surface changes.

    GEOMETRY   = locked
    MATERIAL   = replaceable

## Why generation failed and this cannot

img2img needs contiguous surface to establish a repeating architectural pattern.
The exposed wall here is ~17.5% of frame in disconnected fragments, so any
strength weak enough to preserve the approved geometry was too weak to impose
slats, and any strength strong enough to impose slats moved the geometry.

A compositor does not have that problem: the pattern is *authored*, not
inferred, so fragment size is irrelevant.

## One global wall coordinate system

The single most important rule here. The visible wall is several disconnected
patches around monitors, the chair and the person. Texturing each patch
independently would give each its own slat phase and spacing - the classic
give-away.

Instead the material is generated **once across the full frame width** and every
fragment samples from that same projected surface. A batten passing behind a
monitor emerges on the far side in the correct place, because it was never
computed per-fragment. The wall in this master is essentially fronto-parallel
(the monitors hang flat on it), so vertical battens project as vertical lines
of constant pitch - a homography is available in `--perspective` but the
identity case is the physically correct one here.

## Frequency separation is what makes it sit in the plate

A texture pasted at full strength looks pasted, because it discards the light.
So the composite is split:

    LOW frequency  <- the ORIGINAL wall: illumination, falloff, shadows cast by
                      the chair and gear, monitor spill, vignette
    HIGH frequency <- the NEW material: batten edges, felt recesses, wood grain

The illumination field is extrapolated *behind* occluders using normalised
blurring, so the light continues smoothly across regions where the wall is not
visible. Objects therefore keep their contact shadows and do not detach from the
wall.

## Byte lock

Only masked pixels are written. The assertion at the end is exact equality
outside the mask, not approximate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Walnut, deep and neutral. Explicitly not orange or red - that reads rustic,
# which is the look being replaced.
WALNUT_GRADES = {
    "DARK_NEUTRAL": (62, 74, 92),      # BGR, mid-tone reference
    "DARK_WARM": (58, 76, 104),
    "VERY_DARK_WALNUT": (46, 56, 70),
}
FELT_BGR = (26, 27, 30)                # charcoal, never pure black


def wood_grain(h: int, w: int, seed: int) -> np.ndarray:
    """Multi-octave vertical grain in [0,1].

    Wood grain runs along the batten, so the noise is stretched hard vertically
    and kept fine horizontally. Several octaves because a single frequency reads
    as corduroy rather than timber.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp = 1.0
    total = 0.0
    for octave in range(4):
        sh = max(int(h / (18 * (octave + 1))), 2)
        sw = max(int(w / (1.4 ** octave)), 2)
        n = rng.random((sh, sw)).astype(np.float32)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        acc += n * amp
        total += amp
        amp *= 0.55
    acc /= max(total, 1e-6)
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=0.6, sigmaY=6.0)
    return np.clip((acc - acc.min()) / (np.ptp(acc) + 1e-6), 0, 1)


def build_material(h: int, w: int, pitch: int, gap_frac: float,
                   grade: str, seed: int = 11, ss: int = 3) -> np.ndarray:
    """Render the slat material across the full frame, supersampled.

    Supersampled then area-downsampled so the batten edges land anti-aliased.
    Drawing at final resolution produces stair-stepped edges that alias badly
    under video compression - the brief explicitly warns about shimmer.
    """
    H, W = h * ss, w * ss
    P = max(pitch * ss, 6)
    gap = max(int(P * gap_frac), 2)
    batten = P - gap

    base = np.array(WALNUT_GRADES[grade], np.float32)
    img = np.zeros((H, W, 3), np.float32)
    img[:, :] = np.array(FELT_BGR, np.float32)

    grain = wood_grain(H, W, seed)
    rng = np.random.default_rng(seed + 1)

    # Per-batten variation: each plank is a different piece of timber. Without
    # this the wall reads as one repeated 30px strip.
    x = 0
    while x < W:
        x1 = min(x + batten, W)
        if x1 <= x:
            break
        tone = float(rng.normal(1.0, 0.055))            # plank-to-plank colour
        shift = int(rng.integers(0, max(H // 3, 1)))     # different grain crop
        g = np.roll(grain[:, x:x1], shift, axis=0)

        strip = base[None, None, :] * tone * (0.86 + 0.28 * g[..., None])

        # Slat depth: lit edge catches a highlight, opposite edge falls into
        # shadow, and the felt recess gets ambient occlusion. Subtle - the room
        # lighting stays authoritative.
        ww = x1 - x
        if ww > 3:
            ramp = np.linspace(0.0, 1.0, ww, dtype=np.float32)[None, :, None]
            strip *= (1.0 + 0.16 * np.cos(ramp * np.pi))     # cylindrical shade
            edge = max(int(ww * 0.16), 1)
            strip[:, :edge] *= 1.14                          # lit edge
            strip[:, -edge:] *= 0.80                         # shadowed edge
        img[:, x:x1] = strip

        # Ambient occlusion in the felt gap either side of the batten.
        ao = max(int(gap * 0.5), 1)
        if x - ao >= 0:
            img[:, x - ao:x] *= 0.78
        if x1 + ao <= W:
            img[:, x1:x1 + ao] *= 0.78
        x += P

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0, 255)


def illumination_field(img_bgr: np.ndarray, mask: np.ndarray,
                       sigma: float) -> np.ndarray:
    """Low-frequency light on the wall, extrapolated behind occluders.

    Normalised blurring: blur(image x mask) / blur(mask). Where the wall is
    hidden the estimate is carried in from the nearest visible wall, so the
    field stays smooth across the person, chair and monitors instead of
    collapsing to black at their edges. That is what preserves cast shadows and
    keeps objects attached to the wall.
    """
    m = (mask > 0.35).astype(np.float32)
    lum = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    num = cv2.GaussianBlur(lum * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    field = num / np.maximum(den, 1e-4)
    # Second, wider pass to remove residual structure from the mask shape.
    return cv2.GaussianBlur(field, (0, 0), sigma * 0.6)


def measure_blur_sigma(img_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Rough defocus of the wall region, so the material can match it.

    Crisp wood grain behind a softly-focused room exposes a composite instantly.
    """
    sel = mask > 0.5
    if sel.sum() < 500:
        return 1.0
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    v = float(lap[sel].var())
    # Empirical: sharp plate ~>200, soft ~<40. Map to a plausible blur sigma.
    return float(np.clip(2.4 - 0.010 * v, 0.5, 2.6))


def match_grain(result: np.ndarray, original: np.ndarray,
                mask: np.ndarray) -> tuple[np.ndarray, str]:
    """Match the composited region's noise to the surrounding plate."""
    inside = mask > 0.5
    ring = cv2.dilate(inside.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    ring &= ~(cv2.dilate(inside.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    if inside.sum() < 100 or ring.sum() < 100:
        return result, "grain: skipped"

    def hf(img, sel):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        d = g - cv2.GaussianBlur(g, (0, 0), 1.6)
        return float(d[sel].std())

    target, have = hf(original, ring), hf(result, inside)
    if have >= target:
        return result, f"grain: plate {target:.2f}, wall {have:.2f}, no add"
    add = float(np.sqrt(max(target ** 2 - have ** 2, 0.0)))
    noise = np.random.default_rng(3).normal(0, add, result.shape[:2]).astype(np.float32)
    out = np.clip(result.astype(np.float32) + noise[..., None] * mask[..., None], 0, 255)
    return out.astype(np.uint8), (f"grain: plate {target:.2f}, wall {have:.2f}"
                                  f" -> +{add:.2f}")


def composite(master: np.ndarray, mask: np.ndarray, pitch: int,
              gap_frac: float, grade: str,
              darkness: float = 1.0) -> tuple[np.ndarray, list[str]]:
    h, w = master.shape[:2]
    log = []

    material = build_material(h, w, pitch, gap_frac, grade)

    # Match the plate's depth of field before anything else.
    sigma = measure_blur_sigma(master, mask)
    material = cv2.GaussianBlur(material, (0, 0), sigma)
    log.append(f"defocus: sigma {sigma:.2f}")

    # Frequency separation. Detail is the material's deviation from its own
    # local mean, applied multiplicatively onto the original illumination.
    mat_lum = cv2.cvtColor(material.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mat_lf = cv2.GaussianBlur(mat_lum, (0, 0), 24.0)
    detail = mat_lum / np.maximum(mat_lf, 1e-3)

    light = illumination_field(master, mask, sigma=42.0)

    # Normalise the illumination to RELATIVE falloff.
    #
    # The first attempt transferred the field's absolute luminance, which was
    # wrong in an obvious way once rendered: the plate is a pale grey wall at
    # ~128 luma, so the walnut inherited that brightness and came back a pale
    # peach. Dark walnut is a dark surface - what should carry over from the
    # plate is *where* the light falls and how it decays, not how bright the old
    # material happened to be.
    #
    # So divide the field by its own mean over the wall. What remains is a
    # gradient around 1.0 that still contains the vignette, the monitor spill
    # and the shadows cast by the chair and gear, applied on top of the
    # material's own dark base luminance.
    sel = mask > 0.5
    ref = float(light[sel].mean()) if sel.any() else 128.0
    rel = light / max(ref, 1e-3)
    rel = np.clip(rel, 0.45, 1.9)
    log.append(f"illumination: plate {ref:.1f} luma -> relative field "
               f"{rel[sel].min():.2f}-{rel[sel].max():.2f}")

    # Chroma and base level from the material; gradient from the plate.
    mat_chroma = material / np.maximum(mat_lum[..., None], 1e-3)
    target_lum = mat_lum * detail * rel * darkness
    wall = mat_chroma * target_lum[..., None]
    wall = np.clip(wall, 0, 255)
    log.append(f"wall luma {float(np.clip(target_lum,0,255)[sel].mean()):.1f}")

    m3 = mask[..., None]
    out = wall * m3 + master.astype(np.float32) * (1.0 - m3)
    out = np.clip(out, 0, 255).astype(np.uint8)

    out, gl = match_grain(out, master, mask)
    log.append(gl)

    # Byte lock: outside the mask the plate must be untouched, exactly.
    outside = mask <= 0.0
    if outside.any():
        d = int(np.abs(out[outside].astype(int) - master[outside].astype(int)).max())
        log.append(f"byte lock outside mask: max diff {d}")
        out[outside] = master[outside]
    return out, log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_locked_original.png")
    ap.add_argument("--mask", default="02_wall_mask.png")
    ap.add_argument("--grade", default="DARK_NEUTRAL", choices=sorted(WALNUT_GRADES))
    ap.add_argument("--gap-frac", type=float, default=0.34)
    ap.add_argument("--darkness", type=float, default=1.0,
                    help="scales the walnut base luminance; <1 darker")
    ap.add_argument("--out-dir", default="wall_scales")
    args = ap.parse_args()

    master = cv2.imread(args.source)
    mask_u8 = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if master is None or mask_u8 is None:
        raise FileNotFoundError("source or mask missing")
    mask = mask_u8.astype(np.float32) / 255.0

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Path("assets/materials").mkdir(parents=True, exist_ok=True)

    # A wider, B medium, C narrower. Everything else identical.
    scales = {"A": 22, "B": 16, "C": 11}
    results = {}
    for name, pitch in scales.items():
        img, log = composite(master, mask, pitch, args.gap_frac,
                             args.grade, args.darkness)
        p = out_dir / f"wall_scale_{name}.png"
        cv2.imwrite(str(p), img)
        results[name] = img
        print(f"[mat] scale {name}: pitch {pitch}px  ({', '.join(log)})")
        print(f"[mat]   -> {p}")

    # Save the raw material swatch as a reusable asset.
    swatch = build_material(512, 512, scales["B"], args.gap_frac, args.grade)
    cv2.imwrite("assets/materials/dark_walnut_acoustic_slats.png",
                swatch.astype(np.uint8))
    print("[mat] swatch -> assets/materials/dark_walnut_acoustic_slats.png")

    # Comparison at full frame.
    cw = 900; ch = int(cw * master.shape[0] / master.shape[1])
    tiles = [("ORIGINAL", master)] + [(f"SCALE {k} (pitch {scales[k]}px)", v)
                                      for k, v in results.items()]
    shown = []
    for lab, t in tiles:
        r = cv2.resize(t, (cw, ch))
        cv2.putText(r, lab, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                    (0, 255, 255), 3)
        shown.append(r)
    rows = [np.hstack(shown[i:i + 2]) for i in range(0, len(shown), 2)]
    cv2.imwrite("wall_scale_compare.png", np.vstack(rows))
    print("[mat] comparison -> wall_scale_compare.png")

    # Slat-continuity debug: material everywhere the wall plane exists, with
    # occluders greyed, to confirm battens line up across fragments.
    dbg = build_material(master.shape[0], master.shape[1], scales["B"],
                         args.gap_frac, args.grade).astype(np.uint8)
    grey = (master.astype(np.float32) * 0.25).astype(np.uint8)
    m3 = mask[..., None]
    cv2.imwrite("wall_continuity_debug.png",
                np.clip(dbg * m3 + grey * (1 - m3), 0, 255).astype(np.uint8))
    print("[mat] continuity debug -> wall_continuity_debug.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
