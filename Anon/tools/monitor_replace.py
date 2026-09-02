"""Phase 3: replace LCD content by perspective mapping. Deterministic, no diffusion.

Only the active LCD surface changes. Bezel, frame, stand, the room around it and
everything that occludes it come back byte-identical from the plate.

## The pipeline, and the reason for each step

1. **Homography.** The panel's four corners live in `config/monitor_geometry.json`
   in image coordinates. `cv2.getPerspectiveTransform` maps the source UI's
   corners onto them, so the content lies in the monitor's plane rather than
   being pasted flat. Corners may fall outside the frame - the right panel runs
   off the edge - because the quad has to describe the whole physical screen or
   the visible strip carries the wrong horizontal scale.

2. **Occlusion, from masks and never from the quad.** The subject's head and
   headphones cover a third of the centre panel. Two sources are unioned: the
   segmented human, dilated; and a darkness test inside the quad, because
   segmentation reliably under-covers black headphones and flyaway hair against
   a bright screen - exactly where a pasted UI would betray itself.

3. **Exposure match.** The new content is scaled to the *visible* original LCD's
   own luminance. Section 19 is right that this is critical: a screen that is
   brighter than the plate's exposure stops being a screen in a room and becomes
   a light box, and it would also start competing with the face for the eye.

4. **Defocus.** The panel sits behind the focal plane. The plate's own local
   blur is measured and applied, because a razor-sharp interface behind a
   softly-focused room is the single most obvious composite tell.

5. **Reflections restored.** See below.

6. **Grain match**, then composite under the occlusion mask.

## Preserving the panel's light, without preserving its picture

A monitor is lit by the room as well as by itself: there is a large-scale sheen
across the glass from viewing angle, veiling glare and the practicals. That
low-frequency envelope is what makes a screen sit *in* a room.

So the original LCD's luminance is separated by frequency. The **low-frequency
envelope**, normalised to its own mean, is reapplied multiplicatively to the new
content - the panel keeps its own light distribution while losing its picture
entirely. The high frequencies, which are the old image, are discarded.

Specular highlights get a second, deliberately conservative pass: a pixel is
treated as a reflection only if it is both bright and *achromatic*, since a
reflection of a room light has no colour of its own while screen content does.
The preserved fraction is measured and reported, so if it ever starts dragging
content across it is visible in the numbers rather than only in the picture.

Usage
-----
    python tools/monitor_replace.py
    python tools/monitor_replace.py --only centre --content assets/screens/other.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Fraction of the original LCD luminance the replacement aims for. Below 1.0 on
# purpose - see the exposure step for the argument.
LUMA_TARGET_FRAC = 0.75

# Ceiling on screen chroma, in OpenCV's 0-255 saturation units.
MAX_SCREEN_SAT = 96.0

# How far beyond the segmented human an occluder may reach, in plate pixels.
# Sized for headphone cups and hair, not for content on the panel behind him.
OCCLUDER_REACH_PX = 27


# --- Masks ------------------------------------------------------------------

def quad_mask(shape, quad) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.int32(quad), 255)
    return m.astype(np.float32) / 255.0


def occlusion_mask(img, quad_m, protect, dark_v: int = 62) -> np.ndarray:
    """What sits in front of this screen. Authoritative over the quad.

    The segmented human, plus dark pixels **connected to** the human. The
    darkness test exists because segmentation under-covers black headphones and
    flyaway hair against a lit panel - but it cannot stand alone. A first
    version took every dark pixel inside the quad, which promptly classified the
    dark regions of the *old screen content* as occluders and punched the
    previous picture's silhouette straight through the new interface.

    Connectivity is what separates the two cases. Headphones and hair touch the
    head; a dark patch in a photograph on the monitor touches nothing. So the
    dark mask is split into components and only those overlapping the segmented
    human survive. On a panel the subject does not overlap at all, the test
    correctly contributes nothing.
    """
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    dark = ((v < dark_v) & (quad_m > 0.5)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Two conditions, both required. Connectivity alone was still too
    # permissive: the old game content had a dark band along the bottom of the
    # panel that touched the subject's shoulder, so a single component flooded
    # from his head across the whole screen and left 62% of the old picture in
    # place. A proximity band alone would clip long flyaway hair. Together they
    # describe what actually occludes a screen - something attached to the
    # subject and close to him.
    seed = cv2.dilate((protect > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8))
    near = cv2.dilate(seed, np.ones((OCCLUDER_REACH_PX, OCCLUDER_REACH_PX), np.uint8))
    n, labels = cv2.connectedComponents(dark, 8)
    keep = np.zeros_like(dark)
    for i in range(1, n):
        comp = labels == i
        if (comp & (seed > 0)).any():
            keep[comp] = 1
    keep &= (near > 0)
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))

    occ = np.maximum(protect, keep.astype(np.float32))
    return np.clip(cv2.GaussianBlur(occ, (0, 0), 1.6), 0.0, 1.0)


# --- Plate measurements -----------------------------------------------------

def local_blur_sigma(img, sel) -> float:
    """Defocus of the plate in this region, so the UI can match it."""
    if sel.sum() < 400:
        return 1.2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    v = float(lap[sel].var())
    return float(np.clip(2.4 - 0.010 * v, 0.6, 2.6))


def grain_sigma(img, sel) -> float:
    if sel.sum() < 400:
        return 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float((g - cv2.GaussianBlur(g, (0, 0), 1.6))[sel].std())


def reflection_layers(img, sel):
    """Split the original panel into (light envelope, specular highlights).

    Envelope: low-frequency luminance, normalised to its own mean over the
    visible screen, so it is a pure multiplier with no absolute brightness of
    its own. Specular: bright *and* achromatic residue only.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Blurred hard and clamped tight. At sigma 26 with a +-45% range the
    # "envelope" still carried the old picture's composition - the right panel
    # showed a bright sky at the top and a dark foreground at the bottom, and the
    # new interface inherited that gradient wholesale. At sigma 70 only a genuine
    # across-the-glass falloff survives, and the clamp stops even that from
    # becoming a tonal signature.
    low = cv2.GaussianBlur(lum, (0, 0), 70.0)
    ref = float(low[sel].mean()) if sel.any() else 1.0
    env = np.clip(low / max(ref, 1e-3), 0.82, 1.22)

    resid = np.clip(lum - low, 0, None)
    achromatic = (hsv[:, :, 1].astype(np.float32) < 40)
    bright = lum > (np.percentile(lum[sel], 99.0) if sel.any() else 255)
    spec = np.where(achromatic & bright, resid, 0.0)
    spec = cv2.GaussianBlur(spec, (0, 0), 1.8)
    frac = float(((spec > 2.0) & sel).sum()) / max(int(sel.sum()), 1)
    return env.astype(np.float32), spec.astype(np.float32), frac


# --- The reusable operation -------------------------------------------------

def replace_monitor(frame: np.ndarray, mon: dict, source: np.ndarray,
                    protect: np.ndarray, verbose: bool = True):
    """Map `source` into the monitor described by `mon`, in place on a copy.

    Everything outside the screen mask is left exactly as it was found, so the
    call is safe to chain across several monitors.
    """
    h, w = frame.shape[:2]
    quad = np.float32(mon["quad"])

    qm = quad_mask(frame.shape, quad)
    occ = occlusion_mask(frame, qm, protect)
    screen = np.clip(qm * (1.0 - occ), 0.0, 1.0)
    sel = screen > 0.5
    if sel.sum() < 200:
        if verbose:
            print(f"[mon] {mon['id']}: nothing visible, skipped")
        return frame, {}

    sh, sw = source.shape[:2]
    src_corners = np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]])
    H = cv2.getPerspectiveTransform(src_corners, quad)
    warped = cv2.warpPerspective(source, H, (w, h), flags=cv2.INTER_AREA,
                                 borderMode=cv2.BORDER_REPLICATE).astype(np.float32)

    # 3. Exposure: lift the UI to the plate's own LCD luminance with a gamma
    #    curve, not a multiplier.
    #
    #    A linear gain of 3.2x hit the accents as hard as the charcoal and turned
    #    the muted cyan into a saturated blue panel - the exact failure section 20
    #    warns about. A gamma lift raises the dark base steeply and the bright
    #    accents barely at all, which is also how a display's own transfer curve
    #    behaves.
    #
    #    The target is 0.75x the original, not 1.0x. A charcoal dashboard on the
    #    same monitor at the same brightness genuinely photographs darker than a
    #    bright photograph does; matching the mean exactly would mean building a
    #    UI that is not dark. 0.75 keeps the panel unmistakably lit while letting
    #    it be the cool, quiet counterpoint the room needs.
    orig_lum = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tgt_mean = float(orig_lum[sel].mean()) * LUMA_TARGET_FRAC
    tgt_p95 = float(np.percentile(orig_lum[sel], 95))

    norm = np.clip(warped / 255.0, 0.0, 1.0)
    lum_n = cv2.cvtColor((norm * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)[sel] / 255.0
    lo, hi = 0.30, 1.20
    for _ in range(40):                       # bisection on the gamma
        g = 0.5 * (lo + hi)
        if float(np.power(lum_n, g).mean()) * 255.0 < tgt_mean:
            hi = g
        else:
            lo = g
    gain = 0.5 * (lo + hi)
    warped = np.power(norm, gain) * 255.0

    # 5. Reflections: keep the panel's light, drop its picture.
    env, spec, spec_frac = reflection_layers(frame, sel)
    warped *= env[..., None]
    warped += spec[..., None] * 0.85

    # 4. Defocus to the plate's own softness in this region.
    sigma = local_blur_sigma(frame, sel)
    warped = cv2.GaussianBlur(warped, (0, 0), sigma)

    # Screen bloom: a real panel glows very slightly past its own edges.
    bloom = cv2.GaussianBlur(np.clip(warped - 90.0, 0, None), (0, 0), 9.0)
    warped = warped + bloom * 0.16

    # 6. Grain to the plate's measured level. Measured, never assumed.
    have = grain_sigma(warped.astype(np.uint8), sel)
    want = grain_sigma(frame, sel)
    added = 0.0
    if want > have:
        added = float(np.sqrt(max(want ** 2 - have ** 2, 0.0)))
        rng = np.random.default_rng(int(abs(hash(mon["id"]))) % (2 ** 31))
        warped += rng.normal(0.0, added, (h, w)).astype(np.float32)[..., None]

    out = frame.astype(np.float32) * (1 - screen[..., None]) + \
        np.clip(warped, 0, 255) * screen[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Section 20: keep saturation low. The lift can only ever push chroma up, so
    # this is a ceiling applied inside the screen region and nowhere else.
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.where(sel, np.minimum(hsv[:, :, 1], MAX_SCREEN_SAT), hsv[:, :, 1])
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    outside = screen <= 0.0
    out[outside] = frame[outside]

    fin_lum = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    info = dict(
        visible_px=int(sel.sum()),
        occluded_frac=float((occ[qm > 0.5] > 0.5).mean()),
        luma_before=tgt_mean, luma_after=float(fin_lum[sel].mean()),
        p95_before=tgt_p95, p95_after=float(np.percentile(fin_lum[sel], 95)),
        exposure_gamma=float(gain), defocus_sigma=sigma,
        grain_plate=want, grain_new=have, grain_added=added,
        specular_frac=spec_frac,
    )
    if verbose:
        print(f"[mon] {mon['id']}: {info['visible_px']} px visible "
              f"({100 * info['occluded_frac']:.0f}% of the panel occluded), "
              f"luma {tgt_mean:.1f} -> {info['luma_after']:.1f} "
              f"(p95 {tgt_p95:.0f} -> {info['p95_after']:.0f}), "
              f"gamma {gain:.2f}, defocus sigma {sigma:.2f}, "
              f"grain plate {want:.2f} new {have:.2f} added {added:.2f}, "
              f"specular kept {100 * spec_frac:.1f}%")
    return out, info


def human_protection(img: np.ndarray, grow: float = 0.012) -> np.ndarray:
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

    # Keep only the largest component. Semantic segmentation cannot tell a
    # person in the room from a *picture* of a person on a screen, and the right
    # monitor displays exactly that - it was being reported as 33% occluded by
    # its own content, which would have left the old figure's silhouette
    # untouched under the new interface. The real subject is an order of
    # magnitude larger than anything shown on a panel behind him.
    binm = (p > 0.4).astype(np.uint8)
    n, labels, st, _ = cv2.connectedComponentsWithStats(binm, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        binm = (labels == biggest).astype(np.uint8)

    g = max(int(grow * min(h, w)) | 1, 5)
    return cv2.dilate(binm.astype(np.float32), np.ones((g, g), np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="assets/master/master_v03_lighting.png")
    ap.add_argument("--geometry", default="config/monitor_geometry.json")
    ap.add_argument("--out", default="assets/master/master_v04_final.png")
    ap.add_argument("--only", default=None, help="replace a single monitor by id")
    ap.add_argument("--content", default=None,
                    help="override the content image (requires --only)")
    ap.add_argument("--mask-out", default="07_monitor_mask.png")
    args = ap.parse_args()

    plate = cv2.imread(args.source)
    if plate is None:
        raise FileNotFoundError(args.source)
    geom = json.loads(Path(args.geometry).read_text())

    protect = human_protection(plate)
    frame = plate.copy()

    total_mask = np.zeros(plate.shape[:2], np.float32)
    report = {}
    for mon in geom["monitors"]:
        if not mon.get("replace"):
            print(f"[mon] {mon['id']}: replace=false, left as found "
                  f"({mon.get('note', '')[:60]})")
            continue
        if args.only and mon["id"] != args.only:
            continue
        content_path = args.content if (args.only and args.content) else mon["content"]
        src = cv2.imread(content_path)
        if src is None:
            raise FileNotFoundError(content_path)

        qm = quad_mask(plate.shape, np.float32(mon["quad"]))
        occ = occlusion_mask(frame, qm, protect)
        total_mask = np.maximum(total_mask, np.clip(qm * (1 - occ), 0, 1))

        frame, info = replace_monitor(frame, mon, src, protect)
        report[mon["id"]] = info

    cv2.imwrite(args.mask_out, (total_mask * 255).astype(np.uint8))
    print(f"[mon] screen mask {100 * (total_mask > 0.5).mean():.2f}% of frame "
          f"-> {args.mask_out}")

    outside = total_mask <= 0.0
    d = int(np.abs(frame[outside].astype(int) - plate[outside].astype(int)).max())
    print(f"[mon] byte lock outside screen mask: max diff {d}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, frame)
    print(f"[mon] wrote {args.out}")

    Path("monitor_report.json").write_text(json.dumps(
        {"byte_lock": d, "monitors": report}, indent=2))

    # Before / after, full frame and at 100% on each replaced panel.
    cw = 900
    ch = int(cw * plate.shape[0] / plate.shape[1])
    cells = []
    for lab, t in (("v03 lighting", plate), ("v04 monitors replaced", frame)):
        r = cv2.resize(t, (cw, ch))
        cv2.rectangle(r, (0, 0), (cw, 38), (16, 16, 16), -1)
        cv2.putText(r, lab, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 235, 255), 2)
        cells.append(r)
    rows = [np.hstack(cells)]
    for mon in geom["monitors"]:
        if not mon.get("replace") or (args.only and mon["id"] != args.only):
            continue
        q = np.int32(mon["quad"])
        x0, y0 = max(int(q[:, 0].min()) - 12, 0), max(int(q[:, 1].min()) - 12, 0)
        x1 = min(int(q[:, 0].max()) + 12, plate.shape[1])
        y1 = min(int(q[:, 1].max()) + 12, plate.shape[0])
        pair = []
        for lab, t in (("before", plate), ("after", frame)):
            c = t[y0:y1, x0:x1].copy()
            s = min(2.0, 880.0 / max(c.shape[1], 1))
            c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)),
                           interpolation=cv2.INTER_NEAREST)
            c = cv2.copyMakeBorder(c, 30, 6, 6, 6, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            cv2.putText(c, f"{mon['id']} {lab}", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 235, 255), 1)
            pair.append(c)
        hgt = max(p.shape[0] for p in pair)
        pair = [cv2.copyMakeBorder(p, 0, hgt - p.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=(20, 20, 20)) for p in pair]
        rows.append(np.hstack(pair))
    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(20, 20, 20)) for r in rows]
    cv2.imwrite("08_monitor_before_after.png", np.vstack(rows))
    print("[mon] comparison -> 08_monitor_before_after.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
