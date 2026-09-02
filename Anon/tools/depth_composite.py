"""Depth-composite an externally rendered human into the Blender room.

If the reconstruction turns out to be Gaussians rather than a mesh, the human is
rendered by *its* renderer and the room by Blender - two renderers, one world.
Alpha-pasting the human on top would be wrong: the microphone boom physically
passes in front of him, the chair wraps behind him, and the desk cuts across his
forearms. Those relationships are depth, not layer order.

So both sides emit **RGB + depth in metres along the camera's optical axis**,
and the winner is decided per pixel.

## The convention, stated once

    depth = distance from the camera plane along its viewing direction, in
            metres. Larger = further. Background = +inf.

Blender's `Depth` render pass is already exactly this (Z distance along the
camera axis, not radial). Anything the Gaussian renderer produces must be
converted *to* this before it gets here - and `--debug` exists because a
convention error looks like a subtle occlusion bug rather than an obvious one.

## Why there is a false-colour debug image

A flipped depth axis produces a composite where the human is occluded by
nothing, or by everything, and both can look plausible in a single frame. The
debug image shows room depth, human depth, and which one won per pixel, so an
inversion is visible immediately instead of three cameras later.

Usage
-----
    python tools/depth_composite.py --camera cam1 \
        --human-rgba human_cam1.png --human-depth human_cam1_depth.exr

    python tools/depth_composite.py --camera cam1 --self-test
"""

from __future__ import annotations

import argparse
import os
import time as _time
from pathlib import Path

# OpenCV ships with OpenEXR support compiled in but DISABLED unless this is set,
# and it signals refusal by returning None from imread - which looks exactly
# like a missing file. Set it before cv2 is imported or it has no effect.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAR = 1.0e10
# Chosen so that 65535 / DEPTH_RANGE_M == 1000 exactly: a 16-bit depth PNG then
# stores plain millimetres.
DEPTH_RANGE_M = 65.535
# Below this, a human pixel has no coverage and must not contribute colour no
# matter what its depth buffer claims.
ALPHA_EPS = 1.0 / 255.0


def validate_depth(d, name, require_geometry=True):
    """Enforce the depth contract, loudly.

    The contract, in full:

        valid surface   positive finite metres from the camera plane
        nothing here    +infinity (FAR)
        anything else   a bug

    NaN, zero and negative depths are rejected rather than repaired. They have
    no physical meaning here, and every one of them would quietly change an
    occlusion decision: NaN loses every comparison, zero and negatives win
    every comparison. A human that silently draws in front of the whole room is
    what a negative depth looks like from the outside.

    `require_geometry` catches the uninitialized buffer - the failure that
    actually happened. When cv2 refused to decode the EXR it returned None,
    the depth became all-FAR, and the occlusion self-test passed while
    comparing nothing against nothing.
    """
    d = np.asarray(d)
    if d.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D depth map, got {d.shape}")
    if not np.issubdtype(d.dtype, np.floating):
        raise ValueError(f"{name}: depth must be floating point, got {d.dtype}")

    nan = int(np.isnan(d).sum())
    if nan:
        raise ValueError(f"{name}: {nan} NaN depths. NaN loses every "
                         f"comparison, so those pixels would silently never "
                         f"occlude anything.")
    bad = int((d <= 0.0).sum())
    if bad:
        lo = float(d.min())
        raise ValueError(f"{name}: {bad} depths <= 0 (min {lo}). Zero and "
                         f"negative depths win every comparison, which draws "
                         f"that surface in front of the entire room.")
    if not np.isfinite(d[d < FAR / 2]).all():
        raise ValueError(f"{name}: non-finite depth below the FAR sentinel")

    finite = int((d < FAR / 2).sum())
    if require_geometry and finite == 0:
        raise ValueError(
            f"{name}: every pixel is FAR - the buffer holds no geometry at "
            f"all. This is what an unread or unwritten depth image looks like, "
            f"and it makes any occlusion test pass vacuously.")
    return finite


def _read_exr(path: Path) -> np.ndarray:
    """Read a float EXR through Blender.

    The wheels of opencv-python are built with `OpenEXR: NO`, so cv2.imread
    returns None for any EXR - indistinguishable from a missing file, which is
    how an all-infinity depth buffer can quietly make every occlusion test pass.
    bpy is already a hard dependency of this tool, and it does read EXR.
    """
    import bpy

    im = bpy.data.images.load(str(path.resolve()))
    try:
        w, h, c = im.size[0], im.size[1], im.channels
        if not (im.has_data and w and h and c):
            raise RuntimeError(
                f"Blender loaded no pixels from {path.name}. A multilayer EXR "
                f"cannot be read this way - re-export it as a single-layer, "
                f"single-channel float EXR, or supply a 16-bit PNG in mm.")
        buf = np.empty(w * h * c, dtype=np.float32)
        im.pixels.foreach_get(buf)
        # Blender stores pixels bottom-up; cv2 and the rest of this file are
        # top-down. Flipping here keeps the convention in one place.
        return np.flipud(buf.reshape(h, w, c))[..., 0].copy()
    finally:
        bpy.data.images.remove(im)


def load_depth(path: Path) -> np.ndarray:
    """Read a depth image as float32 metres.

    EXR keeps real distances; a PNG cannot, so a 16-bit PNG is treated as
    millimetres, which is the only sane lossless-integer convention and is what
    most depth exporters use.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    img = (_read_exr(path) if path.suffix.lower() == ".exr"
           else cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH))
    if img is None:
        raise RuntimeError(f"cv2 could not decode {path}")
    if img.ndim == 3:
        img = img[..., 0]
    d = img.astype(np.float32)
    if img.dtype == np.uint16:
        d /= 1000.0
    d[~np.isfinite(d)] = FAR
    d[d <= 0.0] = FAR
    # A 16-bit millimetre image saturates at DEPTH_RANGE_M. Anything at the
    # ceiling of the range is background that was clipped, not a real surface
    # 65 m away, and must not be allowed to occlude anything.
    d[d >= DEPTH_RANGE_M - 0.01] = FAR
    return d


def _hide_human():
    """Hide every part of the Blender proxy human.

    His pixels must come from the other renderer; if both draw him they fight
    over the same silhouette. `hide_viewport` matters as much as `hide_render`
    here because the depth pass below is ray-cast, not rendered.
    """
    import bpy

    named = ("streamer_fitted", "skull", "jaw", "hair", "torso", "shoulders",
             "neck")
    prefixes = ("ear_", "eye_", "iris_", "lid_", "upperarm_", "forearm_",
                "hand_", "headphone")
    hidden = 0
    for ob in bpy.data.objects:
        if ob.name in named or ob.name.startswith(prefixes):
            ob.hide_render = True
            ob.hide_viewport = True
            hidden += 1
    return hidden


def room_depth_map(world, camera_id, width, height, cache=True):
    """Depth of the room, in metres along the camera's optical axis.

    Computed by ray-casting rather than by reading a render pass. That is a
    deliberate choice: Blender 5.0's File Output node writes only multilayer
    EXR, opencv-python is built with `OpenEXR: NO` so it cannot read one back,
    and a compositor Viewer node stays empty in background mode. Ray-casting
    sidesteps all of it and returns true metres with no encoding in between.

    It also avoids a subtler problem. Anti-aliasing averages a rendered depth
    pass across silhouette edges, inventing surfaces halfway between the desk
    and the wall behind it - exactly where occlusion decisions are made. A ray
    either hits or it does not.

    ~20 s at 960x540, but the room is rigid: nothing in it moves between
    frames, so the result is cached per camera and reused.
    """
    import bpy
    from mathutils import Vector

    key = ROOT / "renders" / "depth_cache" / f"{camera_id}_{width}x{height}.npy"
    if cache and key.exists():
        cached = np.load(key)
        if cached.shape == (height, width):
            print(f"[composite] room depth: cached {key.name}")
            return cached

    cam = world.cameras[camera_id]
    scn = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()

    # view_frame gives the four corners of the frustum in camera-local space,
    # so it already accounts for focal length, sensor size and sensor fit -
    # nothing here re-derives the intrinsics by hand.
    tr, br, bl, tl = cam.data.view_frame(scene=scn)
    rot = cam.matrix_world.to_3x3()
    origin = cam.matrix_world.translation
    forward = (rot @ Vector((0.0, 0.0, -1.0))).normalized()

    depth = np.full((height, width), FAR, np.float32)
    t0 = _time.perf_counter()
    for row in range(height):
        v = (row + 0.5) / height
        left = tl.lerp(bl, v)
        right = tr.lerp(br, v)
        for col in range(width):
            u = (col + 0.5) / width
            d = (rot @ left.lerp(right, u)).normalized()
            hit, loc, _, _, _, _ = scn.ray_cast(dg, origin, d)
            if hit:
                # Distance along the optical axis, not radial distance. This is
                # the same quantity Blender's Z pass reports, and the human
                # renderer must supply the same or occlusion will be wrong.
                depth[row, col] = (loc - origin).dot(forward)
        if row % 90 == 0:
            print(f"[composite]   depth row {row}/{height}", flush=True)
    print(f"[composite] room depth: {_time.perf_counter() - t0:.1f}s "
          f"({(depth < FAR / 2).mean():.1%} of frame is geometry)")

    if cache:
        key.parent.mkdir(parents=True, exist_ok=True)
        np.save(key, depth)
    return depth


def render_room(camera_id: str, width: int, height: int, sim_time: float):
    """Render the canonical room through one camera, returning RGB + depth.

    The room is rendered *without* the human: the human comes from the other
    renderer. Everything else - desk, chair, mic, monitors, wall - is Blender's,
    and its depth is what decides occlusion.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tools"))
    import bpy

    from presenter.scene3d.world import build_world
    from render_multicam import frozen_pose

    pose, _ = frozen_pose(sim_time)
    world = build_world(pose)
    print(f"[composite] hid {_hide_human()} human objects")

    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE"
    scn.render.resolution_x, scn.render.resolution_y = width, height
    scn.render.resolution_percentage = 100
    # Identical to render_multicam.render(). The room half of a composite must
    # be tonemapped exactly like the multicam renders or the seam shows up as a
    # brightness step along the silhouette.
    scn.view_settings.view_transform = "Filmic"
    scn.view_settings.look = "None"
    scn.view_settings.exposure = 0.0
    scn.camera = world.cameras[camera_id]

    rgb_path = ROOT / "renders" / f"room_{camera_id}.png"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    scn.render.filepath = str(rgb_path)
    bpy.ops.render.render(write_still=True)
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        raise RuntimeError(f"render wrote no readable image at {rgb_path}")

    depth = room_depth_map(world, camera_id, width, height)
    if depth.shape != rgb.shape[:2]:
        raise RuntimeError(f"depth {depth.shape} != render {rgb.shape[:2]}")
    return rgb, depth


def composite(room_rgb, room_depth, human_rgba, human_depth, debug=False):
    """Per-pixel depth resolution. The nearer surface wins."""
    h, w = room_rgb.shape[:2]
    validate_depth(room_depth, "room depth")
    validate_depth(human_depth, "human depth")
    if room_depth.shape != (h, w) or human_depth.shape != (h, w):
        raise ValueError(f"shape mismatch: rgb {(h, w)}, room depth "
                         f"{room_depth.shape}, human depth {human_depth.shape}")
    if human_rgba.shape[:2] != (h, w):
        raise ValueError(f"human rgba {human_rgba.shape[:2]} != rgb {(h, w)}")

    human_rgb = human_rgba[..., :3].astype(np.float32)
    alpha = (human_rgba[..., 3:4].astype(np.float32) / 255.0
             if human_rgba.shape[2] == 4 else np.ones((h, w, 1), np.float32))

    # Coverage is checked before depth and independently of it. A Gaussian
    # renderer will happily emit a depth value for a pixel it did not cover;
    # without this the human would draw where he is not.
    hd = human_depth.copy()
    hd[alpha[..., 0] < ALPHA_EPS] = FAR

    human_in_front = (hd < room_depth)[..., None]
    # Alpha still matters at the silhouette: a half-covered pixel is a blend of
    # the two surfaces, not a hard choice between them.
    weight = human_in_front.astype(np.float32) * alpha
    out = human_rgb * weight + room_rgb.astype(np.float32) * (1.0 - weight)

    if not debug:
        return out.astype(np.uint8), None

    def colourise(d):
        finite = d[np.isfinite(d) & (d < FAR / 2)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0, 1)
        norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        norm[d >= FAR / 2] = 1.0
        return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    winner = np.zeros((h, w, 3), np.uint8)
    winner[..., 1] = (human_in_front[..., 0] & (alpha[..., 0] > 0.5)) * 255  # human
    winner[..., 2] = (~human_in_front[..., 0]) * 255                        # room
    panel = np.hstack([colourise(room_depth), colourise(hd), winner])
    for i, label in enumerate(("room depth", "human depth", "winner (G=human R=room)")):
        cv2.putText(panel, label, (12 + i * w, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out.astype(np.uint8), panel


def _slab(width, height, depth_m, colour):
    """A flat card standing at a fixed depth, filling the middle of frame."""
    d = np.full((height, width), FAR, np.float32)
    y0, y1 = int(height * 0.10), int(height * 0.95)
    x0, x1 = int(width * 0.33), int(width * 0.67)
    d[y0:y1, x0:x1] = depth_m
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[y0:y1, x0:x1] = colour
    return rgba, d


def self_test(camera_id, width, height, sim_time):
    """Validate the compositor without a reconstruction.

    Two probes, because one is not enough. A card at the subject's depth should
    be in front of almost everything - but so would a card at *any* depth if the
    depth test were broken open. So a second card is placed behind the desk,
    where the room must win. A compositor that passes only the first test is
    indistinguishable from one that always draws the human on top, which is the
    exact bug this whole file exists to avoid.
    """
    import yaml

    room_rgb, room_depth = render_room(camera_id, width, height, sim_time)

    cams = yaml.safe_load((ROOT / "config/cameras.yaml").read_text(encoding="utf-8"))
    spec = next(c for c in cams["cameras"] if c["id"] == camera_id)
    pos = np.array(spec["position"], float)
    head = np.array([0.10, 1.16, 1.22])
    dist = float(np.linalg.norm(head - pos))

    finite = room_depth[room_depth < FAR / 2]
    print("")
    print(f"[composite] --- depth sanity, {camera_id} ---")
    print(f"[composite] camera at {list(pos)}, subject head {dist:.3f} m away")
    print(f"[composite] room depth  min {finite.min():.3f}  "
          f"median {np.median(finite):.3f}  max {finite.max():.3f} m")
    print(f"[composite] background (no geometry hit): {(room_depth >= FAR/2).mean():.1%}")

    # Preconditions. None of the occlusion numbers below mean anything unless
    # the scene actually contains the things being compared, so they are
    # asserted rather than assumed - the vacuous pass is the failure mode this
    # whole file was written around.
    room_finite = validate_depth(room_depth, "room depth")
    print(f"[composite] precondition: {room_finite} room pixels carry geometry")

    results = {}
    for label, depth_m, colour in (("subject", dist, (90, 150, 230, 255)),
                                   ("behind-desk", finite.max() - 0.05,
                                    (200, 120, 90, 255))):
        rgba, hd = _slab(width, height, depth_m, colour)
        card = hd < FAR / 2
        card_px = int(card.sum())
        overlap = int((card & (room_depth < FAR / 2)).sum())
        if card_px == 0:
            raise AssertionError(f"{label}: the probe card has no finite pixels")
        if overlap == 0:
            raise AssertionError(
                f"{label}: the card and the room geometry do not overlap "
                f"anywhere, so no occlusion decision is being exercised. A "
                f"pass here would be meaningless.")
        out, panel = composite(room_rgb, room_depth, rgba, hd, debug=True)
        in_front = int(((hd < room_depth) & card).sum())
        behind = int(((room_depth <= hd) & card).sum())
        results[label] = (in_front, behind)
        print(f"[composite] precondition: {label} card {card_px} px, "
              f"{overlap} px overlap room geometry")
        print(f"[composite] card at {depth_m:6.3f} m ({label:11s}): "
              f"{in_front:7d} px in front, {behind:7d} px occluded by the room")
        if label == "subject":
            cv2.imwrite(str(ROOT / f"renders/{camera_id}_composite_selftest.png"), out)
            cv2.imwrite(str(ROOT / f"renders/{camera_id}_depth_debug.png"), panel)

    # The two probes must disagree. If they do not, the depth test is not
    # actually being consulted.
    near_front, _ = results["subject"]
    _, far_behind = results["behind-desk"]
    ok = near_front > 0 and far_behind > 0
    print("")
    print(f"[composite] {'PASS' if ok else 'FAIL'}: "
          f"near card wins {near_front} px, far card loses {far_behind} px")
    if not ok:
        print("[composite] the depth test is not discriminating. Do not composite "
              "a real human until this passes - a silent depth inversion looks "
              "like a plausible image.")
    print(f"[composite] -> renders/{camera_id}_depth_debug.png")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--time", type=float, default=314.5)
    ap.add_argument("--human-rgba", default=None)
    ap.add_argument("--human-depth", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.self_test or not (args.human_rgba and args.human_depth):
        if not args.self_test:
            print("[composite] no human render supplied; running the self-test")
        return self_test(args.camera, args.width, args.height, args.time)

    room_rgb, room_depth = render_room(args.camera, args.width, args.height,
                                       args.time)
    human_rgba = cv2.imread(args.human_rgba, cv2.IMREAD_UNCHANGED)
    human_depth = load_depth(Path(args.human_depth))
    out, panel = composite(room_rgb, room_depth, human_rgba, human_depth,
                           debug=args.debug)
    dest = ROOT / f"renders/{args.camera}_composite.png"
    cv2.imwrite(str(dest), out)
    print(f"[composite] -> {dest}")
    if panel is not None:
        cv2.imwrite(str(ROOT / f"renders/{args.camera}_depth_debug.png"), panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
