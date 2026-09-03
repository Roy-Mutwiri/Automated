"""Render one still per physical camera, so the camera plan can be looked at.

This answers exactly one question - **are the camera angles good?** - and
deliberately not the other one. The human is the debug proxy mannequin, the
samples are low, and nothing here is meant to be pretty. Identity is evaluated
by the reconstruction pipeline, not by this tool.

It renders all seven cameras including the ones marked `enabled: false` in
config/cameras.yaml. That flag gates *production* rendering, and cameras 4-7
are still blocked as production shots; seeing where a camera points is not the
same as shipping it, and you cannot judge a camera plan from the three views
that already work.

    python tools/render_camera_previews.py
    python tools/render_camera_previews.py --force        # ignore the cache
    python tools/render_camera_previews.py --width 1920 --height 1080

Output: renders/camera_preview/cam1..cam7.png and contact_sheet.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

OUT = ROOT / "renders" / "camera_preview"

# What each camera is *for*, in plain words. These are the labels the UI shows,
# and they describe the physical viewpoint - never the framing of a crop.
INTENT = {
    "cam1": "HERO FRONT",
    "cam2": "LEFT 3/4",
    "cam3": "RIGHT 3/4",
    "cam4": "OVER SHOULDER",
    "cam5": "REAR WIDE",
    "cam6": "HIGH DIAGONAL",
    "cam7": "ROOM WIDE",
}


def fingerprint(width, height, sim_time) -> str:
    """Everything that changes what a preview looks like.

    Camera transforms, lenses and the room all feed the render, so a change to
    any of them makes the cached image a picture of a world that no longer
    exists. Resolution and timestamp are in here too: a preview at a different
    size is a different file, not a stale one.
    """
    h = hashlib.sha256()
    for rel in ("config/cameras.yaml", "config/room_geometry.yaml"):
        h.update((ROOT / rel).read_bytes())
    h.update(f"{width}x{height}@{sim_time}".encode())
    return h.hexdigest()[:12]


def is_fresh(width, height, sim_time) -> bool:
    stamp = OUT / "preview.json"
    if not stamp.exists():
        return False
    try:
        meta = json.loads(stamp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if meta.get("fingerprint") != fingerprint(width, height, sim_time):
        return False
    return all((OUT / f"{cam}.png").exists() for cam in INTENT)


def render_all(width, height, sim_time, samples):
    import bpy

    from presenter.scene3d.world import build_world
    from render_multicam import frozen_pose

    pose, _ = frozen_pose(sim_time)
    world = build_world(pose)

    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE"
    scn.render.resolution_x, scn.render.resolution_y = width, height
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    # Matched across every camera. Auto-exposure drift between cuts is one of
    # the loudest multicam tells, and it would also make the contact sheet
    # unreadable as a comparison.
    scn.view_settings.view_transform = "Filmic"
    scn.view_settings.look = "None"
    scn.view_settings.exposure = 0.0
    if hasattr(scn, "eevee") and hasattr(scn.eevee, "taa_render_samples"):
        scn.eevee.taa_render_samples = samples

    OUT.mkdir(parents=True, exist_ok=True)
    specs = {c["id"]: c for c in world.c["cameras"]}
    written = []

    print(f"\n{'cam':<6}{'lens':>7}  {'position':<24}{'aim':<24}  file")
    print("-" * 92)
    for cam_id in INTENT:
        ob = world.cameras.get(cam_id)
        if ob is None:
            print(f"{cam_id:<6}  NOT IN THE RIG - check config/cameras.yaml")
            continue
        spec = specs[cam_id]
        scn.camera = ob
        path = OUT / f"{cam_id}.png"
        scn.render.filepath = str(path)
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        rot = [round(a, 1) for a in
               (ob.rotation_euler.x * 57.2958, ob.rotation_euler.y * 57.2958,
                ob.rotation_euler.z * 57.2958)]
        pos = "[" + ", ".join(f"{v:+.2f}" for v in spec["position"]) + "]"
        aim = "[" + ", ".join(f"{v:+.2f}" for v in spec["look_at"]) + "]"
        print(f"{cam_id:<6}{spec['focal_length_mm']:>5.0f}mm  {pos:<24}{aim:<24}  "
              f"{path.name}  ({time.perf_counter() - t0:.1f}s)")
        print(f"{'':<6}{'':>7}  rotation XYZ deg {rot}   {INTENT[cam_id]}"
              f"{'' if spec.get('enabled', True) else '   [production-blocked]'}")
        written.append(path)

    (OUT / "preview.json").write_text(json.dumps({
        "fingerprint": fingerprint(width, height, sim_time),
        "width": width, "height": height, "sim_time": sim_time,
        "human": "debug proxy mannequin - identity is NOT evaluated here",
        "cameras": {k: INTENT[k] for k in INTENT},
    }, indent=2), encoding="utf-8")
    return written


def contact_sheet(width, height):
    """CAM1..CAM4 on the top row, CAM5..CAM7 on the second, each labelled."""
    import cv2
    import numpy as np

    cell = (480, int(480 * height / width))
    tiles = []
    for cam_id, intent in INTENT.items():
        p = OUT / f"{cam_id}.png"
        im = cv2.imread(str(p))
        if im is None:
            im = np.full((cell[1], cell[0], 3), 40, np.uint8)
            cv2.putText(im, "MISSING", (14, cell[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 230), 2)
        else:
            im = cv2.resize(im, cell)
        band = im.copy()
        cv2.rectangle(band, (0, 0), (cell[0], 30), (0, 0, 0), -1)
        cv2.addWeighted(band, 0.6, im, 0.4, 0, im)
        cv2.putText(im, f"{cam_id.upper()}  {intent}", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 240, 250), 1, cv2.LINE_AA)
        tiles.append(im)

    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros((cell[1], cell[0], 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    sheet = np.vstack(rows)
    dest = OUT / "contact_sheet.png"
    cv2.imwrite(str(dest), sheet)
    print(f"\ncontact sheet -> {dest}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--time", type=float, default=314.5,
                    help="simulation timestamp to freeze at")
    ap.add_argument("--samples", type=int, default=16,
                    help="EEVEE samples; these are diagrams, not beauty shots")
    ap.add_argument("--force", action="store_true",
                    help="re-render even when the cache is fresh")
    args = ap.parse_args()

    if not args.force and is_fresh(args.width, args.height, args.time):
        print(f"previews are current ({len(INTENT)} cameras in {OUT}).")
        print("Camera transforms, lenses and room geometry are all unchanged.")
        print("Use --force to re-render anyway.")
        return 0

    t0 = time.perf_counter()
    render_all(args.width, args.height, args.time, args.samples)
    contact_sheet(args.width, args.height)
    print(f"\ntotal {time.perf_counter() - t0:.1f}s")
    print("The human is the debug proxy. This shows the ANGLES, not the man.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
