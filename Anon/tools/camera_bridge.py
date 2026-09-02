"""Convert our canonical Blender cameras into the convention Gaussian renderers use.

This is what keeps "ONE WORLD, SEVEN CAMERAS" true when the human is no longer
rendered by Blender. If the reconstruction turns out to be Gaussians rather than
a mesh, the plan is to render the human through its own renderer and
depth-composite it into the Blender room - and that only works if both are given
*the same camera*, to the pixel.

## The two conventions, and why this is easy to get wrong

| | forward | up | right |
|---|---|---|---|
| **Blender / OpenGL** | local **-Z** | +Y | +X |
| **OpenCV / 3DGS / COLMAP** | local **+Z** | **-Y** | +X |

So the rotation differs by a 180-degree turn about X. Getting it backwards
produces an image that looks *almost* right - correctly framed, vertically
mirrored or rotated - which is exactly the kind of bug that survives a casual
glance and ruins a composite.

Rather than trusting the algebra, `--verify` projects real world points through
Blender's own `world_to_camera_view` and through the exported matrices, and
compares. If they disagree the export is wrong, and the script says so.

## Intrinsics

From the physical camera, not from a guessed field of view:

    fx = focal_length_mm / sensor_width_mm * image_width

`sensor_fit` is HORIZONTAL in our rig, so the sensor width maps to image width
and fy = fx (square pixels).

Usage
-----
    python tools/camera_bridge.py                      # write both formats
    python tools/camera_bridge.py --verify             # check against Blender
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Blender/OpenGL -> OpenCV: flip Y and Z.
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def look_at_rotation(position, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world rotation, Blender convention (-Z forward, +Y up).

    Matches `mathutils.Vector.to_track_quat('-Z','Y')`, which is what
    `scene3d/world.py` uses to aim the cameras - so the two cannot drift apart.
    """
    position = np.asarray(position, float)
    target = np.asarray(target, float)
    forward = target - position
    n = np.linalg.norm(forward)
    if n < 1e-9:
        return np.eye(3)
    forward /= n

    world_up = np.asarray(up, float)
    if abs(float(np.dot(forward, world_up))) > 0.999:      # looking along up
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)

    # Columns are the camera's local axes in world space: X=right, Y=up, Z=back.
    return np.column_stack([right, cam_up, -forward])


def intrinsics(focal_mm, sensor_mm, width, height) -> np.ndarray:
    fx = focal_mm / sensor_mm * width
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fx, height / 2.0],
                     [0.0, 0.0, 1.0]])


def build(cameras_yaml=None, width=None, height=None):
    cfg = yaml.safe_load(
        (ROOT / (cameras_yaml or "config/cameras.yaml")).read_text(encoding="utf-8"))
    d = cfg["defaults"]
    W = width or d["resolution"][0]
    H = height or d["resolution"][1]
    sensor = d["sensor_width_mm"]

    out = []
    for spec in cfg["cameras"]:
        pos = np.asarray(spec["position"], float)
        R_c2w_gl = look_at_rotation(pos, spec["look_at"])

        # Camera-to-world in OpenCV convention, then invert for world-to-camera,
        # which is what 3DGS renderers actually consume.
        R_c2w_cv = R_c2w_gl @ GL_TO_CV
        R_w2c = R_c2w_cv.T
        t_w2c = -R_w2c @ pos

        K = intrinsics(spec["focal_length_mm"], sensor, W, H)
        fov_x = 2.0 * math.atan(sensor / (2.0 * spec["focal_length_mm"]))
        fov_y = 2.0 * math.atan((sensor * H / W) / (2.0 * spec["focal_length_mm"]))

        c2w_gl = np.eye(4)
        c2w_gl[:3, :3] = R_c2w_gl
        c2w_gl[:3, 3] = pos

        w2c = np.eye(4)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c

        out.append({
            "id": spec["id"],
            "name": spec.get("name", spec["id"]),
            "enabled": bool(spec.get("enabled", True)),
            "width": W, "height": H,
            "focal_length_mm": spec["focal_length_mm"],
            "sensor_width_mm": sensor,
            "f_stop": spec.get("f_stop"),
            "position": pos.tolist(),
            "look_at": list(spec["look_at"]),
            "K": K.tolist(),
            "fov_x_rad": fov_x, "fov_y_rad": fov_y,
            "world_to_camera_opencv": w2c.tolist(),
            "camera_to_world_opengl": c2w_gl.tolist(),
        })
    return cfg.get("scene_version", "unknown"), out


def verify(cams, tol_px=1.0) -> int:
    """Project world points through Blender and through our matrices.

    Blender is the authority here: it is what renders the room, so if the
    exported matrices disagree with it, the exported matrices are wrong.
    """
    try:
        import bpy
        from bpy_extras.object_utils import world_to_camera_view
        from mathutils import Vector
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] cannot verify without bpy ({exc})")
        return 1

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tools"))
    from presenter.scene3d.world import build_world
    from render_multicam import frozen_pose

    pose, _ = frozen_pose(314.5)
    world = build_world(pose)
    scene = bpy.context.scene

    # Spread test points across the room so a rotation error cannot hide.
    probes = [(0.10, 1.16, 1.22), (0.0, 0.0, 1.5), (-1.5, 0.5, 0.8),
              (1.5, 2.0, 2.0), (0.0, 2.3, 0.74), (1.28, 0.075, 1.55)]

    worst = 0.0
    failures = 0
    for cam in cams:
        ob = world.cameras.get(cam["id"])
        if ob is None:
            continue
        scene.render.resolution_x = cam["width"]
        scene.render.resolution_y = cam["height"]
        K = np.array(cam["K"])
        w2c = np.array(cam["world_to_camera_opencv"])

        for p in probes:
            ref = world_to_camera_view(scene, ob, Vector(p))
            if ref.z <= 0:
                continue                       # behind the lens; nothing to compare
            bx = ref.x * cam["width"]
            by = (1.0 - ref.y) * cam["height"]

            hp = w2c @ np.array([p[0], p[1], p[2], 1.0])
            if hp[2] <= 0:
                failures += 1
                print(f"  {cam['id']}: point {p} behind camera in export but not "
                      f"in Blender - convention error")
                continue
            uv = K @ (hp[:3] / hp[2])
            err = math.hypot(uv[0] - bx, uv[1] - by)
            worst = max(worst, err)
            if err > tol_px:
                failures += 1
                print(f"  {cam['id']}: {p} -> blender ({bx:.1f},{by:.1f}) "
                      f"export ({uv[0]:.1f},{uv[1]:.1f})  err {err:.2f}px")

    print(f"\n[bridge] worst reprojection error: {worst:.4f} px "
          f"over {len(cams)} cameras x {len(probes)} points")
    if failures:
        print(f"[bridge] {failures} MISMATCH(ES) - do not composite with these "
              f"matrices")
        return 1
    print("[bridge] exported matrices agree with Blender. Safe to composite.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cameras", default="config/cameras.yaml")
    ap.add_argument("--out", default="config/cameras_cv.json")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--verify", action="store_true",
                    help="check the export against Blender's own projection")
    args = ap.parse_args()

    version, cams = build(args.cameras, args.width, args.height)
    payload = {
        "_comment": "Canonical cameras in OpenCV/3DGS convention (+Z forward, "
                    "-Y up), generated by tools/camera_bridge.py from "
                    "config/cameras.yaml. Do not hand-edit: regenerate.",
        "scene_version": version,
        "convention": "world_to_camera_opencv: +X right, +Y down, +Z forward",
        "cameras": cams,
    }
    out = ROOT / args.out
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[bridge] {len(cams)} cameras -> {out}")

    # nerfstudio / instant-ngp style, for tools that expect it.
    ns = {
        "camera_model": "OPENCV",
        "w": cams[0]["width"], "h": cams[0]["height"],
        "fl_x": cams[0]["K"][0][0], "fl_y": cams[0]["K"][1][1],
        "cx": cams[0]["K"][0][2], "cy": cams[0]["K"][1][2],
        "frames": [{"file_path": f"{c['id']}.png",
                    "transform_matrix": c["camera_to_world_opengl"]}
                   for c in cams],
    }
    ns_out = ROOT / "config/transforms.json"
    ns_out.write_text(json.dumps(ns, indent=2), encoding="utf-8")
    print(f"[bridge] nerfstudio-style transforms -> {ns_out}")

    for c in cams:
        if c["enabled"]:
            print(f"  {c['id']:6s} {c['focal_length_mm']:>4.0f}mm  "
                  f"fx={c['K'][0][0]:7.1f}  fovx={math.degrees(c['fov_x_rad']):5.1f}deg")

    return verify(cams) if args.verify else 0


if __name__ == "__main__":
    raise SystemExit(main())
