"""Render the MPFB test rig, driven by the same behaviour engine as the 2D face.

This is the proof that the architecture separates behaviour from renderer: the
engine is constructed identically to the one that produced the 2D clip, and
nothing in it knows a rig exists. Only the adapter differs.

The scene is deliberately plain. A chair, a desk, a mouse block, three lights.
Making it pretty would be time spent on the thing that is explicitly not being
judged - what is being judged is whether the chest breathes, the neck
participates in a turn, and the hands sit on something.

Camera presets exist because the brief is right that different angles expose
different failures: breathing and shoulder work are nearly invisible head-on
and obvious from the side.

Usage
-----
    python tools/render_body.py --seconds 5 --camera front --out probe.mp4
    python tools/render_body.py --still --time 40 --camera side
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RIG_BLEND = "assets/rig/body_rig.blend"

# (location, look-at height, lens). Units are the rig's own, which are roughly
# decimetres: the body spans about 18 units head to toe.
CAMERAS = {
    "front":   ((0.0, -17.0, 7.4), 6.2, 62.0),
    "side":    ((15.5, -7.0, 7.2), 6.0, 62.0),
    "three_q": ((10.0, -14.0, 7.8), 6.4, 60.0),
    "rear":    ((-5.0, 14.0, 8.4), 6.2, 58.0),
    "top":     ((0.5, -7.0, 18.0), 3.5, 45.0),
    "torso":   ((0.0, -11.0, 6.4), 5.8, 72.0),
}


def _look_at(obj, target):
    from mathutils import Vector
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def setup_scene(camera: str, width: int, height: int, samples: int = 16):
    import bpy
    from mathutils import Vector

    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = samples

    world = bpy.data.worlds.new("w") if not bpy.data.worlds else bpy.data.worlds[0]
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.055, 0.065, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.6

    # Desk, chair, floor. Blocks, on purpose - what is being judged is the
    # body, not the furniture.
    #
    # Placement is derived from the contact targets rather than typed in. The
    # first version put the desk at +Y, which after the MakeHuman-to-Blender
    # conversion is *behind* him: forward is -Y. He was sitting with his back
    # to his own desk.
    tgt = bpy.data.objects.get("target_mouse")
    desk_z = tgt.location.z if tgt is not None else 2.0
    desk_y = tgt.location.y if tgt is not None else -6.0

    for name, loc, size in (
        ("desk", (0.0, desk_y - 1.6, desk_z - 0.3), (8.5, 3.0, 0.28)),
        ("chair_seat", (0.0, 1.2, 0.9), (4.6, 4.2, 0.35)),
        ("chair_back", (0.0, 4.4, 5.6), (4.6, 0.45, 4.8)),
        ("floor", (0.0, 0.0, -9.4), (26.0, 26.0, 0.2)),
    ):
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=loc)
        o = bpy.context.object
        o.name = name
        o.scale = size
        m = bpy.data.materials.new(name + "_mat")
        m.use_nodes = True
        bsdf = m.node_tree.nodes["Principled BSDF"]
        shade = 0.05 if name.startswith("chair") else (0.10 if name == "desk" else 0.17)
        bsdf.inputs["Base Color"].default_value = (shade, shade, shade * 1.06, 1)
        bsdf.inputs["Roughness"].default_value = 0.72
        o.data.materials.append(m)

    for name, loc, energy, size in (
        ("key", (-9.0, -14.0, 13.0), 9000.0, 7.0),
        ("fill", (11.0, -9.0, 7.0), 2600.0, 9.0),
        ("rim", (2.0, 13.0, 12.0), 4200.0, 5.0),
    ):
        light = bpy.data.lights.new(name, type="AREA")
        light.energy = energy
        light.size = size
        lo = bpy.data.objects.new(name, light)
        lo.location = loc
        bpy.context.collection.objects.link(lo)
        _look_at(lo, (0.0, 0.0, 5.0))

    loc, look_h, lens = CAMERAS[camera]
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = lens
    cam = bpy.data.objects.new("cam", cam_data)
    cam.location = loc
    bpy.context.collection.objects.link(cam)
    _look_at(cam, (0.0, 0.0, look_h))
    scene.camera = cam
    return scene


def render_to(scene, path: Path):
    import bpy
    scene.render.filepath = str(path.resolve())
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default=RIG_BLEND)
    ap.add_argument("--camera", default="front", choices=sorted(CAMERAS))
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--profile", default="PRESENTER_CALM")
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--time", type=float, default=0.0,
                    help="with --still, simulate this many seconds first")
    ap.add_argument("--out", default="body_probe.png")
    ap.add_argument("--frames-dir", default="")
    args = ap.parse_args()

    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.rig).resolve()))

    from presenter.behavior.engine import BehaviorEngine
    from presenter.motion.adapters.mpfb import MPFBAdapter

    rig = bpy.data.objects["body_rig"]
    adapter = MPFBAdapter(rig)
    scene = setup_scene(args.camera, args.width, args.height, args.samples)

    engine = BehaviorEngine(profile=args.profile, seed=args.seed)
    dt = 1.0 / args.fps

    if args.still:
        for _ in range(max(int(args.time * args.fps), 1)):
            engine.update(dt)
        adapter.apply(engine.motion)
        bpy.context.view_layer.update()
        render_to(scene, Path(args.out))
        m = engine.motion
        print(f"[body] t={args.time:.1f}s  chest.rx {m.chest.rx:+.2f}  "
              f"clavL.rz {m.clavicle_l.rz:+.2f}  neck.ry {m.neck.ry:+.2f}  "
              f"head.ry {m.head.ry:+.2f}  breath {m.breathing.drive:.2f}")
        print(f"[body] wrote {args.out}")
        return 0

    out_dir = Path(args.frames_dir or "body_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(args.seconds * args.fps)
    t0 = time.perf_counter()
    for i in range(n):
        engine.update(dt)
        adapter.apply(engine.motion)
        bpy.context.view_layer.update()
        render_to(scene, out_dir / f"f{i:05d}.png")
        if i % 30 == 0 and i:
            el = time.perf_counter() - t0
            print(f"[body] {i}/{n}  {i / el:.2f} fps  "
                  f"eta {(n - i) / max(i / el, 1e-3) / 60:.1f} min", flush=True)
    print(f"[body] {n} frames -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
