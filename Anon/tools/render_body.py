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

# Cameras are derived from the body, not typed in. The presets below are
# (side offset, forward distance, height above the seat, look-at height above
# the seat, lens) in *body-relative* terms; `setup_scene` turns them into world
# positions using the measured geometry, so they stay framed if the rig changes.
#
# The high and rear views exist because they are the cheat detectors: a high
# angle shows whether the hands really touch the desk and where the knees are,
# and a rear view shows shoulder and breathing motion that is nearly invisible
# head-on.
# Forward is **+Z** in this space, so a front camera has a positive forward
# offset. The first version used negative values - Blender's convention, not the
# source data's - and every view came out mirrored front-to-back: the preset
# named "rear" was the one showing his face.
CAMERAS = {
    "front":   (0.00, +2.30, 0.75, 0.42, 50.0),
    "side":    (2.05, +0.55, 0.70, 0.38, 52.0),
    "three_q": (1.35, +1.85, 0.80, 0.42, 50.0),
    # High enough to see the shoulders over the backrest, which is the
    # whole point of a rear view.
    "rear":    (-0.55, -1.75, 1.45, 0.55, 50.0),
    "high":    (0.55, +1.25, 2.30, 0.05, 45.0),
    "torso":   (0.00, +1.55, 0.72, 0.40, 62.0),
    "hands":   (0.30, +1.05, 0.95, 0.28, 68.0),
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

    # Desk, chair, floor. Blocks on purpose - the body is what is being judged.
    #
    # Every plane comes from `rig_geometry`, the same module the hand and foot
    # targets are computed from. Placing furniture independently is what put the
    # desk behind him last time.
    from presenter.motion.rig_geometry import (UNITS_PER_CM, SeatedGeometry,
                                               load_joints, mh_to_blender)

    j = load_joints()
    g = SeatedGeometry.measure(j)
    cm = UNITS_PER_CM

    def slab(name, centre_mh, half_mh, shade):
        loc = mh_to_blender(centre_mh)
        hx, hy, hz = half_mh                       # half-extents in MH axes
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=loc)
        o = bpy.context.object
        o.name = name
        o.scale = (hx, hz, hy)                     # MH (x,y,z) -> BL (x,z,y)
        m = bpy.data.materials.new(name + "_mat")
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (shade, shade, shade * 1.05, 1)
        b.inputs["Roughness"].default_value = 0.72
        o.data.materials.append(m)
        return o

    seat_mid_z = 0.5 * (g.seat_back_z + g.seat_front_z)
    seat_half_z = 0.5 * (g.seat_front_z - g.seat_back_z)

    slab("floor", (0.0, g.floor_y - 1.0 * cm, seat_mid_z),
         (140 * cm, 1.0 * cm, 140 * cm), 0.17)
    slab("chair_seat", (0.0, g.seat_y - 3.0 * cm, seat_mid_z),
         (24 * cm, 3.0 * cm, seat_half_z), 0.05)
    # Mid-back height. A full-height backrest is realistic but hides the
    # shoulders from the rear camera, and the rear view exists precisely to
    # check shoulder and breathing motion.
    slab("chair_back", (0.0, g.seat_y + 24 * cm, g.seat_back_z - 3.0 * cm),
         (23 * cm, 24 * cm, 3.0 * cm), 0.05)
    slab("desk", (0.0, g.desk_y - 1.5 * cm, g.desk_front_z + 35 * cm),
         (60 * cm, 1.5 * cm, 35 * cm), 0.10)

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

    # Body-relative preset -> world placement, using the measured geometry.
    span = (g.shoulder_y - g.floor_y)              # a body-scale length
    sx, fz, hy, look_hy, lens = CAMERAS[camera]
    cam_mh = (sx * span,
              g.seat_y + hy * span,
              g.seat_front_z + fz * span)
    look_mh = (0.0, g.seat_y + look_hy * span, g.seat_back_z + 0.20 * span)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = lens
    cam = bpy.data.objects.new("cam", cam_data)
    cam.location = Vector(mh_to_blender(cam_mh))
    bpy.context.collection.objects.link(cam)
    _look_at(cam, mh_to_blender(look_mh))
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
