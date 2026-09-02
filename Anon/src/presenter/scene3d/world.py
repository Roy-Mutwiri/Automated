"""Build the canonical 3D world in Blender. ONE world, seven projections.

This module is the whole point of the multi-camera architecture: there is a
single scene graph, built once from `config/room_geometry.yaml`, and a camera
is a projection of it. Nothing is authored per camera. If two cameras disagree
about where the desk is, this file is wrong - a camera cannot be wrong about it,
because a camera has no say in the matter.

## Why Blender

`bpy` gives a scene graph, physically-parameterised cameras (sensor size, focal
length in mm, f-stop, focus distance), one lighting rig, and real occlusion,
parallax and shadows by construction. See `docs/multicam_architecture.md` for
the comparison against Gaussian and neural approaches, and why they were
rejected: they need multi-view capture of a subject who does not exist.

## The human is a proxy, and this is stated plainly

The figure built here has correct *proportions, silhouette and articulation* -
skull, ears, neck, shoulders, arms - driven by the real `AvatarPose` from the
behaviour engine. It does not have photoreal skin, hair or eyes and is not
pretending to. Stage 1 proves same-man / same-room / same-moment; Stage 2
replaces this object with a photoreal representation. It is one node in the
graph, which is exactly why that replacement is possible later.

Building a proxy is not a shortcut around the identity problem. It is the
opposite: it makes identity a property of *geometry shared between views*
rather than of a prompt, which is the only thing that can make a rear view and
a front view be the same person.

## Determinism

The world is built from configuration with no randomness anywhere. Two runs
produce byte-identical geometry, which is what allows the frozen-timestamp
contact sheet to mean anything.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
import yaml

__all__ = ["World", "build_world"]

ROOT = Path(__file__).resolve().parents[3]


# -- small helpers ----------------------------------------------------------
def _clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _material(name: str, rgb, roughness=0.6, metallic=0.0, emission=None,
              emission_strength=1.0):
    """One material instance per name, reused everywhere.

    Reuse is not an optimisation here - it is the Material Consistency rule.
    Two cameras cannot see different walnut if there is only one walnut.
    """
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def _box(name, centre, size, mat, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=centre)
    ob = bpy.context.object
    ob.name = name
    ob.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    ob.rotation_euler = rotation
    ob.data.materials.append(mat)
    return ob


def _ellipsoid(name, centre, radii, mat, rotation=(0, 0, 0), segments=32):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=centre,
                                         segments=segments, ring_count=segments // 2)
    ob = bpy.context.object
    ob.name = name
    ob.scale = radii
    ob.rotation_euler = rotation
    bpy.ops.object.shade_smooth()
    ob.data.materials.append(mat)
    return ob


def _capsule(name, a, b, radius, mat):
    """A cylinder between two world points - limbs, mic booms, chair posts."""
    ax, ay, az = a
    bx, by, bz = b
    dx, dy, dz = bx - ax, by - ay, bz - az
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=max(length, 1e-4), vertices=20,
        location=((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2),
    )
    ob = bpy.context.object
    ob.name = name
    # Point local +Z along the segment.
    ob.rotation_euler = (math.acos(max(-1.0, min(1.0, dz / max(length, 1e-9)))),
                         0.0,
                         math.atan2(dy, dx) + math.pi / 2)
    bpy.ops.object.shade_smooth()
    ob.data.materials.append(mat)
    return ob


def _look_at_euler(position, target):
    """Aim an object at a world point.

    Blender cameras and lights look down their local -Z with +Y up. Deriving
    those Euler angles by hand is easy to get subtly wrong - the first version
    of this function did, and every camera rendered a flat grey wall because it
    was aimed at the floor. `to_track_quat` is the supported way and cannot be
    off by a convention.
    """
    from mathutils import Vector

    direction = Vector(target) - Vector(position)
    if direction.length < 1e-9:
        return (0.0, 0.0, 0.0)
    return tuple(direction.to_track_quat("-Z", "Y").to_euler())


def _distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class World:
    """The canonical scene. Built once; cameras only project it."""

    def __init__(self, geometry: dict, cameras: dict) -> None:
        self.g = geometry
        self.c = cameras
        self.objects: dict[str, object] = {}
        self.landmarks: dict[str, tuple[float, float, float]] = {}

    # -- room ---------------------------------------------------------------
    def build_room(self) -> None:
        r = self.g["room"]
        w, d, h = r["width"], r["depth"], r["height"]
        floor = _material("floor", (0.055, 0.042, 0.032), roughness=0.45)
        felt = _material("charcoal_felt", (0.030, 0.030, 0.033), roughness=0.92)
        ceil = _material("ceiling", (0.045, 0.045, 0.048), roughness=0.9)

        _box("floor", (0, d / 2, -0.02), (w, d, 0.04), floor)
        _box("ceiling", (0, d / 2, h + 0.02), (w, d, 0.04), ceil)
        _box("wall_left", (-w / 2 - 0.02, d / 2, h / 2), (0.04, d, h), felt)
        _box("wall_right", (w / 2 + 0.02, d / 2, h / 2), (0.04, d, h), felt)
        _box("wall_rear", (0, d + 0.02, h / 2), (w, 0.04, h), felt)

    def build_walnut_wall(self) -> None:
        """The slat wall, on ONE global coordinate system.

        Every batten exists at a fixed world X derived from `slat_phase_x` and
        `slat_pitch`. This is the 3D form of the rule `tools/wall_material.py`
        established for the 2D plate: compute the pattern once across the whole
        surface, never per visible fragment, or each fragment gets its own phase
        and the wall changes between views.
        """
        wcfg = self.g["wall_walnut"]
        felt = _material("charcoal_felt", (0.030, 0.030, 0.033), roughness=0.92)
        walnut = _material("dark_walnut", (0.085, 0.048, 0.030), roughness=0.55)

        z0, z1 = wcfg["z_from"], wcfg["z_to"]
        _box("wall_backing", (0, -0.01, (z0 + z1) / 2),
             (wcfg["x_to"] - wcfg["x_from"], 0.02, z1 - z0), felt)

        pitch = wcfg["slat_pitch"]
        x = wcfg["slat_phase_x"]
        i = 0
        while x <= wcfg["x_to"]:
            _box(f"slat_{i:03d}", (x, wcfg["slat_depth"] / 2, (z0 + z1) / 2),
                 (wcfg["slat_width"], wcfg["slat_depth"], z1 - z0), walnut)
            self.landmarks[f"slat_{i:03d}"] = (x, 0.0, (z0 + z1) / 2)
            x += pitch
            i += 1

    def build_monitors(self) -> None:
        shell = _material("monitor_shell", (0.020, 0.020, 0.022), roughness=0.45)
        for m in self.g["monitors"]:
            cx, cy, cz = m["centre"]
            sx, sy, sz = m["size"]
            # Real shell with depth, not a paper-thin plane: Camera 5 sees the
            # back of these and a plane would vanish edge-on.
            _box(f"{m['id']}_shell", (cx, cy + sy / 2, cz), (sx, sy, sz), shell)
            _capsule(f"{m['id']}_arm", (cx, cy + sy, cz), (cx, 0.02, cz),
                     0.018, shell)
            if m.get("powered"):
                panel = _material(
                    f"{m['id']}_panel", (0.06, 0.09, 0.14), roughness=0.25,
                    emission=(0.16, 0.26, 0.42),
                    emission_strength=self.g["lighting"]["monitor_emission"][
                        "power_w_per_m2"] / 20.0,
                )
            else:
                panel = _material("panel_off", (0.012, 0.012, 0.014),
                                  roughness=0.18)
            _box(f"{m['id']}_panel", (cx, cy + sy + 0.002, cz),
                 (sx * 0.955, 0.004, sz * 0.93), panel)
            self.landmarks[m["id"]] = (cx, cy, cz)

    def build_desk_and_props(self) -> None:
        black = _material("matte_black", (0.021, 0.021, 0.023), roughness=0.55)
        steel = _material("black_steel", (0.030, 0.030, 0.032), roughness=0.4,
                          metallic=0.7)

        d = self.g["desk"]
        cx, cy, _ = d["centre"]
        tw, td, tt = d["top_size"]
        top_z = d["top_z"]
        _box("desk_top", (cx, cy, top_z - tt / 2), (tw, td, tt), black)
        for sx in (-1, 1):
            _capsule(f"desk_leg_{'lr'[max(sx,0)]}",
                     (cx + sx * (tw / 2 - 0.08), cy, 0.0),
                     (cx + sx * (tw / 2 - 0.08), cy, top_z - tt), 0.028, steel)
        tray = d["cable_tray"]
        _box("desk_cable_tray", tray["centre"], tray["size"], steel)
        self.landmarks["desk_main"] = (cx, cy, top_z)

        ch = self.g["chair"]
        leather = _material("black_leatherette", (0.017, 0.017, 0.019),
                            roughness=0.62)
        bx, by, _ = ch["base_centre"]
        sh = ch["seat_height"]
        _box("chair_seat", (bx, by, sh), ch["seat_size"], leather)
        back_sx, back_sy, back_sz = ch["back_size"]
        _box("chair_back", (bx, by - ch["seat_size"][1] / 2 + back_sy / 2,
                            sh + back_sz / 2 + 0.05),
             ch["back_size"], leather)
        hx, hy, hz = ch["headrest_size"]
        _box("chair_headrest",
             (bx, by - ch["seat_size"][1] / 2 + back_sy / 2,
              sh + back_sz + hz / 2 + 0.10), ch["headrest_size"], leather)
        _capsule("chair_post", (bx, by, 0.02), (bx, by, sh - 0.05), 0.035, steel)
        for k in range(5):
            a = 2 * math.pi * k / 5
            _capsule(f"chair_foot_{k}", (bx, by, 0.03),
                     (bx + 0.30 * math.cos(a), by + 0.30 * math.sin(a), 0.02),
                     0.018, steel)
        self.landmarks["chair_main"] = (bx, by, sh)

        mic = self.g["microphone"]
        joints = mic["joints"]
        for k in range(len(joints) - 1):
            _capsule(f"mic_boom_{k}", joints[k], joints[k + 1], 0.014, steel)
        _ellipsoid("mic_capsule", mic["capsule_centre"],
                   (mic["capsule_size"][0], mic["capsule_size"][1],
                    mic["capsule_size"][2] / 2), black)
        self.landmarks["mic_main"] = tuple(mic["capsule_centre"])

        for s in self.g.get("speakers", []):
            _box(s["id"], s["centre"], s["size"], black)
            self.landmarks[s["id"]] = tuple(s["centre"])

        walnut = _material("dark_walnut", (0.085, 0.048, 0.030), roughness=0.55)
        green = _material("plant", (0.035, 0.075, 0.030), roughness=0.75)
        for lm in self.g.get("landmarks", []):
            mat = green if "plant" in lm["id"] else (
                walnut if "shelf" in lm["id"] else black)
            _box(lm["id"], lm["centre"], lm["size"], mat)
            self.landmarks[lm["id"]] = tuple(lm["centre"])

    # -- the human ----------------------------------------------------------
    def build_human(self, pose) -> None:
        """Proxy figure, posed from the behaviour engine's `AvatarPose`.

        Proportions are a real seated adult male. The head is built as a skull
        with a separate jaw/beard mass and *actual ears*, because the ears and
        the skull silhouette are what a rear or three-quarter camera uses to
        say "same man" - see the specification's identity priority order.
        """
        h = self.g["human"]
        hx, hy, hz = h["hip"]
        skin = _material("skin", (0.34, 0.20, 0.13), roughness=0.58)
        hair = _material("hair", (0.020, 0.014, 0.012), roughness=0.72)
        shirt = _material("shirt", (0.055, 0.058, 0.062), roughness=0.85)
        cup = _material("headphone", (0.018, 0.018, 0.020), roughness=0.5)

        shoulder_z = hz + 0.50
        neck_base_z = shoulder_z + 0.05
        head_c_z = h["eye_height"] + 0.045
        breath = (pose.scale - 1.0) if hasattr(pose, "scale") else 0.0

        # Torso. Breathing scales the chest only - it must not move the chair.
        _ellipsoid("torso", (hx, hy, (hz + shoulder_z) / 2),
                   (0.20 * (1 + breath * 0.6), 0.145 * (1 + breath),
                    (shoulder_z - hz) / 2 + 0.06), shirt)
        _box("shoulders", (hx, hy, shoulder_z), (0.46, 0.20, 0.11), shirt)

        # Arms forward to the desk, one stable pose for every camera.
        desk_z = self.g["desk"]["top_z"]
        for side, sx in (("l", -1), ("r", 1)):
            sh = (hx + sx * 0.225, hy, shoulder_z - 0.02)
            elbow = (hx + sx * 0.30, hy + 0.28, hz + 0.30)
            hand = (hx + sx * 0.24, hy + 0.62, desk_z + 0.04)
            _capsule(f"upperarm_{side}", sh, elbow, 0.052, skin)
            _capsule(f"forearm_{side}", elbow, hand, 0.044, skin)
            _ellipsoid(f"hand_{side}", hand, (0.045, 0.075, 0.028), skin)

        _capsule("neck", (hx, hy, shoulder_z - 0.02),
                 (hx, hy - 0.01, neck_base_z + 0.05), 0.058, skin)

        # Head group, rotated as one so the neck and skull cannot separate.
        yaw = math.radians(getattr(pose, "yaw", 0.0))
        pitch = math.radians(getattr(pose, "pitch", 0.0))
        roll = math.radians(getattr(pose, "roll", 0.0))
        head_c = (hx, hy - 0.012, head_c_z)

        skull = _ellipsoid("skull", head_c, (0.085, 0.098, 0.108), skin)
        jaw = _ellipsoid("jaw", (head_c[0], head_c[1] + 0.020, head_c[2] - 0.062),
                         (0.070, 0.082, 0.055), hair)      # beard mass
        hair_shell = _ellipsoid(
            "hair", (head_c[0], head_c[1] - 0.012, head_c[2] + 0.022),
            (0.094, 0.104, 0.104), hair)
        parts = [skull, jaw, hair_shell]

        # Ears: identity features, and the thing a profile camera exposes.
        for side, sx in (("l", -1), ("r", 1)):
            parts.append(_ellipsoid(
                f"ear_{side}",
                (head_c[0] + sx * 0.086, head_c[1] + 0.004, head_c[2] - 0.004),
                (0.012, 0.026, 0.034), skin))
            parts.append(_ellipsoid(
                f"headphone_{side}",
                (head_c[0] + sx * 0.100, head_c[1] + 0.004, head_c[2] - 0.004),
                (0.022, 0.044, 0.050), cup))
        parts.append(_capsule(
            "headphone_band",
            (head_c[0] - 0.098, head_c[1] + 0.004, head_c[2] + 0.030),
            (head_c[0] + 0.098, head_c[1] + 0.004, head_c[2] + 0.030),
            0.010, cup))

        # Eyes as real geometry with a world-space gaze direction. Never
        # painted per camera.
        target = self.g["human"]["gaze_targets"]["main_camera"]
        eye_white = _material("sclera", (0.72, 0.70, 0.68), roughness=0.25)
        iris = _material("iris", (0.055, 0.035, 0.020), roughness=0.2)
        for side, sx in (("l", -1), ("r", 1)):
            ec = (head_c[0] + sx * 0.032, head_c[1] + 0.078, head_c[2] + 0.012)
            parts.append(_ellipsoid(f"eye_{side}", ec, (0.0125,) * 3, eye_white,
                                    segments=16))
            d = math.sqrt(sum((t - e) ** 2 for t, e in zip(target, ec)))
            gx = (target[0] - ec[0]) / d
            gy = (target[1] - ec[1]) / d
            gz = (target[2] - ec[2]) / d
            parts.append(_ellipsoid(
                f"iris_{side}",
                (ec[0] + gx * 0.0105, ec[1] + gy * 0.0105, ec[2] + gz * 0.0105),
                (0.0058,) * 3, iris, segments=12))
            # Eyelid: a lid mass that drops with the blink signal, so a blink is
            # visible from every camera rather than being a face-crop effect.
            open_amount = getattr(pose, f"eye_open_{side}", 1.0)
            lid_drop = (1.0 - open_amount) * 0.020
            parts.append(_ellipsoid(
                f"lid_{side}", (ec[0], ec[1] - 0.002, ec[2] + 0.016 - lid_drop),
                (0.016, 0.013, 0.010), skin, segments=14))

        # Rotate the whole head about the neck pivot.
        pivot = (hx, hy - 0.01, neck_base_z + 0.02)
        for ob in parts:
            ob.rotation_euler = (pitch, roll, yaw)
            rel = (ob.location[0] - pivot[0], ob.location[1] - pivot[1],
                   ob.location[2] - pivot[2])
            cy_, sy_ = math.cos(yaw), math.sin(yaw)
            rx = rel[0] * cy_ - rel[1] * sy_
            ry = rel[0] * sy_ + rel[1] * cy_
            cp, sp = math.cos(pitch), math.sin(pitch)
            rz = rel[2] * cp - ry * sp
            ry = ry * cp + rel[2] * sp
            ob.location = (pivot[0] + rx, pivot[1] + ry, pivot[2] + rz)

        self.landmarks["head_centre"] = head_c
        self.landmarks["eye_l"] = (head_c[0] - 0.032, head_c[1] + 0.078,
                                   head_c[2] + 0.012)
        self.landmarks["eye_r"] = (head_c[0] + 0.032, head_c[1] + 0.078,
                                   head_c[2] + 0.012)
        self.landmarks["ear_l"] = (head_c[0] - 0.086, head_c[1], head_c[2])
        self.landmarks["ear_r"] = (head_c[0] + 0.086, head_c[1], head_c[2])

    def build_fitted_human(self, pose, obj_path, clip_below=0.95) -> bool:
        """Load the fitted CC0 mesh human instead of the proxy.

        Coordinates have to be converted, and getting this wrong is silent:
        MakeHuman's base mesh is **Y-up and in decimetres**, our world is
        **Z-up and in metres**. So the mesh is scaled by 0.1 and rotated +90
        degrees about X.

        `clip_below` hides geometry beneath roughly mid-chest. The mesh is a
        *standing* figure and this stage has no rig to seat it, so its legs
        would pass straight through the desk. Cameras 1-3 see head, neck,
        shoulders and upper torso, which is exactly the region the identity
        gate judges, so the rest is hidden rather than faked. This is a stated
        limitation of the identity experiment, not a trick to pass it.
        """
        obj_path = Path(obj_path)
        if not obj_path.exists():
            print(f"[world] no fitted mesh at {obj_path}; using the proxy")
            return False

        before = set(bpy.data.objects.keys())
        bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis="NEGATIVE_Z",
                              up_axis="Y")
        new = [bpy.data.objects[n] for n in bpy.data.objects.keys()
               if n not in before]
        if not new:
            print("[world] import produced no object")
            return False
        ob = new[0]
        ob.name = "streamer_fitted"

        # Metres, and stood upright in a Z-up world.
        ob.scale = (0.1, 0.1, 0.1)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

        # Place by the head, because the head is what the cameras are aimed at
        # and what the gate is about. The feet land wherever they land.
        h = self.g["human"]
        lo = [min(v.co[i] for v in ob.data.vertices) for i in range(3)]
        hi = [max(v.co[i] for v in ob.data.vertices) for i in range(3)]
        head_top_target = h["eye_height"] + 0.18
        ob.location = (
            h["hip"][0] - (lo[0] + hi[0]) / 2,
            h["hip"][1] - (lo[1] + hi[1]) / 2,
            head_top_target - hi[2],
        )
        print(f"[world] fitted mesh: {len(ob.data.vertices)} verts, "
              f"height {hi[2] - lo[2]:.3f} m, placed at z offset "
              f"{ob.location[2]:.3f}")

        if clip_below is not None:
            mesh = ob.data
            zoff = ob.location[2]
            keep = [p for p in mesh.polygons
                    if max(mesh.vertices[i].co[2] for i in p.vertices) + zoff
                    >= clip_below]
            drop = len(mesh.polygons) - len(keep)
            if drop:
                import bmesh
                bm = bmesh.new()
                bm.from_mesh(mesh)
                bm.faces.ensure_lookup_table()
                doomed = [f for f in bm.faces
                          if max(v.co[2] for v in f.verts) + zoff < clip_below]
                bmesh.ops.delete(bm, geom=doomed, context="FACES")
                bm.to_mesh(mesh)
                bm.free()
                print(f"[world] hid {drop} faces below z={clip_below} "
                      f"(standing mesh, no rig to seat it yet)")

        ob.data.materials.clear()
        bpy.ops.object.shade_smooth()

        # Turn him round to face the cameras. The MakeHuman base mesh looks
        # down -Y once converted to our Z-up world, and our subject faces +Y.
        # Behaviour yaw is applied on top of that half turn.
        yaw = math.radians(getattr(pose, "yaw", 0.0))
        ob.rotation_euler = (0.0, 0.0, math.pi + yaw)

        self.landmarks["head_centre"] = (h["hip"][0], h["hip"][1] - 0.012,
                                         h["eye_height"] + 0.045)
        return ob

    def project_identity_texture(self, ob, plate_path, camera_id="cam1"):
        """Project the approved plate onto the mesh through Camera 1.

        Skin tone, brows, beard and lip colour are the strongest identity
        signals a face has, and none of them are geometry. The plate already
        contains all of them, photographed through a camera whose parameters we
        know exactly - so projecting it back along that camera's rays puts the
        right colour on the right part of the mesh by construction, with no
        painting and no guessing.

        This is a *front* projection and it is honest about what that means:

        * Surfaces facing Camera 1 get correct colour.
        * Surfaces turned away from it - the sides of the head, behind the
          ears, the back of the skull - get stretched colour, because the plate
          contains no information about them. Camera 2 and 3 will show that.

        Nothing is invented for the hidden regions here. That is a separate
        decision, taken once and locked, and it should be made after seeing how
        far the honest projection gets.
        """
        cam = self.cameras.get(camera_id)
        if cam is None or not Path(plate_path).exists():
            return False

        from bpy_extras.object_utils import world_to_camera_view

        scene = bpy.context.scene
        mesh = ob.data
        uv = mesh.uv_layers.new(name="cam1_projection")
        mw = ob.matrix_world
        for loop in mesh.loops:
            co = mw @ mesh.vertices[loop.vertex_index].co
            p = world_to_camera_view(scene, cam, co)
            uv.data[loop.index].uv = (p.x, p.y)
        mesh.uv_layers.active = uv

        mat = bpy.data.materials.new("identity_projected")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(plate_path))
        tex.extension = "EXTEND"
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.55
        # Some subsurface, or skin reads as painted plastic under a key light.
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = 0.12
        mesh.materials.clear()
        mesh.materials.append(mat)
        print(f"[world] projected {Path(plate_path).name} through {camera_id} "
              f"onto {len(mesh.loops)} loops")
        return True

    # -- lighting -----------------------------------------------------------
    def build_lighting(self) -> None:
        """One rig. Every camera observes it; no camera relights the scene."""
        lg = self.g["lighting"]

        def kelvin_rgb(k):
            # Cheap but monotonic: warm below 5000 K, cool above.
            t = (k - 2000) / 4500.0
            return (1.0, 0.62 + 0.30 * min(t, 1.0), 0.35 + 0.62 * min(t, 1.0))

        for name in ("key", "fill", "rim"):
            spec = lg[name]
            bpy.ops.object.light_add(type="AREA", location=spec["position"])
            ob = bpy.context.object
            ob.name = f"light_{name}"
            ob.data.energy = spec["power_w"]
            ob.data.size = spec["size"][0]
            ob.data.size_y = spec["size"][1]
            ob.data.shape = "RECTANGLE"
            ob.data.color = kelvin_rgb(spec["colour_k"])
            ob.rotation_euler = _look_at_euler(spec["position"], spec["target"])

        for p in lg.get("practicals", []):
            bpy.ops.object.light_add(type="AREA", location=p["position"])
            ob = bpy.context.object
            ob.name = f"practical_{p['id']}"
            ob.data.energy = p["power_w"]
            ob.data.size = p["size"][0]
            ob.data.color = kelvin_rgb(p["colour_k"])
            ob.rotation_euler = (0.0, 0.0, 0.0)

        world = bpy.data.worlds.new("world")
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs[0].default_value = (
            0.012, 0.012, 0.014, 1.0)
        world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
        bpy.context.scene.world = world

    # -- cameras ------------------------------------------------------------
    def build_cameras(self) -> dict:
        """Create every camera. They differ only in transform and lens."""
        d = self.c["defaults"]
        made = {}
        for spec in self.c["cameras"]:
            data = bpy.data.cameras.new(spec["id"])
            data.sensor_fit = "HORIZONTAL"
            data.clip_start = d.get("clip_start", 0.1)
            data.clip_end = d.get("clip_end", 100.0)
            data.sensor_width = d["sensor_width_mm"]
            data.lens = spec["focal_length_mm"]
            data.dof.use_dof = True
            data.dof.aperture_fstop = spec["f_stop"]
            focus = self._focus_point(spec)
            data.dof.focus_distance = _distance(spec["position"], focus)
            ob = bpy.data.objects.new(spec["id"], data)
            ob.location = spec["position"]
            ob.rotation_euler = _look_at_euler(spec["position"], spec["look_at"])
            bpy.context.scene.collection.objects.link(ob)
            made[spec["id"]] = ob
        return made

    def _focus_point(self, spec):
        name = spec.get("focus_target", "head")
        if name == "head":
            return self.landmarks.get("head_centre", spec["look_at"])
        return self.landmarks.get(name, spec["look_at"])

    def validate_cameras(self) -> list[str]:
        """No impossible cameras: inside the room, not inside furniture."""
        r = self.g["room"]
        problems = []
        solids = {
            "desk": (self.g["desk"]["centre"], self.g["desk"]["top_size"]),
            "chair": (self.g["chair"]["base_centre"], (0.6, 0.6, 1.3)),
        }
        for spec in self.c["cameras"]:
            x, y, z = spec["position"]
            if not (-r["width"] / 2 < x < r["width"] / 2):
                problems.append(f"{spec['id']}: x={x} outside the room")
            if not (0.05 < y < r["depth"]):
                problems.append(f"{spec['id']}: y={y} outside the room")
            if not (0.1 < z < r["height"]):
                problems.append(f"{spec['id']}: z={z} outside the room")
            for name, (c, s) in solids.items():
                if (abs(x - c[0]) < s[0] / 2 and abs(y - c[1]) < s[1] / 2
                        and z < c[2] + s[2]):
                    problems.append(f"{spec['id']}: inside {name}")
        return problems


def build_world(pose, geometry_path="config/room_geometry.yaml",
                cameras_path="config/cameras.yaml",
                human_mesh: str | None = None) -> World:
    """Build the entire canonical scene for one frozen simulation state."""
    geometry = yaml.safe_load((ROOT / geometry_path).read_text(encoding="utf-8"))
    cameras = yaml.safe_load((ROOT / cameras_path).read_text(encoding="utf-8"))

    _clear()
    world = World(geometry, cameras)
    world.build_room()
    world.build_walnut_wall()
    world.build_monitors()
    world.build_desk_and_props()
    # The fitted CC0 mesh if one has been built, otherwise the debug proxy.
    # The proxy proved the camera system and is explicitly not the product.
    if not (human_mesh and world.build_fitted_human(pose, human_mesh)):
        world.build_human(pose)
    world.build_lighting()
    world.cameras = world.build_cameras()
    return world
