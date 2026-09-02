# Avatar architecture: identity, rig and runtime as three separate stages

Written in response to a correct challenge. I had argued "HumanNOVA is not
animatable, therefore it cannot be canonical", and that conflates two different
problems. A reconstruction is judged on *how much of the person it recovers*; a
rig is judged on *how well it deforms*. A system can be excellent at one and
useless at the other, and picking one model to do both is what forced the false
choice.

    SOURCE PLATE  (assets/reference/avatar_identity_camera1.png)
          |
      SEGMENTATION            done - avatar_rgba.png, frozen
          |
    ==== STAGE A - IDENTITY ====        what does he look like, in 3D
          |
    ==== STAGE B - RIG ==========        how does that geometry deform
          |
    ==== STAGE C - RUNTIME ======        render it, driven by BehaviorEngine
          |
      CAM1 / CAM2 / CAM3

The stages are separable, which means each can be sourced independently and,
crucially, **licensed independently**.

## Stage A - identity

Produce geometry and appearance that is recognisably *this man*.

| Option | Commercial | Cost to try | Identity ceiling |
|---|---|---|---|
| **MPFB2 / MakeHuman base mesh, shaped to the plate** | **CC0 assets - yes** | free, no toolchain | medium; depends on fitting and texture projection |
| Licensed service (Meshcapade / Avaturn / Didimo) | yes, paid | subscription | high |
| LHM / HumanNOVA / HumanLift | **no** - MPI body models | ~9 GB toolchain + weights | high |
| Commercial SMPL-X licence, then any of the above | yes, paid | procurement | high |

[MakeHuman's assets - base mesh, targets, skins - are CC0](https://static.makehumancommunity.org/about/license.html),
and MPFB2's own FAQ is explicit that
[GPL covers the addon code, not exported characters](https://static.makehumancommunity.org/mpfb/faq/use_in_closed_source.html).
MPFB2 is a vetted Blender extension for 4.2+. We are on `bpy` 5.0.1.

## Stage B - rig

Deform that geometry from `AvatarPose`.

**We do not need SMPL-X in the runtime.** Our subject's shape is fixed - it is
one man, not a population - so nothing at runtime needs shape blendshapes. What
runtime needs is a skeleton, skinning weights and facial blendshapes, all of
which can be authored in Blender:

- armature + weights (Rigify, or a custom armature - simpler and more
  predictable for a seated presenter)
- shape keys for expressions, ideally on ARKit-compatible names so future
  audio-driven animation has a standard target
- real eye geometry as separate objects with a world-space gaze target, which
  the proxy already demonstrates works

And if a parametric body ever *is* wanted at runtime,
[SMPL-X BODY is CC BY 4.0](https://smpl-x.is.tue.mpg.de/bodylicense.html) -
mesh, skeleton, pose blendshapes, commercial use with attribution. It is only
the *shape* blendshapes, which are a reconstruction-stage concern, that fall
under the non-commercial Model licence. That distinction is the single most
useful thing to come out of the licence audit.

## Stage C - runtime

Unchanged from the multicam baseline: canonical world, seven physical cameras,
one `BehaviorEngine`, one simulation clock. The human is one node in the scene
graph, which is exactly why Stage A can be replaced later without touching the
camera work.

One caveat carried from the licence audit: `bpy` is **GPL-3.0**. Rendering
already runs in a subprocess, which was chosen for dependency isolation and
turns out to matter here too. It should stay that way.

## The rig interface the behaviour terminal gets

Documented now so the other terminal can build against it before Stage A
lands, per requirement 47:

| Control | Type | Source in `AvatarPose` |
|---|---|---|
| head yaw / pitch / roll | bone rotation, neck + head | `yaw`, `pitch`, `roll` |
| gaze | world-space target, eyes aim at it | `gaze_x`, `gaze_y` |
| eyelids | blendshape or lid bones, per eye | `eye_open_l`, `eye_open_r` |
| brows | blendshapes | `brow_l`, `brow_r`, `brow_furrow` |
| breathing | chest scale / spine bone | `scale`, `breathing_phase` |
| shoulders, posture | spine + clavicle bones | posture channels |
| jaw / mouth | blendshapes, reserved | `mouth_open` (unused until lip-sync) |

Head rotation must move the **neck**, not spin a head sphere - side and rear
cameras expose that immediately.

## Why the stages must not be collapsed again

The temptation is to find one model that does all three. Every such model in
the current field is built on an MPI parametric body or head, so collapsing the
stages also collapses the licensing into a single non-commercial verdict.
Keeping them separate is what makes a commercially clean path exist at all.
