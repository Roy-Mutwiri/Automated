# Multi-camera architecture: comparison and recommendation

Companion to `multicam_technology_research.md`. That document establishes what
each technique needs as input; this one scores the five candidate architectures
against our requirements and commits to one.

## The requirements, weighted

From the specification's own priority order:

1. same person
2. same room
3. same timestamp
4. physical camera
5. realistic human
6. FPS

Note that **realistic human is fifth, below same-person and same-room**. The
specification says this explicitly: "A beautiful Camera 5 showing a different
man is worthless." That ordering is what decides the comparison, and it is worth
stating because the intuitive ranking would put realism first.

## Comparison

Scored 1-5, 5 best. "Camera 1 match" means: can this reach the quality of the
existing approved plate?

| | A: 3D human + 3D room, raster | B: Gaussian human + 3D room | C: Gaussian human + Gaussian room | D: 3D + neural face enhance | E: Canonical 3D + constrained neural enhance |
|---|---|---|---|---|---|
| Identity consistency across views | 5 | 5 | 5 | 4 | **5** |
| 360 completeness (rear, top) | 5 | 4 | 3 | 5 | **5** |
| Skin realism | 2 | 5 | 5 | 4 | **4** |
| Hair | 2 | 4 | 4 | 3 | **3** |
| Eyes | 3 | 4 | 4 | 4 | **4** |
| Animation from `AvatarPose` | 5 | 3 | 2 | 5 | **5** |
| Real-time FPS | 3 | 4 | 4 | 2 | **3** |
| Windows support | 5 | 2 | 2 | 4 | **4** |
| Feasible on 16 GB with ~2 GB free | 4 | 2 | 1 | 3 | **3** |
| Camera 1 quality match | 2 | 4 | 4 | 4 | **4** |
| Integration complexity (5 = simplest) | 4 | 2 | 1 | 3 | **3** |
| **Input we actually possess** | **5** | **1** | **1** | **5** | **5** |

The last row is decisive and is why B and C are not viable regardless of their
scores elsewhere: they require monocular video or multi-view capture of a subject
who does not exist.

## Recommendation: Architecture E, built in two locked stages

**Stage 1 - the canonical world, with a proxy human.**
Blender scene graph. Room, desk, chair, monitors, boom mic and shelving modelled
from the approved Camera 1 plate. One lighting rig. Seven physically
parameterised cameras. The human present as **articulated proxy geometry** with
correct skull, ear, shoulder and body proportions, driven by the existing
`BehaviorEngine` through `AvatarPose`.

This stage delivers, provably: same person, same room, same timestamp, physical
cameras, real parallax, real occlusion, real shadows. It delivers **none** of the
skin, hair or eye realism of Camera 1, and it must not be presented as if it
does.

**Stage 2 - human fidelity.**
Replace the proxy with a photoreal representation. Options ranked by expected
result: an SVAD-shaped pipeline (single image -> video diffusion -> synthetic
multi-view -> 3DGS avatar); a purchased or scanned photoreal human re-textured
toward the approved identity; MetaHuman with Unreal. All are multi-week and all
carry real risk of not matching Camera 1.

### Why staged rather than all at once

Because the two halves have completely different risk profiles. The room is
ordinary engineering with a predictable outcome. The human is an open research
problem. Coupling them means the predictable half cannot be validated until the
unpredictable half lands, and the specification's own milestone plan
(cam1-3 first, "same man / same room", only then continue) is a staged plan.

### What this costs, stated before starting

**Stage 1 will not look like Camera 1.** A proxy human in a modelled room, EEVEE
rendered, is a previsualisation - correct geometry, plausible materials,
unmistakably not a photograph. The specification anticipates this
("first prove consistency at reduced/debug resolution if necessary",
"architecture correctness comes first") but it is worth being blunt: the first
contact sheet will show seven consistent views of a man who does not yet look
real.

The alternative - keeping the current photoreal Camera 1 and generating the
others - has already been tried in this project and produced seven different men
in seven different rooms. That is the failure this architecture exists to fix.

## Consequences for existing code

| Component | Fate |
|---|---|
| `behavior/` | **Unchanged.** Already renderer-agnostic and time-driven. Drives the 3D rig directly. |
| `render/liveportrait.py` | Retained for the existing single-camera photoreal path. Not part of the canonical world. |
| `render/cameras.py`, `config/cameras.yaml` | **Replaced.** The current model selects between separate images, which is the rejected architecture. |
| `tools/generate_cameras.py` | **Retired.** Generating a camera as an independent image is what this work exists to stop. |
| `ui.py` button row, mouse dispatcher | Reused as-is. |
| `assets/reference/camera1_approved.png` | Immutable reference plate. |
| `tools/wall_material.py` design decisions | Carried into 3D: one global wall coordinate system, fixed slat pitch, dark base, black gaps. |

## Renderer choice within Stage 1

Blender `bpy` 5.0.1, verified working headless on this machine, EEVEE with
Cycles as fallback. Rendering runs **as a subprocess**, not in the app process:

- `bpy` pins numpy to 1.26.4; keeping it out of the torch process avoids a
  second dependency negotiation later
- the app keeps its VRAM budget, which on this machine is frequently under 2 GB
- a crashed render cannot take the stream down

The cost is that camera switching in Stage 1 is not real time. That is accepted
for a consistency proof and is revisited in Stage 2.

## Lock stages

Per the specification, these become explicit gates:

- `ROOM_GEOMETRY_LOCKED` - after Camera 1 match is convincing
- `HUMAN_IDENTITY_LOCKED` - after the identity turntable passes
- `MATERIALS_LOCKED`
- `CAMERAS_LOCKED`

Nothing downstream of a lock is edited casually.
