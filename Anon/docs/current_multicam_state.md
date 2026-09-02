# Current state, before multi-camera work

Inspection date: 2026-09-02. This is what exists, what is reusable, and what is
missing. Written before any multicam code, as required.

## The headline

**There is no 3D anywhere in this project.** Every pixel produced so far is 2D:
a diffusion-generated photograph, refined by deterministic 2D compositing, with
a face crop warped by LivePortrait. There is no scene graph, no world
coordinate system, no geometry, no camera model, and no renderer that can
produce a second viewpoint of anything.

That is not a criticism of the existing work - it is a very capable 2D pipeline
and Camera 1 is good precisely because of it. But it means multi-camera cannot
be an extension of what is here. It needs a representation that does not exist
yet.

## Dependency audit

| Capability | Present | Notes |
|---|---|---|
| `bpy` (Blender as a Python module) | **installed during this inspection** | 5.0.1, Windows, EEVEE renders headless - verified |
| `trimesh`, `pyrender`, `moderngl`, `PyOpenGL` | no | |
| `open3d` | no | |
| `pytorch3d`, `nvdiffrast` | no | neither builds trivially on Windows/sm_120 |
| Gaussian splatting (any) | no | |
| `smplx`, FLAME, DECA/EMOCA | no | no parametric body or head model |
| MediaPipe | **yes** | face mesh available; currently unused by the renderer |
| torch 2.11.0+cu128, CUDA | yes | sm_120 verified working |
| diffusers, SDXL base + SDXL inpainting | yes | ~13 GB of local weights |
| Blender application (GUI) | no | not installed; the pip module is enough |
| Unreal Engine | no | |

`bpy` pinned **numpy to 1.26.4** (down from 2.4.6). torch, OpenCV and all 36
tests were re-verified after the downgrade and pass. This is a real constraint
to remember, not a footnote.

## What exists and is directly reusable

### The approved Camera 1 plate
`assets/master/master_v04_final.png`, produced by a four-stage deterministic
chain from `master_v01_original.png`:

1. `tools/generate_scene.py` - the whole scene in one SDXL pass
2. `tools/wall_material.py` - walnut slats composited with **one global wall
   coordinate system** and frequency separation
3. `tools/scene_lighting.py` - practicals added as real emitters in linear light
4. `tools/monitor_replace.py` - screen content by homography
5. `tools/refine_face.py` - head region re-rendered at higher internal resolution

Backed up to `assets/reference/camera1_approved.png`. **This file is the
reference plate and must not be overwritten.**

### Partial geometry already recorded
`config/monitor_geometry.json` holds screen quadrilaterals in plate image
coordinates, with corners deliberately allowed outside the frame so the quad
describes the *whole physical panel*. This is the only existing artefact that
encodes anything about the room's 3D layout, and it is a genuine starting
constraint for camera solving.

### The behaviour engine
`src/presenter/behavior/` is renderer-agnostic and time-driven. It already
satisfies the specification's global-clock requirement: it is advanced by
measured elapsed seconds and produces an `AvatarPose`. It knows nothing about
cameras, which is exactly right - it can drive a 3D rig unchanged.

### The wall material system
`tools/wall_material.py` establishes the design decisions the specification
asks to preserve in 3D: a global wall coordinate system, dark base, black gaps,
fixed slat pitch. Its comment that texturing fragments independently gives each
its own phase "the classic give-away" is the same argument the specification
makes about props moving between cameras.

## What exists and is *not* reusable for multicam

- `src/presenter/render/environment.py` - procedurally drawn 2D room. Superseded
  by the master-frame architecture and irrelevant to 3D.
- `src/presenter/render/cameras.py` + `config/cameras.yaml` - the seven-button
  rig from the previous phase. It selects between **separate images**, which is
  precisely the architecture this specification rejects. The UI (button row,
  mouse dispatcher, still path) is reusable; the model behind it is not.
- `tools/generate_cameras.py` - generates a camera as an independent image.
  Contradicts the specification and should be retired, not extended.
- `assets/cameras/cam4..cam7.png` - separately generated angles. Different room,
  different man. Kept only as mood reference.

## Measured performance and budget

From `docs/progress.md`, re-verified this session:

| | |
|---|---|
| LivePortrait frame time | 74.7 ms (13.4 FPS) with `torch.compile` |
| GPU steady state | 58-73 % utilisation, 76-78 W |
| VRAM, renderer alone | ~1.25 GB reserved |
| VRAM, whole machine | 16.3 GB total, frequently **under 2 GB free** - other applications (a 3D editor, several browsers, an LLM server) routinely hold 13+ GB |
| Blender EEVEE first render | 26.4 s including shader compilation, 320x180 |

The VRAM situation is the single most important practical constraint and it is
not under our control. `tools/generate_scene.py` already has a `vram_preflight`
that reports and adapts rather than terminating anything; any 3D renderer must
do the same.

## The gap, stated plainly

To satisfy "one world, seven projections" the project needs, and does not have:

1. A world coordinate system and a scene graph.
2. Room geometry - walls, desk, chair, monitors, mic, shelving - as meshes.
3. A physically-parameterised camera model (sensor, focal length, f-stop, focus).
4. A lighting rig defined in world space.
5. **A 360-degree-complete human** with consistent skull, ears, hair and body,
   drivable by the existing `AvatarPose`.

Items 1-4 are ordinary engineering and are achievable now. **Item 5 is the hard
one**, and the architecture document treats it separately rather than pretending
it is the same size of problem.
