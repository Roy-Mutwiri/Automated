# Multi-view digital human: technology research

Research date: 2026-09-02. Evaluated against *our* requirements, not against
novelty. Our requirements are unusual in one specific way that eliminates most
of the field, so that is stated first.

## The constraint that decides almost everything

**We have one photograph.**

Not a capture rig. Not a video. Not a turntable. One diffusion-generated
1344x768 image of a man in a room, and a hard requirement that Cameras 4-7 show
his back, his crown, and the parts of the room the photograph never contained.

Every technique below was evaluated on the question: *what does it need as
input, and do we have it?*

| Technique class | Needs | Have it? |
|---|---|---|
| 3D Gaussian Splatting (scene) | dozens-hundreds of calibrated views | **no** |
| Human Gaussian avatars (GauHuman, Animatable Gaussians, TaoAvatar) | monocular *video* of the subject moving, or multi-view capture | **no** |
| NeRF / neural rendering | multi-view | **no** |
| MetaHuman | a face scan or reference photos, plus Unreal | photos yes, Unreal no |
| FLAME / SMPL-X fitting | one image is enough | **yes** |
| Traditional 3D modelling | reference images and labour | **yes** |
| Single-image scene lifting (Pano2Room, PanoDreamer) | one panorama | we have a *narrow* view, not a panorama |

## Human reconstruction

**Human Gaussian avatars are the state of the art for realism and are the wrong
tool here.** [GauHuman](https://arxiv.org/html/2311.17113v2),
[Animatable Gaussians](https://arxiv.org/pdf/2311.16096) and
[TaoAvatar](https://arxiv.org/pdf/2503.17032) all reconstruct from monocular
*video* of the actual person or from multi-camera rigs. They render in real time
once trained, which is attractive, but training input we do not have is still
input we do not have. Our subject does not exist and never moved in front of a
camera.

**[SVAD](https://arxiv.org/pdf/2505.05475) is the closest match to our situation
and is worth naming precisely because it is the honest answer to "single image
to 3D avatar".** Its pipeline is: single image -> video diffusion to synthesise
the subject moving -> use those synthetic frames as pseudo multi-view data ->
train a 3DGS avatar. It reports better identity preservation across novel views
than other single-image methods and renders in real time. That is exactly our
problem statement.

It is also a research pipeline: video diffusion over a subject, a data
augmentation stage, then avatar training. It is weeks of work, substantial VRAM,
and Linux-leaning tooling, on a machine that frequently has under 2 GB of VRAM
free. It is the right long-term answer and the wrong milestone-1 answer.

**Mesh-based parametric models (FLAME for the head, SMPL-X for the body) work
from one image** and give complete, animatable, 360-degree geometry immediately.
Their weakness is documented and directly relevant to us: mesh methods
"struggle with modeling complex hairstyles", and hair is "difficult to handle
due to topological constraints and difficulty in handling opacity"
([survey](https://arxiv.org/pdf/2407.17418),
[MeGA](https://arxiv.org/pdf/2404.19026)). Our subject has thick dark hair, a
full beard and headphones. Camera 5 is a rear view whose entire content is that
hair.

So: parametric meshes give us *consistency* immediately and *photorealism* not
at all. Gaussian methods give photorealism but need input we do not have.

**Hybrid mesh+Gaussian head avatars** ([MeGA](https://arxiv.org/pdf/2404.19026),
[GaussianAvatars](https://openaccess.thecvf.com/content/CVPR2024/papers/Qian_GaussianAvatars_Photorealistic_Head_Avatars_with_Rigged_3D_Gaussians_CVPR_2024_paper.pdf))
are where the field has landed - mesh for controllable geometry, Gaussians for
hair and skin detail. Same input problem.

## Environment reconstruction

[Pano2Room](https://arxiv.org/pdf/2408.11413) and
[PanoDreamer](https://arxiv.org/html/2412.04827v1) reconstruct a full 3D indoor
scene from a **single panorama**, by estimating depth, inpainting occluded
regions and converting to a mesh or 3DGS field. If we had a 360 panorama of this
room, this would be a strong candidate.

We have a ~40 mm view of one wall. Lifting that to a room means inventing three
walls, the ceiling, the floor behind camera, and everything behind the desk -
which is *hallucination*, and the specification's own Hidden Geometry Rule says
hidden geometry should be **designed once and locked**, not hallucinated per
camera. Designing it deliberately in a modelling package is both more honest and
more controllable than asking an inpainter to guess it differently every run.

**Conclusion for the room: model it. Do not reconstruct it.** The room is
simple, rectilinear, and made of exactly the kinds of objects (slat wall, desk,
monitors, chair, boom mic) that are quick to model and trivially consistent once
modelled.

## Renderer / engine

**MetaHuman + Unreal** is the industry answer for photoreal digital humans with
physical cine cameras, and 5.7 added a
[Python/Blueprint API for batch automation](https://dev.epicgames.com/documentation/metahuman/metahuman-5-7-release-notes).
Its Cine Camera exposes exactly the filmback / focal length / aperture / focus
model the specification asks for. Rejected for now on two grounds, both
practical rather than technical: Unreal is not installed and is a very large
dependency to add mid-project, and MetaHuman gives us *a* photoreal human, not
*this* man - matching the approved Camera 1 identity through MetaHuman Creator
is its own multi-day task with an uncertain result.

**Blender via the `bpy` PyPI module** is installed and verified this session:
5.0.1, Windows, EEVEE rendering headless in-process, 26.4 s for a first render
including shader compilation. It gives us, for free:

- a real scene graph and world coordinate system
- physically-parameterised cameras: `sensor_width`, `lens` in mm, `dof.aperture_fstop`, `focus_object` - a direct match for the specification's camera model
- one lighting rig observed by all cameras
- real shadows, occlusion and parallax by construction
- materials shared across cameras by definition

[The `bpy` module](https://pypi.org/project/bpy/) is intended for exactly this -
Blender as a library in a pipeline. Note the known caveat that
[EEVEE can fail on headless machines](https://blog.cg-wire.com/blender-programmatic-rendering/)
depending on GPU/driver context; it works here, and Cycles is the fallback.

## What the research actually settles

1. **The room should be modelled, not reconstructed.** Single-image scene
   lifting needs a panorama we do not have, and the missing geometry is better
   designed than guessed.
2. **The human is the hard problem and cannot be solved at the same time as the
   room.** Every method that would give us a photoreal 360-degree version of
   *this specific man* requires input we do not possess.
3. **Blender is the correct vehicle for the canonical world** on this machine
   today, and it does not preclude replacing the human representation later -
   the human is one object in the scene graph.
4. **Nothing in the literature lets us skip the honest trade:** we can have
   geometric consistency now with reduced human fidelity, or photoreal humans
   from one camera angle, and the path between them is a real research project
   (SVAD-shaped), not a configuration change.
