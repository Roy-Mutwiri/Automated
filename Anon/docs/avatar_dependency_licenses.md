# Avatar reconstruction: dependency and licence audit

Audit date: 2026-09-02. Licensing is a first-class architecture criterion here,
not a footnote, because the project policy is explicit: **no non-commercial
component anywhere in the runtime.**

**This is an engineering audit, not legal advice.** Anything marked
`BLOCKED_PENDING_LICENSE_REVIEW` needs a human decision before it can influence
production.

## The distinction that the whole audit turns on

The specification was right to warn against oversimplifying SMPL-X. There are
two different licences and they are not interchangeable:

| | Contents | Licence | Commercial |
|---|---|---|---|
| [**SMPL-X BODY**](https://smpl-x.is.tue.mpg.de/bodylicense.html) | a 3D mesh, a skeleton rig, pose blendshapes, dynamic blendshapes | **CC BY 4.0** | **yes**, with attribution |
| [**SMPL-X MODEL**](https://smpl-x.is.tue.mpg.de/modellicense.html) | the above **plus the shape blendshapes** and the tools to create bodies with them | non-commercial research | **no** - commercial via Meshcapade |

SMPL-X BODY is explicitly "a subset of SMPL-X Model which **excludes the shape
blendshapes** or the tools to create 3D bodies using them".

That line falls in a very convenient place for us, and it splits our
architecture cleanly:

- **Reconstruction** estimates a person's *shape* - the β parameters. That is
  precisely what the shape blendshapes are for, so any reconstruction model
  that fits SMPL-X shape needs the **full Model**, and is therefore
  non-commercial.
- **Runtime posing** does not. Our subject's shape will be baked into whatever
  geometry we end up with; the runtime only has to *pose* a fixed body. That is
  mesh + skeleton + pose blendshapes, which is **CC BY 4.0**.

So "uses SMPL-X" is not one licensing situation, exactly as the specification
warned. It is at least two, and the answer depends on *which stage*.

## Candidate matrix

| | Code licence | Weights licence | Body model needed | Commercial status |
|---|---|---|---|---|
| **LHM** | Apache-2.0 (repo) | **not stated to be Apache** - separate release on HF/ModelScope | SMPL-X, shape-fitting | `BLOCKED_PENDING_LICENSE_REVIEW` |
| **HumanNOVA** | not stated on the repo page | not stated | SMPL assets, downloaded separately | `BLOCKED_PENDING_LICENSE_REVIEW` |
| **HumanLift** | Apache-2.0 | from CST Drive, terms unstated | SMPL-X; also **Wan2.1-14B** and LHM | `BLOCKED_PENDING_LICENSE_REVIEW` |
| **UIKA** | no code released | none | FLAME (MPI, same posture as SMPL) | not assessable |
| **FiCA** | not released | none | FLAME | not assessable |
| **FHAvatar** | not released | none | FLAME | not assessable |
| **IDOL** (CVPR 2025) | to audit | to audit | SMPL-X | not yet audited |

### Why every row above is blocked rather than merely "probably fine"

The specification's warning is exactly right and worth restating: **a repository
licence badge does not cover the model weights.** Apache-2.0 on the source tree
says nothing about a checkpoint hosted on HuggingFace or a Chinese cloud drive,
and it says nothing about the body model the code loads at runtime. LHM and
HumanLift both show Apache-2.0 on the repository and both require assets that
are not covered by it.

There is a second, sharper issue that is easy to miss: **if creating an artefact
requires a non-commercial-licensed tool, the artefact itself may be
encumbered.** The SMPL-X Model licence prohibits using the software "to train
methods, algorithms or neural networks" for commercial use and prohibits
"production of other artifacts for commercial purposes". A 3D human produced by
fitting SMPL-X shape blendshapes is plausibly such an artefact. That is a
question for a human, not for me, and it is why reconstruction outputs are
quarantined in `research/` rather than exported to `assets/`.

## Our own stack, for comparison

Recorded because the audit is only useful against a baseline.

| Component | Licence | Commercial |
|---|---|---|
| LivePortrait code | MIT | yes |
| LivePortrait `landmark.onnx` | with the repo | yes |
| **InsightFace** | non-commercial | **removed from this project** for that reason |
| DeepLabV3 weights (torchvision) | BSD-style | yes |
| SDXL base / SDXL inpainting | CreativeML Open RAIL++-M | yes |
| Blender / `bpy` | **GPL-3.0** | yes, but see below |
| MediaPipe | Apache-2.0 | yes |

**Blender's GPL is worth a decision of its own before it hardens.** Using
Blender as a tool to produce assets is unencumbered - rendered images are ours.
But `bpy` is Blender-as-a-library, and importing it into the application
process makes the GPL question live for the code that imports it. The current
architecture already keeps rendering in a **subprocess**, which was chosen for
dependency isolation and turns out to matter here too. It should stay that way,
and if Blender ever becomes a runtime rather than an offline tool, that needs
its own review.

## What is commercially clean today

- Camera geometry, room model, floorplan, validation - all ours.
- The frozen segmentation (`avatar_rgba.png`) - DeepLabV3 (BSD) + OpenCV.
- The approved identity plate - SDXL under RAIL++-M, already documented.
- The proxy human - ours, and worthless as a product.

**We currently have no commercially clean path to a photoreal reconstructed
human.** That is the honest state, and it is a real finding rather than a
temporary gap: every released single-image human reconstruction system found in
this survey is built on an MPI parametric body or head model.

## The routes that could be clean

1. **Reconstruct research-only, ship nothing from it.** Use LHM/HumanNOVA in
   `research/noncommercial/` purely to answer "how much identity is recoverable
   from this plate", then rebuild the winning approach on clean foundations.
   Answers the technical question without creating a shippable artefact.
2. **Commercial SMPL-X licence via Meshcapade.** Turns most of the field green
   at once. Cost unknown. Probably the shortest path to a shippable photoreal
   human if the project has a budget.
3. **Clean-room rig.** Reconstruct *geometry* only from a clean source, then rig
   it ourselves in Blender - Rigify, custom armature, shape keys, ARKit-style
   blendshape curves. Avoids MPI models in the runtime entirely. More work, and
   the identity quality then depends on a reconstruction step we still need.
4. **Buy a licensed photoreal human asset** and re-texture toward the approved
   identity. Commercially clean by construction, identity match uncertain.

Routes 2 and 4 are procurement decisions, not engineering ones.

## Machine constraints that interact with licensing

From `tools/check_reconstruction_gpu.py`:

- torch 2.11.0+cu128 **does** carry sm_120 kernels and they launch.
- **No CUDA toolkit and no MSVC.** Every Gaussian-splatting candidate needs
  `diff-gaussian-rasterization` and `simple-knn` compiled from source, per
  project, and no prebuilt sm_120 wheels exist for them.
- WSL2 is enabled with **no distro installed**; ~709 GB free.

So even the research-only route costs a compiler toolchain first - roughly 6 GB
of Visual Studio Build Tools plus ~3 GB of CUDA Toolkit on Windows, or an
Ubuntu distro plus toolchain in WSL. That cost lands *before* any model weights,
which is exactly why it is being reported before anything is downloaded.
