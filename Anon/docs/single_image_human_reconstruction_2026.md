# Single-image human reconstruction: 2026 candidates

Research date: 2026-09-02. **This supersedes the conclusion in
`multicam_technology_research.md`, which was wrong.**

That earlier document concluded that a single image left only a multi-week
SVAD-shaped research route. That was correct for the papers it looked at and
incorrect as a statement about the field: there are now feed-forward
single-image systems with public code and public checkpoints, at least one of
which lists Windows support and fits 16 GB. The earlier conclusion was reached
by searching for *techniques* rather than for *released systems*, which is the
mistake.

## Scorecard

Assessed on practical viability, not paper claims. "Runnable here" is the column
that decides.

| | HumanNOVA | HumanLift | LHM / LHM-MINI | UIKA | FiCA | FHAvatar |
|---|---|---|---|---|---|---|
| Venue | CVPR 2026 Highlight | SIGGRAPH Asia 2025 | ICCV 2025 | CVPR 2026 | 2026 | CVPR 2026 |
| Single image | yes | yes | yes | yes | yes | **no** - few captures |
| Scope | full human | full body | full body | head | head | head + hair |
| **Animatable** | **no** - reconstruction only | yes (via LHM) | **yes**, SMPL-X driven | yes | yes, real-time | yes |
| Representation | mesh | 3DGS | 3DGS | 3DGS | 3DGS | 3DGS, strand hair |
| **Code public** | **yes** | **yes** | **yes** | **"coming soon"** | project page | paper |
| **Checkpoints public** | yes | yes (CST Drive) | yes (HF / ModelScope) | no | unclear | unclear |
| Code licence | unstated | Apache-2.0 | **Apache-2.0** | - | - | - |
| **Windows** | unstated, conda + `setup.sh` | unstated | **documented install guide** | - | - | - |
| VRAM | unstated | needs **Wan2.1-14B** | **MINI 16 GB**, 500M 24 GB, 14 GB memory-saving | - | - | - |
| CUDA | 12.1 | 11.7 era | 11.8 / 12.1 | - | - | - |
| Runtime | < 1 s | slow, offline | 1.4-6.6 s | - | 5 s | minutes |
| Blocking issue | **not animatable** | 14B model on a 16 GB card; manual Photoshop alignment step | CUDA/sm_120, SMPL-X licence | **no code** | synthetic-domain risk, code unclear | wrong input modality |

Sources: [HumanNOVA](https://github.com/HumanNOVA/HumanNOVA) ·
[HumanLift](https://github.com/IGLICT/HumanLift) ·
[LHM](https://github.com/aigc3d/LHM) ·
[UIKA](https://zijian-wu.github.io/uika-page/) ·
[FiCA](https://kim-youwang.github.io/FiCA) ·
[FHAvatar](https://arxiv.org/abs/2603.23345)

## The finding that matters most, and it is not on the scorecard

**Every viable candidate is SMPL or SMPL-X based, and SMPL-X is
non-commercial-only.**

The [SMPL-X model licence](https://smpl-x.is.tue.mpg.de/modellicense.html)
grants use "for the sole purpose of performing non-commercial scientific
research, non-commercial education, or non-commercial artistic projects", and
explicitly prohibits "incorporation in a commercial product" or "use in a
commercial service". Commercial licensing is separate, via Meshcapade.

This project has already made a deliberate decision on exactly this question.
`README.md` records that InsightFace was removed because its "models are
licensed for non-commercial research only", and `assets/PROVENANCE.md` states
with some pride that "the runtime ends up with **no non-commercial component
anywhere in it**." Every model above would put one straight back.

Three honest positions, and this is the user's call rather than mine:

1. **Research/prototype only.** Use SMPL-X now, prove the identity gate, accept
   that shipping needs a commercial licence from Meshcapade or a replacement.
2. **Licence it.** Cost and terms unknown; a real option if this becomes a
   product.
3. **Avoid SMPL entirely.** That rules out essentially the whole current
   single-image human reconstruction field. Head-only systems that avoid it
   exist but they are FLAME-based, and FLAME carries the same MPI terms.

Note the licence gates *use*, not download - and the body model files require
registration and manual acceptance on the MPI site, which is not something I
should click through on someone else's behalf.

## Recommendation

**Test LHM-MINI first, not HumanNOVA.** The instruction was to start with
HumanNOVA as the full-human baseline, and I would push back on that for one
concrete reason: HumanNOVA is described as reconstruction, **not animatable**.
Requirement 47 asks for a rig interface the behaviour terminal can drive, and a
static mesh cannot provide one. It remains valuable as a *quality comparator* -
a CVPR Highlight feed-forward reconstruction in under a second is a good
yardstick for how much identity a model can recover from one image - but it
cannot be the canonical human.

LHM is the only candidate that is simultaneously: animatable, Apache-2.0 code,
public checkpoints that auto-download, documented on Windows, and sized for
16 GB (`LHM-MINI`). HumanLift is interesting and arguably higher quality, but it
depends on Wan2.1-14B for its multi-view synthesis stage and then uses LHM for
animation anyway - so LHM is the floor of that pipeline regardless, and the
right thing to characterise first.

Expected obstacle, already familiar to this project: LHM targets CUDA 11.8/12.1
and Python 3.10. This machine is Blackwell **sm_120** on cu128 with Python 3.11.
`README.md` documents precisely this trap - repos pinning older CUDA install
cleanly and then fail at the first kernel launch with "no kernel image is
available for execution on the device" - and documents the fix: override the
pins. This should be done in a **separate environment**, not the working one:
`bpy` already pinned numpy to 1.26.4 here, and a second dependency negotiation
in the same venv is asking for trouble.

## Input is prepared

`tools/segment_identity.py` produces the reconstruction input from the locked
plate:

- `assets/reference/avatar_rgba.png` - human only, alpha matted
- `assets/reference/avatar_mask.png`
- `renders/segmentation_debug.png`

Subject covers 27.6 % of frame. The gaming chair is excluded by seeding it as
background from DeepLab's chair channel before GrabCut, which is far more
reliable than hoping a colour model separates leather from a dark shirt. The
boom arm and its cable, which overlap him and therefore survive any
silhouette-based matte, are detected structurally (a median filter wide enough
to swallow a thin structure leaves broad fabric folds alone, so the difference
image *is* the thin structures regardless of whether their contrast runs bright
or dark) and the shirt behind them is inpainted - 14,472 px. A tube-shaped void
in a torso is worse than the tube, because a reconstruction model will invent
geometry to fill it.

Headphones are deliberately kept: they are worn, and they are part of the
silhouette the identity is judged on.

## What has NOT been done

No reconstruction has been run. The identity gate is not attempted, let alone
passed. The next step needs two decisions that are not mine:

1. **SMPL-X licence.** Nothing can proceed without the body model files, and
   accepting that licence is the user's decision.
2. **A separate environment**, several GB of weights, and a Blackwell/CUDA
   fight of unknown length.
