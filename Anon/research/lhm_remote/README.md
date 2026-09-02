# LHM remote reconstruction package

**RESEARCH / PERSONAL USE ONLY.** Self-contained. Copy this directory to a
disposable Linux GPU machine, run two scripts, get one avatar back.

Nothing here runs on, or changes, the Windows workstation. LHM is a one-time
offline reconstruction tool; the streaming application stays Windows + Blender.

### The path to run

LHM++ is now the target, not LHM. `environment_manifest.json` pins the exact
repository commit, model and stack; read it before changing anything.

    bash setup_lhmpp.sh     # documented stack: python 3.10 / CUDA 12.1 / torch 2.3.0
    bash run_lhmpp.sh       # single image -> canonical T-pose 3DGS .ply
    # look at outputs/lhmpp_identity_turntable.png

**Do not download SMPL-X first.** `run_lhmpp.sh` deliberately does not fetch it,
because whether `LHMPP-700M-SMPLX-FREE` actually needs it is an open question
the run itself answers — see `open_question_to_resolve_on_the_box` in the
manifest. Register with MPI only if the run demands it.

The original LHM path (`setup.sh` / `run_reconstruction.sh`) is kept as a
fallback and still works; it needs SMPL-X up front.

## What you must download yourself

**I have deliberately not automated this, and the setup script does not fetch
it.** It is a licence acceptance, and that is the project owner's decision.

LHM expects the body model at:

    LHM/pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz

Get it from **https://smpl-x.is.tue.mpg.de** → Download → *SMPL-X v1.1*
(`models_smplx_v1_1.zip`), register, accept the licence, and take
`SMPLX_NEUTRAL.npz` out of `models/smplx/`.

Put it in `private_models/` on this machine. `run_reconstruction.sh` copies it
into place and prefers it over anything else.

**A note worth reading before you bother:** LHM's own
`LHM_prior_model.tar` appears to bundle `human_model_files`, so the pipeline may
run without you downloading anything. If it does, that is Alibaba redistributing
MPI's model, not MPI licensing it to us. Under the current personal/non-commercial
policy either route is usable, but obtaining it officially is cleaner and is what
this package is set up to prefer. It also matters later: SMPL-X MODEL is
non-commercial, and if this project is ever commercialised that dependency has to
be revisited (see `docs/avatar_dependency_licenses.md`).

`private_models/` and `outputs/` are git-ignored. Do not commit either.

## Remote GPU requirements

| | |
|---|---|
| GPU | NVIDIA, **24 GB recommended**, 16 GB workable |
| Why | LHM-1B-HF is the best half-body model; the published memory-saving path runs the full pipeline on 14 GB and LHM-MINI on 16 GB. 24 GB removes the question. |
| Compute capability | Anything from Ampere up. Blackwell is handled - see below. |
| CUDA toolkit | Must include `nvcc`; most GPU cloud images do |
| Disk | **~60 GB** — checkpoints, prior models, torch, and three source-built CUDA extensions |
| RAM | 32 GB comfortable, 16 GB likely enough |
| OS | Linux. Ubuntu 22.04 or 24.04 |
| Python | 3.10 (LHM's tested version) |
| Time | ~30–45 min setup, dominated by compiling; inference is 1.4–6.6 s |

A single A100 40 GB, L40S, or a 4090 is more than enough. This is one run.

## The one non-obvious engineering decision

`setup.sh` **does not blindly follow LHM's `torch==2.3.0` pin.**

It reads the GPU's compute capability first:

* **sm_89 and below** (Ampere, Ada) → honour the pin exactly. It works, and
  deviating would only invite new problems.
* **sm_120 and above** (Blackwell) → override to current torch on cu128. The
  2.3.0 wheels contain no sm_120 kernels, so that stack installs cleanly,
  imports cleanly, and then dies at the first kernel launch with *"no kernel
  image is available for execution on the device."*

That failure is silent until runtime, which is why it is decided up front from
`compute_cap` rather than discovered later. Patch 01 also strips the torch pins
out of `requirements.txt` so pip cannot quietly downgrade them back.

## Files

| | |
|---|---|
| `environment_manifest.json` | **the pinned truth**: repo commit, model, stack, GPU policy, and the blanks to fill in after the run |
| `setup_lhmpp.sh` | **primary** — builds the LHM++ environment at the pinned commit; refuses Blackwell unless `FORCE_MODERN=1` |
| `run_lhmpp.sh` | **primary** — one image, one run, canonical T-pose `.ply`; fetches no SMPL-X |
| `verify_environment.py` | **run before downloading weights** — proves CUDA, torch arch, and every compiled dependency; `--lhmpp` adds `pointops`/`spconv`/`torch_scatter`; `--build` compiles a test kernel |
| `inspect_ply.py` | reports Gaussian count and confirms a `.ply` really is 3DGS |
| `setup.sh` | fallback — the original LHM environment |
| `run_reconstruction.sh` | fallback — one model, one image, one run; exports mesh |
| `export_results.py` | renders the turntable, packages the outputs |
| `patches/apply_patches.py` | the minimal recorded edits to upstream |
| `patches/APPLIED_PATCHES.md` | written on each run — what we changed vs upstream |
| `inputs/` | the frozen identity inputs (below) |
| `outputs/` | results (git-ignored) |
| `private_models/` | your SMPL-X (git-ignored) |

## Inputs

Frozen, from the segmentation stage. **Do not regenerate** unless a
reconstruction failure points at the matte.

| | |
|---|---|
| `avatar_identity_camera1.png` | the approved plate; the thing every result is judged against |
| `avatar_rgba.png` | **the reconstruction input** — human only, alpha matted, chair excluded, mic and cable removed and the shirt inpainted behind them |
| `avatar_mask.png` | the matte alone |

Headphones are deliberately kept: they are worn, and part of the silhouette.
**If reconstruction fuses them into the skull, hair or ears, that is the signal
to try a second input with them removed** — and it is worth checking explicitly
in the turntable rather than assuming.

## Model choice is not about size

LHM's table splits models by input requirement, and our plate is a **half body**:

| Model | Input | Use |
|---|---|---|
| LHM-1B-HF | half & full | **default here** |
| LHM-500M-HF | half & full | faster alternative |
| LHM-MINI | half & full | 16 GB fallback |
| LHM-500M / LHM-1B | **full body only** | wrong for our plate at any size |

`run_reconstruction.sh` refuses a non-HF model rather than letting it fail
obscurely.

## The gate

Look at `outputs/lhm_identity_turntable.png` — 0, ±20, ±40 degrees — **before
anything else**.

The bar is not "is it a good 3D human". It is: *is this recognisably the same
fictional man as `inputs/avatar_identity_camera1.png`?* MPFB scored about 2/10
on that. We need a large improvement, not a marginal one.

If it fails: stop, report, do not export or integrate.

If it passes: bring `outputs/avatar_v01/` back to Windows. The application never
needs LHM installed — the reconstruction is just an asset, consumed through an
adapter, and rendered through the existing canonical camera world.

## What happens if the output is Gaussians rather than a mesh

Do not convert it badly to force it into Blender. Two options, decided on
merit:

* **A** — extract or convert a mesh, if the geometry survives it.
* **B** — render the human through its own Gaussian renderer using the *same*
  camera matrices as Blender, then depth-composite the two.

Option B keeps "one world, seven cameras" intact and is likely to preserve more
quality. `config/cameras.yaml` already stores position, aim, sensor size, focal
length and f-stop per camera, which is everything a converter needs.

---

## Why LHM++ rather than LHM

Researched after this package was first written, and now the decided path. `LHM-plusplus` (March 2026, same
team) changes two things that matter to us, and both point the same way.

| | LHM | LHM++ (`LHMPP-700M-*`) |
|---|---|---|
| VRAM | 16–24 GB | **8 GB** |
| Single-view inference | 1.4–6.6 s | **0.79 s** |
| 3DGS output | internal | **standard `.ply`, incl. canonical T-pose** |
| SMPL-X | required | **`LHMPP-700M-SMPLX-FREE` variant exists** |
| Build deps | 3 CUDA extensions | those **plus** `pointops`, `spconv`, `torch_scatter` |

**Standard `.ply` at canonical T-pose is the important one.** It is exactly the
asset the "Gaussian case" plan in this README wants: a pose-neutral 3DGS file we
can either convert to mesh or render through its own renderer using our camera
matrices. LHM's own export is less directly consumable.

**`LHMPP-700M-SMPLX-FREE` may remove the licence-gated download entirely** -
worth confirming before you spend time registering with MPI. Do the environment
build first; if that variant runs without `human_model_files`, the SMPL-X step
disappears. If it still needs it for input pose estimation, fall back to the
instructions above.

Its dependency set is *heavier*, not lighter, so it remains a remote-only job.
Since the box is disposable, that costs nothing but the 8 GB VRAM requirement
makes the box considerably cheaper.

Model: `Damo_XR_Lab/LHMPP-700M-SMPLX-FREE` on ModelScope. Note the README's
caveat that hub weights for the `PixelShuffle` default were still pending at the
time of writing.

**Decided order:** `LHMPP-700M-SMPLX-FREE` first, `LHM-1B-HF` kept in this
package as the fallback if LHM++ misbehaves.

`PixelShuffle` is the newer upstream default but its hub weights were still
marked pending, so it is not a candidate for a first run. Do not substitute it.

## After the run

Fill in `recorded_after_the_run` in `environment_manifest.json` — GPU, peak
VRAM, wall time, Gaussian count, and above all whether `human_model_files` was
required. A second run that behaves differently is impossible to diagnose
without it.

Bring back only `outputs/`. Nothing else on that box matters, and the
checkpoints must not be redistributed.
