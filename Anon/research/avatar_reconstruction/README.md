# Avatar reconstruction research sandbox

**Nothing in here is production.** This tree is disposable. The application
must only ever consume an *exported artefact* from `outputs/`, and only after
technical validation, identity validation and licence validation.

## Hard rules

- No package installed for an experiment goes into the application venv
  (`../../.venv`). That venv already carries a `bpy`-imposed numpy pin and a
  Blackwell-specific torch build; a second dependency negotiation in it would
  break the working renderer.
- Nothing here is imported from `src/presenter/`.
- `noncommercial/` holds anything whose licence forbids shipping. It is
  research reference only and is never a source of production assets.

## Layout

    inputs/        the frozen reconstruction input (see below)
    env/           per-candidate isolated environments (never committed)
    experiments/   one directory per candidate, with its own LICENSE_NOTES.md
    outputs/       exported artefacts, each with provenance and licence status
    ../noncommercial/   research-only candidates, marked DO NOT SHIP

## Frozen inputs

`inputs/avatar_rgba.png` and `inputs/avatar_mask.png` are copies of the frozen
segmentation in `assets/reference/`. Produced by `tools/segment_identity.py`
from the locked plate. **Do not regenerate** unless a reconstruction candidate
reveals a concrete matte defect - the specification freezes them deliberately so
that differences between candidates are differences between candidates, not
differences in their input.

## Machine constraints, measured

Run `python tools/check_reconstruction_gpu.py --build-test` for the current
state. As of 2026-09-02:

| | |
|---|---|
| GPU | RTX 5080 Laptop, compute capability **12.0 (sm_120)**, 16.3 GB |
| Driver | 610.88 |
| torch | 2.11.0+cu128, arch_list includes sm_120, kernels launch |
| **nvcc** | **not installed** |
| **MSVC / Visual Studio** | **not installed** |
| WSL | WSL2 available, **no distro installed** |
| Disk | ~709 GB free |

The last three rows are the ones that decide the plan. Every Gaussian-splatting
pipeline needs `diff-gaussian-rasterization` and `simple-knn`, which are
compiled from source per project; there are no prebuilt sm_120 wheels for them.
So a 3DGS candidate cannot be built on this machine today without first
installing a compiler toolchain, on Windows or in WSL.
