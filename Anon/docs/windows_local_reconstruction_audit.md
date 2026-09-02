# Can any good reconstruction run on our Windows box as-is?

Audited 2026-09-02, as the backup option. **Constraint: no nvcc, no MSVC, no
WSL** - so nothing that compiles a CUDA extension, and nothing that needs a
C++ toolchain. Our existing environment is torch 2.11.0+cu128 with sm_120.

## Answer: no competitive candidate

| Candidate | Blocker on this machine |
|---|---|
| **LHM** | `diff-gaussian-rasterization`, `simple-knn`, `pytorch3d` - all compiled from source |
| **LHM++** | the same three, **plus** `pointops` (source build), `spconv`, `torch_scatter` |
| **HumanLift** | same 3DGS stack, plus a 14B diffusion model |
| **HumanNOVA** | conda + CUDA 12.1 setup script; not animatable in any case |
| **Deep3DFaceRecon** | `nvdiffrast` - compiled |
| **VRN (PyTorch)** | genuinely pure PyTorch, but it is a 2017 volumetric face regressor. Low resolution, no hair, no body. Would not beat MPFB meaningfully |

Generic image-to-3D models (TripoSR, InstantMesh, TRELLIS, Hunyuan3D) mostly
pull in `nvdiffrast`, `torchmcubes` or custom rasterizers too, and none is
human-specific - which is the whole reason a human-specific model was chosen.

## The conclusion, and it is not a close call

There is no way to get a *competitive* single-image human reconstruction running
on this machine without a compiler. The only pure-PyTorch options are old enough
that they would not clear the bar MPFB already failed, and the instruction was
explicit: do not downgrade quality merely to avoid a remote machine.

So: **remote reconstruction stands.** `research/lhm_remote/` is ready.

One thing this audit *did* change: LHM++'s 8 GB requirement means the remote box
can be small and cheap rather than a 24 GB card. That is a better outcome than
anything the local path could have offered.
