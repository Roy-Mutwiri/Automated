# Avatar rendering: research and model selection

Research date: 2026-09-02. Target hardware measured on this machine, not quoted
from a paper.

## Hardware constraint that drives everything

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 Laptop, 16 GB (14.7 GB free) |
| Compute capability | **12.0 (Blackwell, sm_120)** |
| CPU / RAM | AMD Ryzen AI 9 HX 375, 12C/24T, 31.3 GB |
| PyTorch | 2.11.0+cu128, `arch_list` includes sm_120 — verified working |
| Measured | 38 TFLOPS fp16 matmul; conv2d 512² in 0.195 ms |

**sm_120 is the single most important constraint and it invalidates most
installation instructions found online.** Blackwell needs CUDA 12.8+ and a
PyTorch built with sm_120 kernels. Every repo below that pins `torch==2.0.1`,
`torch==2.3.0`, or a cu118/cu121/cu124 wheel will install, import, and then
fail at the first CUDA kernel launch with `no kernel image is available for
execution on the device`. Pinned requirements files must be overridden, not
followed.

This also rules out convenience paths: FasterLivePortrait's installation-free
Windows package targets CUDA 12.2, and PersonaLive's README notes xformers is
not yet compatible with RTX 50-series.

## Candidates evaluated

| Project | Approach | Real-time? | Driven by | License | Verdict |
|---|---|---|---|---|---|
| [LivePortrait](https://github.com/KwaiVGI/LivePortrait) | Implicit-keypoint warping + decoder | 12.8 ms/frame on RTX 4090 (~78 FPS) | **Explicit motion params** | MIT (code) | **Selected** |
| [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) | LivePortrait + TensorRT/ONNX | 30+ FPS on RTX 3090 incl. pre/post | same | MIT | Optimisation path, later |
| [PersonaLive](https://github.com/GVCLab/PersonaLive) (CVPR 2026) | Diffusion, live-streaming oriented | claimed, unbenchmarked | driving video only | Apache-2.0 | Rejected for now |
| [EmbodiedHead](https://arxiv.org/pdf/2604.17211) | Rectified-flow DiT + differentiable renderer | 4 sampling steps | audio + listen/speak state | paper | Watch closely |
| [StreamAvatar](https://arxiv.org/pdf/2512.22065) | Streaming diffusion | claimed | audio | paper | Watch |
| [VASA-3D](https://arxiv.org/pdf/2512.14677) | Gaussian head avatar from one image | — | audio | paper | Watch |
| [SyncTalk++](https://arxiv.org/pdf/2506.14742) | Gaussian splatting | efficient | audio | paper | Per-identity training needed |
| MuseTalk / LatentSync / Wav2Lip | Mouth-region lip-sync | yes | audio | varies | **Phase 2**, mouth only |
| SadTalker / AniPortrait / Hallo / EchoMimic | Diffusion talking head | no (seconds/frame) | audio | varies | Rejected: not real-time |

## Why LivePortrait

The decisive property is not speed — several candidates claim real-time. It is
**how motion is specified.**

LivePortrait's motion representation is explicit and low-dimensional:

```
x_d = s_d · (x_c · R_d + δ_d) + t_d
```

where `x_c` is canonical keypoints, `R` head rotation, `δ` expression
deformation, `s` scale, `t` translation — plus separate eye and lip retargeting
modules that adjust those regions from scalar controls.

Every other real-time candidate is driven by **a driving video or an audio
track**. This project has neither: it needs a face that is convincingly alive
while *silent and unprompted*, potentially for hours. A video-driven pipeline
would require a driving clip, and any finite clip loops — which the brief
forbids outright, and which is the failure mode most likely to be noticed on a
long stream.

Because LivePortrait consumes motion *parameters*, the behaviour engine can
synthesise `R`, `δ`, `s`, `t` procedurally and there is no driving video at
all. Loop-freedom becomes structural rather than something to be engineered
around. That is worth more here than any FPS advantage.

Secondary reasons: MIT-licensed code, small model (256² internal, comfortably
inside 16 GB), an existing TensorRT acceleration path via FasterLivePortrait if
needed, and active maintenance.

## Licensing — action required before public use

- **LivePortrait code: MIT.** No commercial restriction.
- **InsightFace models: non-commercial research only.** LivePortrait uses
  InsightFace for face detection/alignment by default. This is the one real
  licensing hazard in the stack.
- **Mitigation:** FasterLivePortrait documents MediaPipe (Apache-2.0) as a
  drop-in replacement for InsightFace detection. If this is ever deployed
  commercially, that swap is mandatory. Flagged now rather than discovered
  later.

## Not yet decided

The lip-sync engine is deliberately unchosen. MuseTalk, LatentSync and the 2026
diffusion approaches all target the mouth region, and the renderer interface
(`render/base.py`) is narrow specifically so that decision can be made on
benchmark evidence when audio arrives, without disturbing anything upstream.

## Sources

- LivePortrait — https://github.com/KwaiVGI/LivePortrait · paper https://arxiv.org/pdf/2407.03168
- FasterLivePortrait — https://github.com/warmshao/FasterLivePortrait
- PersonaLive (CVPR 2026) — https://github.com/GVCLab/PersonaLive
- EmbodiedHead — https://arxiv.org/pdf/2604.17211
- StreamAvatar — https://arxiv.org/pdf/2512.22065
- VASA-3D — https://arxiv.org/pdf/2512.14677
- SyncTalk++ — https://arxiv.org/pdf/2506.14742
- LLIA low-latency interactive avatars — https://arxiv.org/pdf/2506.05806
- Awesome-Talking-Head-Synthesis — https://github.com/Kedreamix/Awesome-Talking-Head-Synthesis
