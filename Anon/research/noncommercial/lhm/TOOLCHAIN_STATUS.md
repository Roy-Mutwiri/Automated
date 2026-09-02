# LHM sandbox — toolchain status

**RESEARCH / PERSONAL USE ONLY. Not imported by the application.**

## Blocked, and on exactly one thing

WSL2 cannot create a distribution until the **VirtualMachinePlatform** Windows
feature is enabled, which needs administrator rights and a reboot. I have
neither.

Everything else on the WSL path is already done or is doable without further
admin:

| Step | State |
|---|---|
| WSL2 present, default version 2 | done |
| WSL kernel updated (was the first failure, 0x800701bc) | **done** - now 2.7.12.0, kernel 6.18.33.2 |
| Ubuntu Store package installed | **done** - CanonicalGroupLimited.Ubuntu 2604.1.75.0 |
| Register the distro (`ubuntu.exe install --root`) | **FAILS: 0x80370102** |
| VirtualMachinePlatform feature | **not enabled - needs admin + reboot** |
| Virtualisation in firmware | **already enabled** - VM Monitor Mode Extensions: Yes, SLAT: Yes |

The firmware line matters: this is not a BIOS trip. `HypervisorPresent` is False
only because the Windows feature is off.

## The command needed

In an **administrator** PowerShell, then reboot:

    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

After the reboot, nothing further needs elevation - the distro registers as
root and every package inside it installs without touching Windows.

## Why WSL rather than native Windows

Both need administrator rights, so WSL is not the more expensive option:

* **Native Windows** needs MSVC Build Tools for `cl.exe`, which torch's
  `cpp_extension` requires to build any CUDA extension. That installer needs
  admin, and there is no supported way around it. `nvcc` alone can now come from
  pip (`nvidia-cuda-nvcc-cu12`, 12.9.86), but a compiler cannot.
* Several LHM dependencies are awkward-to-hostile on Windows: `chumpy`,
  `decord`, `pyrender`, `open3d`, `basicsr`.
* **WSL** needs one elevated command and a reboot, after which `gcc`, the CUDA
  toolkit and everything else install inside the distro as root.

## What LHM actually needs to compile

Not `diff-gaussian-rasterization`, as assumed earlier - LHM pins **`gsplat==1.4.0`**,
which JIT-compiles its CUDA kernels on first import. So a compiler and `nvcc`
are required either way. No prebuilt gsplat wheel for cu128/sm_120 was found.

## Compatibility work already anticipated

LHM's pins target an older stack and will not run as written on Blackwell:

| Pin | Problem | Plan |
|---|---|---|
| `torch==2.3.0` | no sm_120 kernels; would fail at first launch | override to 2.11.x+cu128, per the project's existing rule about not following old CUDA pins |
| `numpy==1.23.0` | too old for that torch | raise |
| `chumpy` | unmaintained, breaks on modern numpy/python | patch or vendor |
| `gsplat==1.4.0` | must build for sm_120 | `TORCH_CUDA_ARCH_LIST=12.0`, compile from source |

Every change gets recorded here as it is made.

## Second thing that will need a human

LHM is SMPL-X based. The body model files come from the MPI site behind a
registration and licence acceptance, which is a decision for the project owner
rather than something to click through on someone's behalf. Acceptable under the
current personal/non-commercial policy - but it is a manual step, and it is
worth doing at the same time as the reboot rather than discovering it later.
