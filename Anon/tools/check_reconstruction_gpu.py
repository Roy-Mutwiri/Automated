"""Blackwell/sm_120 compatibility probe. Run BEFORE installing a candidate.

Most single-image human reconstruction repositories were written against CUDA
11.8 or 12.1 and PyTorch 2.0-2.3. On this machine that combination installs
cleanly, imports cleanly, and then dies at the first kernel launch with

    no kernel image is available for execution on the device

because the wheels contain no sm_120 kernels. `README.md` already documents that
trap for the LivePortrait stack; this script is that lesson made runnable, so
the answer costs seconds instead of an afternoon of dependency installation.

## Four different things get called "CUDA" and they are not the same

The specification is right to insist on separating them, because a repo can be
broken by any one of the four while the other three look fine:

1. **Driver** - what `nvidia-smi` reports. Backwards compatible; a new driver
   runs old CUDA.
2. **CUDA toolkit** (`nvcc`) - what *compiles* custom extensions. This is what
   decides whether `diff-gaussian-rasterization` can be built for sm_120 at all.
3. **PyTorch CUDA runtime** - the version torch was built against, and more
   importantly `torch.cuda.get_arch_list()`, which is the actual list of
   architectures its kernels were compiled for. **A new driver does not put
   sm_120 kernels into an old wheel.**
4. **Custom extension architectures** - each compiled `.pyd`/`.so` has its own
   arch list, independent of torch's.

Only 3 and 4 can produce the "no kernel image" failure, and only 2 can fix it.

Usage
-----
    python tools/check_reconstruction_gpu.py
    python tools/check_reconstruction_gpu.py --build-test   # compile a kernel
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import textwrap

SM = "sm_120"


def line(k, v):
    print(f"  {k:<34} {v}")


def run(cmd) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<{type(exc).__name__}: {exc}>"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-test", action="store_true",
                    help="compile a trivial CUDA extension for this "
                         "architecture; this is the check that actually "
                         "predicts whether a Gaussian rasterizer will build")
    args = ap.parse_args()

    verdict = {}
    print("\n=== 1. host ===")
    line("python", platform.python_version())
    line("platform", f"{platform.system()} {platform.release()}")
    line("executable", sys.executable)

    print("\n=== 2. driver ===")
    smi = run(["nvidia-smi",
               "--query-gpu=name,compute_cap,memory.total,memory.free,driver_version",
               "--format=csv,noheader"])
    line("nvidia-smi", smi)
    cap = smi.split(",")[1].strip() if "," in smi else "?"
    verdict["driver"] = smi and "<" not in smi

    print("\n=== 3. cuda toolkit (compiles extensions) ===")
    nvcc = shutil.which("nvcc")
    if nvcc:
        ver = [l for l in run([nvcc, "--version"]).splitlines() if "release" in l]
        line("nvcc", nvcc)
        line("version", ver[0].strip() if ver else "?")
        verdict["nvcc"] = True
    else:
        line("nvcc", "NOT FOUND on PATH")
        line("consequence", "custom CUDA extensions cannot be compiled from "
                            "source; only prebuilt wheels will work")
        verdict["nvcc"] = False
    line("CUDA_HOME", os.environ.get("CUDA_HOME", "<unset>"))

    print("\n=== 4. pytorch runtime ===")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        line("torch", f"IMPORT FAILED: {exc}")
        return 2
    line("torch", torch.__version__)
    line("built against CUDA", torch.version.cuda)
    line("cuda available", torch.cuda.is_available())
    arch = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    line("arch_list", ", ".join(arch) or "<none>")
    has_sm120 = SM in arch
    line(f"{SM} present", "YES" if has_sm120 else "NO  <-- THE decisive line")
    verdict["torch_sm120"] = has_sm120

    if torch.cuda.is_available():
        line("device", torch.cuda.get_device_name(0))
        major, minor = torch.cuda.get_device_capability(0)
        line("compute capability", f"{major}.{minor}")
        free, total = torch.cuda.mem_get_info()
        line("VRAM free / total", f"{free/1e9:.1f} / {total/1e9:.1f} GB")

    print("\n=== 5. does a kernel actually launch? ===")
    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        b = (a @ a).float().sum().item()
        torch.cuda.synchronize()
        line("fp16 matmul", f"OK ({b:.1f})")
        verdict["kernel"] = True
    except Exception as exc:  # noqa: BLE001
        line("fp16 matmul", f"FAILED: {type(exc).__name__}: {exc}")
        verdict["kernel"] = False

    print("\n=== 6. optional deps research repos assume ===")
    for name in ("xformers", "pytorch3d", "diff_gaussian_rasterization",
                 "simple_knn", "nvdiffrast", "smplx", "trimesh", "open3d"):
        try:
            mod = __import__(name)
            line(name, getattr(mod, "__version__", "installed"))
        except Exception:  # noqa: BLE001
            line(name, "-")

    if args.build_test:
        print("\n=== 7. custom CUDA extension build ===")
        if not verdict.get("nvcc"):
            line("result", "SKIPPED - no nvcc")
        else:
            import tempfile
            from pathlib import Path
            src = textwrap.dedent("""
                #include <torch/extension.h>
                __global__ void addk(float* o, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]+=1.0f;}
                void addone(torch::Tensor t){int n=t.numel();
                    addk<<<(n+255)/256,256>>>(t.data_ptr<float>(), n);}
                PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){m.def("addone",&addone);}
            """)
            tmp = Path(tempfile.mkdtemp(prefix="smprobe_"))
            (tmp / "probe.cu").write_text(src, encoding="utf-8")
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
            try:
                from torch.utils.cpp_extension import load
                ext = load(name="smprobe", sources=[str(tmp / "probe.cu")],
                           verbose=False, build_directory=str(tmp))
                t = torch.zeros(16, device="cuda")
                ext.addone(t)
                torch.cuda.synchronize()
                ok = bool((t == 1).all().item())
                line("compile + run", "OK" if ok else "compiled but wrong result")
                verdict["build"] = ok
            except Exception as exc:  # noqa: BLE001
                line("compile + run", f"FAILED: {type(exc).__name__}")
                print(textwrap.indent(str(exc)[:900], "      "))
                verdict["build"] = False

    print("\n=== verdict ===")
    if not verdict.get("torch_sm120"):
        print("  This environment CANNOT run sm_120 kernels. Any candidate "
              "installed here will fail at first launch.")
    elif not verdict.get("kernel"):
        print("  arch_list claims sm_120 but a kernel launch failed. "
              "Investigate before installing anything.")
    else:
        print(f"  Torch is Blackwell-capable ({SM} present, kernel launches).")
        if not verdict.get("nvcc"):
            print("  BUT nvcc is absent: any candidate needing a custom CUDA "
                  "extension (Gaussian rasterizers, simple-knn, PyTorch3D from "
                  "source) cannot be built here without installing a toolkit.")
        if args.build_test and verdict.get("build"):
            print("  Custom CUDA extensions compile and run for this arch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
