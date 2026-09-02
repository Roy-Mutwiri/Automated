#!/usr/bin/env python3
"""Prove the GPU environment works BEFORE downloading any model weights.

LHM needs three CUDA extensions compiled from source
(`diff-gaussian-rasterization`, `simple-knn`, `pytorch3d`) plus a working
`gsplat`. If any of them cannot build for the GPU actually present, nothing
later will work - and finding that out after pulling several gigabytes of
checkpoints wastes both time and money on a rented machine.

So this runs first, and it is deliberately noisy about *which* thing failed.

    python verify_environment.py            # checks only
    python verify_environment.py --build    # also compiles a test CUDA kernel

Exit code 0 means the environment is ready for `run_reconstruction.sh`.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys

CHECKS: list[tuple[str, bool, str]] = []


def record(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {name:<38} {detail}")
    return ok


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return (p.stdout or p.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<{type(exc).__name__}>"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true",
                    help="compile a trivial CUDA extension for this GPU; this "
                         "is the check that actually predicts whether the "
                         "Gaussian rasterizers will build")
    args = ap.parse_args()

    print("\n=== host ===")
    record("python", sys.version_info[:2] >= (3, 9),
           f"{platform.python_version()} ({sys.executable})")
    record("linux", platform.system() == "Linux", platform.platform())

    print("\n=== toolchain ===")
    nvcc = shutil.which("nvcc")
    record("nvcc", bool(nvcc),
           next((l.strip() for l in run([nvcc, "--version"]).splitlines()
                 if "release" in l), "") if nvcc else "NOT FOUND")
    for tool in ("gcc", "g++", "cmake", "ninja"):
        path = shutil.which(tool)
        record(tool, bool(path), run([tool, "--version"]).splitlines()[0]
               if path else "NOT FOUND")

    print("\n=== gpu ===")
    smi = run(["nvidia-smi",
               "--query-gpu=name,compute_cap,memory.total,driver_version",
               "--format=csv,noheader"])
    record("nvidia-smi", "," in smi, smi)

    print("\n=== torch ===")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        record("torch import", False, str(exc)[:120])
        return summary()

    record("torch", True, f"{torch.__version__} (built for CUDA "
                          f"{torch.version.cuda})")
    cuda_ok = record("cuda available", torch.cuda.is_available())
    if not cuda_ok:
        return summary()

    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    arch_list = torch.cuda.get_arch_list()
    record("device", True, torch.cuda.get_device_name(0))
    record(f"{arch} in torch arch_list", arch in arch_list,
           ", ".join(arch_list))
    free, total = torch.cuda.mem_get_info()
    record("VRAM", total / 1e9 >= 15,
           f"{free/1e9:.1f} free / {total/1e9:.1f} GB "
           f"(LHM-1B-HF wants ~24 GB, MINI ~16 GB)")

    try:
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        (a @ a).sum().item()
        torch.cuda.synchronize()
        record("cuda kernel launch", True, "fp16 matmul ran")
    except Exception as exc:  # noqa: BLE001
        record("cuda kernel launch", False, f"{type(exc).__name__}: {exc}"[:140])

    print("\n=== LHM's compiled dependencies ===")
    for mod, why in (
        ("diff_gaussian_rasterization", "3DGS rasterizer, compiled"),
        ("simple_knn", "compiled"),
        ("pytorch3d", "compiled"),
        ("gsplat", "JIT-compiles on first use"),
        ("xformers", "attention"),
        ("sam2", "segmentation, modified fork"),
        ("smplx", "body model loader"),
        ("roma", ""), ("kiui", ""), ("rembg", ""),
    ):
        try:
            m = __import__(mod)
            record(mod, True, f"{getattr(m, '__version__', 'installed')}  {why}")
        except Exception as exc:  # noqa: BLE001
            record(mod, False, f"{type(exc).__name__}  {why}")

    print("\n=== SMPL-X body model ===")
    root = os.path.dirname(os.path.abspath(__file__))
    found = []
    for base in (os.path.join(root, "LHM", "pretrained_models", "human_model_files"),
                 os.path.join(root, "private_models")):
        if os.path.isdir(base):
            for dirpath, _, files in os.walk(base):
                for f in files:
                    if f.lower().startswith(("smplx", "smpl_")) and \
                            f.lower().endswith((".npz", ".pkl")):
                        found.append(os.path.join(dirpath, f))
    record("SMPL-X model files present", bool(found),
           "; ".join(sorted(found)[:3]) if found
           else "none found - see README section 'What you must download yourself'")

    if args.build:
        print("\n=== test CUDA extension build ===")
        build_test(arch)

    return summary()


def build_test(arch):
    import tempfile
    import textwrap

    import torch
    from torch.utils.cpp_extension import load

    src = textwrap.dedent("""
        #include <torch/extension.h>
        __global__ void addk(float* o, int n){
            int i = blockIdx.x*blockDim.x + threadIdx.x; if(i<n) o[i]+=1.0f; }
        void addone(torch::Tensor t){ int n=t.numel();
            addk<<<(n+255)/256,256>>>(t.data_ptr<float>(), n); }
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){ m.def("addone", &addone); }
    """)
    tmp = tempfile.mkdtemp(prefix="lhmprobe_")
    with open(os.path.join(tmp, "probe.cu"), "w") as fh:
        fh.write(src)
    major, minor = torch.cuda.get_device_capability(0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    try:
        ext = load(name="lhmprobe", sources=[os.path.join(tmp, "probe.cu")],
                   verbose=False, build_directory=tmp)
        t = torch.zeros(8, device="cuda")
        ext.addone(t)
        torch.cuda.synchronize()
        record(f"compile + run for {arch}", bool((t == 1).all().item()),
               f"TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")
    except Exception as exc:  # noqa: BLE001
        record(f"compile + run for {arch}", False, f"{type(exc).__name__}")
        print(f"\n{str(exc)[:1500]}\n")


def summary() -> int:
    failed = [n for n, ok, _ in CHECKS if not ok]
    print("\n" + "=" * 62)
    if failed:
        print(f"NOT READY - {len(failed)} check(s) failed:")
        for n in failed:
            print(f"  - {n}")
        print("\nDo NOT download model weights until these pass.")
        return 1
    print("READY. Environment satisfies LHM's requirements.")
    print("Next: bash run_reconstruction.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
