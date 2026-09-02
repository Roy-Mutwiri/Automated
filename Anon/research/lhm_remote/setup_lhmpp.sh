#!/usr/bin/env bash
# Build the LHM++ environment on a disposable Linux GPU box.
#
# Policy change from setup.sh (which targeted the older LHM): this script runs
# the repository's DOCUMENTED stack - python 3.10, CUDA 12.1, torch 2.3.0 - and
# does not modernise anything unless it actually breaks. The whole point of a
# disposable box is that the old research stack gets to run in its native
# environment instead of being fought.
#
#   bash setup_lhmpp.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

COMMIT=$(python3 -c "import json;print(json.load(open('environment_manifest.json'))['target']['commit'])")
log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
log "1/8  check the GPU is the right kind"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader \
  || { echo "No NVIDIA GPU visible. Stop."; exit 1; }

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
CAP_NUM=${CAP/./}
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

if (( CAP_NUM >= 120 )); then
  cat <<'EOF'

!! This is a Blackwell-class GPU (sm_120).

   torch 2.3.0+cu121 has no sm_120 kernels. It will install, it will import,
   and it will die at the first kernel launch with
   "no kernel image is available for execution on the device".

   Choosing this box recreates exactly the compatibility problem the remote
   plan exists to escape. Pick an A10 / A5000 / A6000 / 3090 / 4090 / L40S.

   If you genuinely have no alternative, re-run with FORCE_MODERN=1 and expect
   to debug the research stack rather than the reconstruction.
EOF
  [[ "${FORCE_MODERN:-0}" == "1" ]] || exit 1
  TORCH_SPEC="torch torchvision torchaudio"; TORCH_INDEX="https://download.pytorch.org/whl/cu128"
  XFORMERS_SPEC="xformers"
else
  echo ">>> sm_${CAP_NUM}: running the documented stack unmodified."
  TORCH_SPEC="torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0"
  TORCH_INDEX="https://download.pytorch.org/whl/cu121"
  XFORMERS_SPEC="xformers==0.0.26.post1"
fi
(( VRAM >= 15000 )) || echo "!! only ${VRAM} MiB VRAM; 16 GB is the practical floor"
export TORCH_CUDA_ARCH_LIST="${CAP}"

# ---------------------------------------------------------------------------
log "2/8  system build tools"
sudo apt-get update -qq
sudo apt-get install -y -qq git wget curl build-essential cmake ninja-build \
    python3.10 python3.10-dev python3.10-venv libgl1 libglib2.0-0 libegl1 libxrender1 \
  || sudo apt-get install -y -qq git wget curl build-essential cmake ninja-build \
       python3-dev python3-venv libgl1 libglib2.0-0 libegl1 libxrender1

command -v nvcc >/dev/null || { echo "!! nvcc missing. export PATH=/usr/local/cuda/bin:\$PATH"; exit 1; }
nvcc --version | grep release

# ---------------------------------------------------------------------------
log "3/8  clone LHM++ at the pinned commit"
if [[ ! -d LHM-plusplus ]]; then
  git clone https://github.com/aigc3d/LHM-plusplus.git
fi
cd LHM-plusplus
git fetch --depth 50 origin || true
git checkout -q "$COMMIT" || { echo "!! commit $COMMIT unavailable; check the manifest"; exit 1; }
echo "at $(git rev-parse --short HEAD)"
cd "$HERE"

# ---------------------------------------------------------------------------
log "4/8  python 3.10 environment"
PY=$(command -v python3.10 || command -v python3)
"$PY" -m venv .venv-pp
# shellcheck disable=SC1091
source .venv-pp/bin/activate
python -V
python -m pip install -q --upgrade pip setuptools wheel

log "5/8  torch (documented version unless overridden above)"
pip install -q $TORCH_SPEC --index-url "$TORCH_INDEX"
pip install -q -U $XFORMERS_SPEC --index-url "$TORCH_INDEX" || \
  echo "!! xformers unavailable; continuing without it"
python - <<'PY'
import torch
cap = torch.cuda.get_device_capability(0)
print("torch", torch.__version__, "cuda", torch.version.cuda, "arch", torch.cuda.get_arch_list())
assert torch.cuda.is_available()
assert f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list(), \
    "torch has no kernels for this GPU - wrong box or wrong index-url"
print("kernel ok:", float((torch.randn(64,64,device='cuda')@torch.randn(64,64,device='cuda')).sum()))
PY

log "6/8  python dependencies"
pip install -q -r LHM-plusplus/requirements.txt || {
  echo "!! bulk install failed; retrying with relaxed pins"
  python patches/apply_patches.py --repo LHM-plusplus
  pip install -q -r LHM-plusplus/requirements.txt
}
pip install -q "rembg[cpu]"

log "7/8  compiled extensions  (the step that fails on a wrong box)"
# LHM++ needs more than LHM did: pointops is built from source in-tree, and
# spconv / torch_scatter must match the CUDA and torch versions exactly.
( cd LHM-plusplus/lib/pointops && python setup.py install )
pip install -q spconv-cu121 || echo "!! spconv-cu121 failed - check CUDA version"
pip install -q torch_scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html \
  || echo "!! torch_scatter wheel not found for this torch/CUDA pair"
pip install -q --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html \
  || pip install -q "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -q "git+https://github.com/ashawkey/diff-gaussian-rasterization/"
pip install -q "git+https://github.com/camenduru/simple-knn/"

log "8/8  verify"
python verify_environment.py --lhmpp --build

cat <<'EOF'

Environment built. No model weights and no SMPL-X downloaded yet - that is
deliberate.

Next:
  bash run_lhmpp.sh
EOF
