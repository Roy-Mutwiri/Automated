#!/usr/bin/env bash
# Build the LHM reconstruction environment on a fresh Linux GPU machine.
#
# Designed to be run once on a disposable box. It does NOT assume the repo's
# published pins are correct for the GPU present - see "the torch decision"
# below, which is the one genuinely non-obvious part of this script.
#
#   bash setup.sh              # full setup
#   bash setup.sh --skip-apt   # if you cannot apt-get (no sudo)
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SKIP_APT=0
[[ "${1:-}" == "--skip-apt" ]] && SKIP_APT=1

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
log "1/8  inspect the machine"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
           --format=csv,noheader || { echo "No NVIDIA GPU visible. Stop."; exit 1; }

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
CAP_NUM=${CAP/./}
echo "compute capability: $CAP  (sm_${CAP_NUM})"

# ---------------------------------------------------------------------------
# The torch decision.
#
# LHM pins torch==2.3.0+cu121. That is correct for Ampere and Ada, and it is
# WRONG for Blackwell: those wheels contain no sm_120 kernels, so the install
# succeeds, imports succeed, and the first kernel launch dies with
# "no kernel image is available for execution on the device".
#
# So the pin is honoured where it works and overridden where it cannot.
# Do not "simplify" this into a single pin.
# ---------------------------------------------------------------------------
if (( CAP_NUM >= 120 )); then
  TORCH_SPEC="torch torchvision torchaudio"
  TORCH_INDEX="https://download.pytorch.org/whl/cu128"
  XFORMERS_SPEC="xformers"
  echo ">>> Blackwell or newer: using current torch on cu128, NOT the 2.3.0 pin."
else
  TORCH_SPEC="torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0"
  TORCH_INDEX="https://download.pytorch.org/whl/cu121"
  XFORMERS_SPEC="xformers==0.0.26.post1"
  echo ">>> Ampere/Ada: honouring LHM's published pins (torch 2.3.0 + cu121)."
fi
export TORCH_CUDA_ARCH_LIST="${CAP}"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# ---------------------------------------------------------------------------
if (( SKIP_APT == 0 )); then
  log "2/8  system build tools"
  sudo apt-get update -qq
  sudo apt-get install -y -qq git wget curl build-essential cmake ninja-build \
      python3-dev python3-venv libgl1 libglib2.0-0 libegl1 libxrender1
else
  log "2/8  system build tools  [skipped]"
fi

command -v nvcc >/dev/null || {
  echo "!! nvcc not on PATH. Install the CUDA toolkit, or add it:"
  echo "   export PATH=/usr/local/cuda/bin:\$PATH"
  echo "   (most GPU cloud images already ship it)"
  exit 1
}
nvcc --version | grep release

# ---------------------------------------------------------------------------
log "3/8  python environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip setuptools wheel

log "4/8  torch"
pip install -q $TORCH_SPEC --index-url "$TORCH_INDEX"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch_list", torch.cuda.get_arch_list())
assert torch.cuda.is_available(), "CUDA not available to torch"
cap = torch.cuda.get_device_capability(0)
assert f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list(), \
    "This torch build has no kernels for this GPU. Stop and fix the index-url."
print("kernel check:", float((torch.randn(64,64,device='cuda')@torch.randn(64,64,device='cuda')).sum()))
PY

# ---------------------------------------------------------------------------
log "5/8  clone LHM and apply patches"
[[ -d LHM ]] || git clone --depth 1 https://github.com/aigc3d/LHM.git
python patches/apply_patches.py --repo LHM

log "6/8  python dependencies"
pip install -q -r LHM/requirements.txt || {
  echo "!! bulk install failed - retrying without the torch pins that patch 01 relaxed"
  pip install -q -r requirements/fallback.txt
}
pip install -q -U $XFORMERS_SPEC --index-url "$TORCH_INDEX" || \
  echo "!! xformers unavailable for this torch; LHM can run without it (slower attention)"

# basicsr from source: the released wheel imports a torchvision path that moved.
pip uninstall -y -q basicsr || true
pip install -q "git+https://github.com/XPixelGroup/BasicSR"

log "7/8  compile the CUDA extensions  (this is the step that fails on a bad env)"
pip install -q "git+https://github.com/hitsz-zuoqi/sam2/"
pip install -q "git+https://github.com/ashawkey/diff-gaussian-rasterization/"
pip install -q "git+https://github.com/camenduru/simple-knn/"
pip install -q "git+https://github.com/facebookresearch/pytorch3d.git"

log "8/8  verify"
python verify_environment.py --build

cat <<'EOF'

Environment built. Nothing has been downloaded except code and wheels.

Next:
  1. Put the SMPL-X body model where the README says (section: "What you must
     download yourself"). setup.sh deliberately does not fetch it.
  2. bash run_reconstruction.sh
EOF
