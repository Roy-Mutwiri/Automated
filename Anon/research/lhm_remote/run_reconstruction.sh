#!/usr/bin/env bash
# ONE reconstruction of ONE avatar. No seed lottery, no parameter sweep.
#
#   bash run_reconstruction.sh                    # default: LHM-1B-HF
#   bash run_reconstruction.sh LHM-500M-HF
#   bash run_reconstruction.sh LHM-1B-HF inputs/avatar_rgba.png
#
# Model choice matters and is not free:
#
#   Our input is a HALF-BODY image - head, shoulders, upper torso, one arm.
#   LHM's model table splits on exactly this. LHM-500M and LHM-1B are marked
#   "full body" only. The **-HF** variants and MINI accept "half & full body".
#   So a non-HF model is the wrong tool for our plate regardless of its size.
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODEL_NAME="${1:-LHM-1B-HF}"
IMAGE="${2:-inputs/avatar_rgba.png}"

case "$MODEL_NAME" in
  *-HF|LHM-MINI) ;;
  *) echo "!! $MODEL_NAME expects a FULL-BODY image; ours is half body."
     echo "   Use LHM-1B-HF, LHM-500M-HF or LHM-MINI."; exit 1 ;;
esac

[[ -f "$HERE/.venv/bin/activate" ]] || { echo "Run setup.sh first."; exit 1; }
# shellcheck disable=SC1091
source "$HERE/.venv/bin/activate"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

log "0/4  re-verify before spending money on downloads"
python verify_environment.py || {
  echo "Environment is not ready. Fix that before downloading weights."; exit 1; }

cd LHM

log "1/4  prior model weights (SMPL-X assets, trackers, segmentation)"
if [[ ! -d pretrained_models/human_model_files ]]; then
  wget -q --show-progress -c \
    https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar
  tar -xf LHM_prior_model.tar
  rm -f LHM_prior_model.tar
else
  echo "prior models already present"
fi

# If the project owner supplied SMPL-X officially, prefer it over anything the
# prior tar happens to bundle. Provenance matters more than convenience.
if compgen -G "../private_models/*" > /dev/null; then
  log "1b/4  installing owner-supplied SMPL-X over the bundled copy"
  mkdir -p pretrained_models/human_model_files/smplx
  cp -v ../private_models/SMPLX_NEUTRAL.npz \
        pretrained_models/human_model_files/smplx/ 2>/dev/null || true
  cp -rv ../private_models/smplx/. \
        pretrained_models/human_model_files/smplx/ 2>/dev/null || true
fi

log "2/4  model checkpoint: $MODEL_NAME"
python - "$MODEL_NAME" <<'PY'
import sys
from huggingface_hub import snapshot_download
name = sys.argv[1]
p = snapshot_download(repo_id=f"3DAIGC/{name}",
                      cache_dir="./pretrained_models/huggingface")
print("checkpoint at", p)
PY

log "3/4  reconstruct + export mesh  (single run)"
IMG_ABS="$(cd "$HERE" && python -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$IMAGE")"
mkdir -p "$HERE/outputs"
# export_mesh=True is the identity-gate path: geometry we can look at and, if
# it is good, take back to Windows. Animation comes later and only if it passes.
bash ./inference_mesh.sh "$MODEL_NAME" "$IMG_ABS" 2>&1 | tee "$HERE/outputs/reconstruction.log"

log "4/4  turntable + package"
cd "$HERE"
python export_results.py --model-name "$MODEL_NAME" --image "$IMAGE"

cat <<'EOF'

Done. Look at outputs/lhm_identity_turntable.png FIRST.

If the man in it is not recognisably the man in inputs/avatar_identity_camera1.png,
stop here and report the failure - do not download the package or spend time on
Blender integration.
EOF
