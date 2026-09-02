#!/usr/bin/env bash
# ONE reconstruction. Single image -> canonical 3DGS PLY. No pose, no animation.
#
#   bash run_lhmpp.sh
#
# This deliberately answers ONLY stage A+B:
#
#   A. single-image reconstruction
#   B. canonical 3DGS export
#
# Stage C (animation / pose driving) is NOT attempted, and no SMPL-X asset is
# fetched. Omitting --pose_dir makes to_gs_ply.py take the canonical T-pose
# path through `inference_gs` with a synthetic camera. If that completes with no
# `human_model_files` present, we have our answer: the SMPL-X registration can
# wait until animation, and may never be needed for the identity gate.
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODEL_NAME="LHMPP-700M-SMPLX-FREE"
IMAGE="${1:-inputs/avatar_rgba.png}"
OUT="$HERE/outputs/lhmpp_avatar_v01"

[[ -f .venv-pp/bin/activate ]] || { echo "Run setup_lhmpp.sh first."; exit 1; }
# shellcheck disable=SC1091
source .venv-pp/bin/activate
log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

log "0/5  re-verify before downloading anything"
python verify_environment.py --lhmpp || { echo "Environment not ready."; exit 1; }

mkdir -p "$OUT/original" "$OUT/previews"
cd LHM-plusplus

log "1/5  prior models"
# Needed for segmentation / preprocessing regardless of the SMPL-X question.
if [[ ! -d pretrained_models ]] || [[ -z "$(ls -A pretrained_models 2>/dev/null)" ]]; then
  echo "Fetching prior models per upstream instructions..."
  bash -c "$(grep -m1 -o 'wget[^\"]*prior[^\" ]*' README.md || true)" || \
    echo "!! could not auto-fetch prior models; see README.md 'prior models'"
fi

log "2/5  checkpoint: $MODEL_NAME"
python - "$MODEL_NAME" <<'PY'
import sys
name = sys.argv[1]
try:
    from modelscope import snapshot_download
    p = snapshot_download(model_id=f"Damo_XR_Lab/{name}", cache_dir="./pretrained_models")
except Exception as exc:
    print("modelscope failed:", exc, "- trying huggingface")
    from huggingface_hub import snapshot_download as hf
    p = hf(repo_id=f"3DAIGC/{name}", cache_dir="./pretrained_models/huggingface")
print("CHECKPOINT_PATH", p)
PY

CKPT="./pretrained_models/Damo_XR_Lab/${MODEL_NAME}"
[[ -d "$CKPT" ]] || CKPT=$(find ./pretrained_models -maxdepth 4 -type d -name "*${MODEL_NAME}*" | head -1)
echo "using checkpoint: $CKPT"

log "3/5  record what SMPL-X state we are actually in"
if [[ -d pretrained_models/human_model_files ]]; then
  echo "human_model_files IS present (came with the prior models)"
  find pretrained_models/human_model_files -name "*.npz" -o -name "*.pkl" | head -5
else
  echo "human_model_files is ABSENT - if the export still succeeds, the SMPL-X"
  echo "download can be postponed. That is the open question in the manifest."
fi

log "4/5  reconstruct: single image -> canonical T-pose Gaussians -> PLY"
IMG_ABS="$(python -c "import os,sys;print(os.path.abspath(sys.argv[1]))" "$HERE/$IMAGE")"
REFDIR="$HERE/outputs/_refdir"; mkdir -p "$REFDIR"; cp -f "$IMG_ABS" "$REFDIR/"

START=$(date +%s)
python scripts/inference/to_gs_ply.py \
  --model_path "$CKPT" \
  --model_name "$MODEL_NAME" \
  --ref_images "$REFDIR" \
  2>&1 | tee "$OUT/reconstruction.log"
END=$(date +%s)
echo "inference wall time: $((END-START))s" | tee -a "$OUT/reconstruction.log"

log "5/5  collect, turntable, and compare renderers"
PLY=$(find . -name "*.ply" -newermt "-1 hour" | head -1)
if [[ -n "$PLY" ]]; then
  cp -v "$PLY" "$OUT/original/canonical_gs.ply"
  echo "gaussians: $(python "$HERE/inspect_ply.py" "$OUT/original/canonical_gs.ply")"
else
  echo "!! no PLY produced - read $OUT/reconstruction.log"
fi

cd "$HERE"
python export_results.py --model-name "$MODEL_NAME" --image "$IMAGE" \
                         --out outputs/lhmpp_avatar_v01 || true

cat <<'EOF'

Look at outputs/lhmpp_identity_turntable.png FIRST.

The question is not "is this a good 3D human". It is:
  is this recognisably the same fictional man as inputs/avatar_identity_camera1.png,
  FROM THE SIDE as well as the front?

MPFB scored ~2/10. If this is not dramatically better, stop and report - do not
proceed to Blender integration, and do not start adding hair or beard.
EOF
