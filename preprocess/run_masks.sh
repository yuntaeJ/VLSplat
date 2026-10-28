#!/bin/bash
set -euo pipefail
# Stage 0 - build the hybrid SAM3 + SAM2 supervision masks for one scene.
# Run from the VLSplat root, in the sam3 env (see preprocess/requirements.txt):
#   DATASET=scannetpp SCENE=8b5caf3398  bash preprocess/run_masks.sh
#   DATASET=scannetv2 SCENE=scene0590_00 bash preprocess/run_masks.sh
#
# Writes into the scene dir:  object_mask/  object_mask_json/  object_mask_vis/

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "${here}")"

dataset="${DATASET:-scannetpp}"
scene="${SCENE:-8b5caf3398}"
data_root="${DATA_ROOT:-${repo_root}/data}"
img_dir="${IMAGE_DIR:-${data_root}/${dataset}/${scene}/images}"

sam3_ckpt="${SAM3_CKPT:-${here}/sam3/weights/sam3.pt}"
sam2_ckpt="${SAM2_CKPT:-${here}/sam2/weights/sam2.1_hiera_large.pt}"

[[ -d "${img_dir}" ]] || { echo "[preprocess] images not found: ${img_dir}" >&2; exit 1; }
for checkpoint in "${sam3_ckpt}" "${sam2_ckpt}"; do
    [[ -f "${checkpoint}" ]] || { echo "[preprocess] checkpoint not found: ${checkpoint}" >&2; exit 1; }
done

# SAM3 text prompts (same list reused for every image; encoded once then cached).
TEXT_PROMPTS=(
  Wall floor cabinet ceiling bed chair sofa table door window monitor picture desk curtain
)

echo "[preprocess] ${dataset}/${scene}  ${img_dir}"

python "${here}/sam3/generate_object_masks_sam3.py" \
  --image-dir "${img_dir}" \
  --sam3-ckpt "${sam3_ckpt}" \
  --text-prompts "${TEXT_PROMPTS[@]}"

python "${here}/sam2/generate_object_masks_sam2.py" \
  --image-dir "${img_dir}" \
  --sam2-ckpt "${sam2_ckpt}" \
  --skip-existing

python "${here}/merge_sam2_sam3.py" \
  --image-dir "${img_dir}"
