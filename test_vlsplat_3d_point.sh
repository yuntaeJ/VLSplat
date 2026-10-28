#!/bin/bash
# 3D point-level semantic eval (mIoU / mAcc) vs ScanNetV2 GT vertices.
# Prints VLSplat and VLSplat+ROI-thing (stuff groups; thing ∩ bbox).
#   cd /path/to/VLSplat && bash test_vlsplat_3d_point.sh

scene="${SCENE:-scene0590_00}"
gt_dir="${GT_DIR:-./data/scannetv2/${scene}/gt/points}"
pred_dir="${PRED_DIR:-./output/scannetv2/${scene}/vlsplat_30k_fixed}"

python script/eval_3d_point_semantic.py \
    --pred-dir "${pred_dir}" \
    --gt-dir "${gt_dir}"
