#!/bin/bash
# Unified-Lift baseline 2D eval (SAM3-only graph, no VLM / intra/inter-refine).
#   cd /path/to/VLSplat && bash test_unified-lift_2d_render.sh

scene="${SCENE:-8b5caf3398}"
exp_dir="${EXP_DIR:-./output/scannetpp/${scene}/unified_lift_30k/point_cloud/iteration_30000}"
scene_root="${SCENE_ROOT:-./data/scannetpp/${scene}}"
gt_dir="${GT_DIR:-${scene_root}/gt/render}"
scene_graph_json="${SCENE_GRAPH_JSON:-${exp_dir}/scene_graph_eval.json}"
object_path="${OBJECT_PATH:-${scene_root}/object_mask}"
label_json_root="${LABEL_JSON_ROOT:-${scene_root}/object_mask_json}"
images="${IMAGES:-images}"
sam3_resolution="${SAM3_RESOLUTION:-4}"
target_classes="${TARGET_CLASSES:-cabinet bed chair desk sofa monitor}"
split="${SPLIT:-test}"
out_dir="${OUT_DIR:-${exp_dir}/semantic_eval_${split}}"
tile_width="${TILE_WIDTH:-360}"
data_device="${DATA_DEVICE:-cuda}"

echo "[eval] ${scene}  ${split}  ${exp_dir}"

python script/prepare_scene_graph_for_unified_lift.py \
    --exp-dir "${exp_dir}" \
    --source-path "${scene_root}" \
    --object-path "${object_path}" \
    --label-json-root "${label_json_root}" \
    --images "${images}" \
    --resolution "${sam3_resolution}" \
    --target-classes ${target_classes} \
    --out-json "${scene_graph_json}"

python script/eval_ckpt_semantic_vs_gt.py \
    --exp-dir "${exp_dir}" \
    --scene-root "${scene_root}" \
    --gt-dir "${gt_dir}" \
    --scene-graph-json "${scene_graph_json}" \
    --split "${split}" \
    --out-dir "${out_dir}" \
    --tile-width "${tile_width}" \
    --data-device "${data_device}"
