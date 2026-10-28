#!/bin/bash
# VLSplat 2D eval: rendered semantic maps vs GT on the test split, using the scene_graph.json written by step 3.
#   cd /path/to/VLSplat && bash test_vlsplat_2d_render.sh

scene="${SCENE:-8b5caf3398}"
exp_dir="${EXP_DIR:-./output/scannetpp/${scene}/vlsplat_30k/point_cloud/iteration_30000}"
scene_root="${SCENE_ROOT:-./data/scannetpp/${scene}}"
gt_dir="${GT_DIR:-${scene_root}/gt/render}"
scene_graph_json="${SCENE_GRAPH_JSON:-${exp_dir}/scene_graph.json}"
split="${SPLIT:-test}"
out_dir="${OUT_DIR:-${exp_dir}/semantic_eval_${split}}"
tile_width="${TILE_WIDTH:-360}"
data_device="${DATA_DEVICE:-cuda}"

echo "[eval] ${scene}  ${split}  ${exp_dir}"

python script/eval_ckpt_semantic_vs_gt.py \
    --exp-dir "${exp_dir}" \
    --scene-root "${scene_root}" \
    --gt-dir "${gt_dir}" \
    --scene-graph-json "${scene_graph_json}" \
    --split "${split}" \
    --out-dir "${out_dir}" \
    --tile-width "${tile_width}" \
    --data-device "${data_device}"
