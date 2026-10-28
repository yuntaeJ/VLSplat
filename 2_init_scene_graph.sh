#!/bin/bash
# Scene graph (structure -> VLM class review and text_prompt -> merge):  cd /path/to/VLSplat && bash 2_init_scene_graph.sh
# Needs (PIPELINE_VARIANT=sam3_vlm, default): export OPENROUTER_API_KEY='sk-or-...'
# Settings: GRAPH section of config/<DATASET_CONFIG>.py; override one with a flag, e.g.  bash 2_init_scene_graph.sh --roi-k 8
dataset_config="${DATASET_CONFIG:-vlsplat_render_scannetpp}"
eval "$(python config/load_config.py "${dataset_config}" --shell)"

scene="${SCENE:-${cfg_default_scene}}"
scene_dir="${DATASET_ROOT_DIR:-${cfg_dataset_root_dir}}/${scene}"
variant="${PIPELINE_VARIANT:-sam3_vlm}"   # sam3 = SAM3 label votes only (no VLM); sam3_vlm = VLM assigns every node's class

python scene_graph_builder/init_scene_graph.py \
    "${ITER_DIR:-${OUTPUT_DIR:-${cfg_output_dir}}/${scene}/${RUN_NAME:-${cfg_base_run}}/point_cloud/iteration_${ITER:-${cfg_base_iters}}}" \
    --dataset-config "${dataset_config}" \
    --source-path "${scene_dir}" \
    --object-path "${OBJECT_PATH:-${scene_dir}/${cfg_object_mask_dir}}" \
    --label-json-root "${LABEL_JSON_ROOT:-${scene_dir}/${cfg_object_mask_dir}_json}" \
    --images "${IMAGES:-${cfg_images_dir}}" \
    --resolution "${RESOLUTION:-${cfg_graph_resolution}}" \
    --variant "${variant}" \
    --out-name "${OUT_NAME:-scene_graph_${variant}.json}" \
    "$@"
