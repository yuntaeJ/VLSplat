#!/bin/bash
# Resume 15k -> 30k with intra-/inter-node refinement:  cd /path/to/VLSplat && bash 3_train_15k_30k.sh
# Dataset, training mode (the unified_lift+ours_scannetpp config trains the Unified-Lift baseline) and refinement
# settings come from config/<DATASET_CONFIG>.py; override one setting with a flag, e.g.  bash 3_train_15k_30k.sh --sg-gate-thr 0.6
# PIPELINE_VARIANT=sam3_vlm (default)|sam3 picks scene_graph_<variant>.json from step 2 (use the same value as in step 2).
dataset_config="${DATASET_CONFIG:-vlsplat_render_scannetpp}"
eval "$(python config/load_config.py "${dataset_config}" --shell)"

scene="${SCENE:-${cfg_default_scene}}"
variant="${PIPELINE_VARIANT:-sam3_vlm}"
base_dir="${OUTPUT_DIR:-${cfg_output_dir}}/${scene}/${cfg_base_run}"
start_iter="${START_ITER:-${cfg_base_iters}}"

python train_vlsplat.py \
    --iterations "${END_ITER:-${cfg_final_iters}}" \
    --save_iterations "${END_ITER:-${cfg_final_iters}}" \
    --checkpoint_iterations "${END_ITER:-${cfg_final_iters}}" \
    -s "${DATASET_ROOT_DIR:-${cfg_dataset_root_dir}}/${scene}" \
    -m "${OUTPUT_DIR:-${cfg_output_dir}}/${scene}/${REFINE_RUN:-${cfg_refine_run}}" \
    --images "${IMAGES_DIR:-${cfg_images_dir}}" \
    --object_path "${OBJECT_MASK_DIR:-${cfg_object_mask_dir}}" \
    --weight_loss 1e-0 \
    --seed "${SEED:-${cfg_seed}}" \
    --resolution 2 \
    --port "${PORT:-6009}" \
    --start_checkpoint "${START_CHECKPOINT:-${base_dir}/chkpnt${start_iter}.pth}" \
    --dataset-config "${dataset_config}" \
    --scene-graph "${SCENE_GRAPH_PATH:-${base_dir}/point_cloud/iteration_${start_iter}/scene_graph_${variant}.json}" \
    --sg-refine-start "${start_iter}" \
    ${cfg_train_args} ${cfg_step3_args} \
    "$@"
