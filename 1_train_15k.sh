#!/bin/bash
# 15k pretrain:  cd /path/to/VLSplat && bash 1_train_15k.sh
# Dataset, training mode and run names come from config/<DATASET_CONFIG>.py (default vlsplat_render_scannetpp);
# SCENE, OUTPUT_DIR, PORT, ... override it.
dataset_config="${DATASET_CONFIG:-vlsplat_render_scannetpp}"
eval "$(python config/load_config.py "${dataset_config}" --shell)"

python train_vlsplat.py \
    --iterations "${cfg_base_iters}" \
    -s "${DATASET_ROOT_DIR:-${cfg_dataset_root_dir}}/${SCENE:-${cfg_default_scene}}" \
    -m "${OUTPUT_DIR:-${cfg_output_dir}}/${SCENE:-${cfg_default_scene}}/${RUN_NAME:-${cfg_base_run}}" \
    --images "${IMAGES_DIR:-${cfg_images_dir}}" \
    --object_path "${OBJECT_MASK_DIR:-${cfg_object_mask_dir}}" \
    --weight_loss 1e-0 \
    --seed "${SEED:-${cfg_seed}}" \
    --resolution 2 \
    --port "${PORT:-6009}" \
    --dataset-config "${dataset_config}" \
    ${cfg_train_args}
