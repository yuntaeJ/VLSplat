"""
SAM2.1 Hiera-Large architecture, inlined from
`sam2/configs/sam2.1/sam2.1_hiera_l.yaml` so callers only pass a checkpoint path
(same style as SAM3: architecture in code, weights on disk / HF).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Equivalent to configs/sam2.1/sam2.1_hiera_l.yaml (+ image AMG postprocess defaults).
SAM2_1_HIERA_L_CFG = {
    "model": {
        "_target_": "sam2.modeling.sam2_base.SAM2Base",
        "image_encoder": {
            "_target_": "sam2.modeling.backbones.image_encoder.ImageEncoder",
            "scalp": 1,
            "trunk": {
                "_target_": "sam2.modeling.backbones.hieradet.Hiera",
                "embed_dim": 144,
                "num_heads": 2,
                "stages": [2, 6, 36, 4],
                "global_att_blocks": [23, 33, 43],
                "window_pos_embed_bkg_spatial_size": [7, 7],
                "window_spec": [8, 4, 16, 8],
            },
            "neck": {
                "_target_": "sam2.modeling.backbones.image_encoder.FpnNeck",
                "position_encoding": {
                    "_target_": "sam2.modeling.position_encoding.PositionEmbeddingSine",
                    "num_pos_feats": 256,
                    "normalize": True,
                    "scale": None,
                    "temperature": 10000,
                },
                "d_model": 256,
                "backbone_channel_list": [1152, 576, 288, 144],
                "fpn_top_down_levels": [2, 3],
                "fpn_interp_model": "nearest",
            },
        },
        "memory_attention": {
            "_target_": "sam2.modeling.memory_attention.MemoryAttention",
            "d_model": 256,
            "pos_enc_at_input": True,
            "layer": {
                "_target_": "sam2.modeling.memory_attention.MemoryAttentionLayer",
                "activation": "relu",
                "dim_feedforward": 2048,
                "dropout": 0.1,
                "pos_enc_at_attn": False,
                "self_attention": {
                    "_target_": "sam2.modeling.sam.transformer.RoPEAttention",
                    "rope_theta": 10000.0,
                    "feat_sizes": [64, 64],
                    "embedding_dim": 256,
                    "num_heads": 1,
                    "downsample_rate": 1,
                    "dropout": 0.1,
                },
                "d_model": 256,
                "pos_enc_at_cross_attn_keys": True,
                "pos_enc_at_cross_attn_queries": False,
                "cross_attention": {
                    "_target_": "sam2.modeling.sam.transformer.RoPEAttention",
                    "rope_theta": 10000.0,
                    "feat_sizes": [64, 64],
                    "rope_k_repeat": True,
                    "embedding_dim": 256,
                    "num_heads": 1,
                    "downsample_rate": 1,
                    "dropout": 0.1,
                    "kv_in_dim": 64,
                },
            },
            "num_layers": 4,
        },
        "memory_encoder": {
            "_target_": "sam2.modeling.memory_encoder.MemoryEncoder",
            "out_dim": 64,
            "position_encoding": {
                "_target_": "sam2.modeling.position_encoding.PositionEmbeddingSine",
                "num_pos_feats": 64,
                "normalize": True,
                "scale": None,
                "temperature": 10000,
            },
            "mask_downsampler": {
                "_target_": "sam2.modeling.memory_encoder.MaskDownSampler",
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
            },
            "fuser": {
                "_target_": "sam2.modeling.memory_encoder.Fuser",
                "layer": {
                    "_target_": "sam2.modeling.memory_encoder.CXBlock",
                    "dim": 256,
                    "kernel_size": 7,
                    "padding": 3,
                    "layer_scale_init_value": 1e-6,
                    "use_dwconv": True,
                },
                "num_layers": 2,
            },
        },
        "num_maskmem": 7,
        "image_size": 1024,
        "sigmoid_scale_for_mem_enc": 20.0,
        "sigmoid_bias_for_mem_enc": -10.0,
        "use_mask_input_as_output_without_sam": True,
        "directly_add_no_mem_embed": True,
        "no_obj_embed_spatial": True,
        "use_high_res_features_in_sam": True,
        "multimask_output_in_sam": True,
        "iou_prediction_use_sigmoid": True,
        "use_obj_ptrs_in_encoder": True,
        "add_tpos_enc_to_obj_ptrs": True,
        "proj_tpos_enc_in_obj_ptrs": True,
        "use_signed_tpos_enc_to_obj_ptrs": True,
        "only_obj_ptrs_in_the_past_for_eval": True,
        "pred_obj_scores": True,
        "pred_obj_scores_mlp": True,
        "fixed_no_obj_ptr": True,
        "multimask_output_for_tracking": True,
        "use_multimask_token_for_obj_ptr": True,
        "multimask_min_pt_num": 0,
        "multimask_max_pt_num": 1,
        "use_mlp_for_obj_ptr_proj": True,
        "compile_image_encoder": False,
        # Same extras as build_sam2(..., apply_postprocessing=True) for image AMG.
        "sam_mask_decoder_extra_args": {
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    }
}


def build_sam2_hiera_large(
    ckpt_path: str,
    device: str = "cuda",
    mode: str = "eval",
) -> torch.nn.Module:
    """Build SAM2.1 Hiera-L from in-code cfg and load `ckpt_path`."""
    cfg = OmegaConf.create(SAM2_1_HIERA_L_CFG)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(sd)
    if missing_keys:
        logging.error(missing_keys)
        raise RuntimeError(f"missing keys when loading {ckpt_path}")
    if unexpected_keys:
        logging.error(unexpected_keys)
        raise RuntimeError(f"unexpected keys when loading {ckpt_path}")

    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model
