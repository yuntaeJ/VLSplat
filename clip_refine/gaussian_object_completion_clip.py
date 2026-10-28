# CLIP-based gaussian_add gate optimization on synthetic orbit poses (no COLMAP).
#
# Usage:
#   python clip_refine/gaussian_object_completion_clip.py -m output/exp/8b5caf3398 \
#     --filtered_ply ... --roi_ply ... --iters 1200

import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# scene / utils / arguments / gaussian_renderer come from the repo root (shared with training)
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state, inverse_sigmoid


# ----------------------------
# 0) Gaussian normalizer (xyz + scale together)
# ----------------------------
@dataclass
class NormalizeParams:
    """Params for reverting normalization (save for inverse)."""
    cen: np.ndarray      # (3,)
    scale: float
    log_s: float        # log(s) for _scaling correction

def normalize_gaussians(gs_in: GaussianModel, gs_add: GaussianModel,
                       target_extent: float = 0.5) -> Tuple[GaussianModel, GaussianModel, NormalizeParams]:
    """
    Normalize xyz and scaling. 3DGS _scaling is log-space: scaling' = _scaling + log(s).
    cen = (min+max)/2, val = max extent, s = target_extent / val.
    """
    x_in = gs_in.get_xyz.detach().cpu().numpy()
    x_add = gs_add.get_xyz.detach().cpu().numpy()
    X_all = np.vstack([x_in, x_add])

    mn = X_all.min(axis=0)
    mx = X_all.max(axis=0)
    cen = (mn + mx) / 2
    ext = np.max(mx - mn)
    val = max(ext, 1e-6)
    s = target_extent / val
    log_s = float(np.log(s))

    # xyz
    cen_t = torch.tensor(cen, dtype=torch.float32, device=gs_in.get_xyz.device)
    x_in_n = (gs_in.get_xyz - cen_t) * s
    x_add_n = (gs_add.get_xyz - cen_t) * s

    # scaling: _scaling is log(σ), σ' = σ*s => _scaling' = _scaling + log(s)
    gs_in_n = GaussianModel(gs_in.max_sh_degree)
    gs_in_n._xyz = nn.Parameter(x_in_n)
    gs_in_n._features_dc = nn.Parameter(gs_in._features_dc.detach().clone())
    gs_in_n._features_rest = nn.Parameter(gs_in._features_rest.detach().clone())
    gs_in_n._scaling = nn.Parameter(gs_in._scaling.detach() + log_s)
    gs_in_n._rotation = nn.Parameter(gs_in._rotation.detach().clone())
    gs_in_n._opacity = nn.Parameter(gs_in._opacity.detach().clone())
    gs_in_n._objects_dc = nn.Parameter(gs_in._objects_dc.detach().clone())
    gs_in_n.active_sh_degree = gs_in.active_sh_degree

    gs_add_n = GaussianModel(gs_add.max_sh_degree)
    gs_add_n._xyz = nn.Parameter(x_add_n)
    gs_add_n._features_dc = nn.Parameter(gs_add._features_dc.detach().clone())
    gs_add_n._features_rest = nn.Parameter(gs_add._features_rest.detach().clone())
    gs_add_n._scaling = nn.Parameter(gs_add._scaling.detach() + log_s)
    gs_add_n._rotation = nn.Parameter(gs_add._rotation.detach().clone())
    gs_add_n._opacity = nn.Parameter(gs_add._opacity.detach().clone())
    gs_add_n._objects_dc = nn.Parameter(gs_add._objects_dc.detach().clone())
    gs_add_n.active_sh_degree = gs_add.active_sh_degree

    params = NormalizeParams(cen=cen, scale=s, log_s=log_s)
    return gs_in_n, gs_add_n, params


def unnormalize_gaussians(gs: GaussianModel, params: NormalizeParams, device: str = "cuda") -> GaussianModel:
    """Revert normalization for export."""
    cen_t = torch.tensor(params.cen, dtype=torch.float32, device=device)
    gs_out = GaussianModel(gs.max_sh_degree)
    xyz = gs.get_xyz
    if xyz.shape[0] == 0:
        gs_out._xyz = nn.Parameter(torch.empty(0, 3, device=device, dtype=xyz.dtype))
    else:
        gs_out._xyz = nn.Parameter(xyz / params.scale + cen_t)
    gs_out._features_dc = nn.Parameter(gs._features_dc.detach().clone())
    gs_out._features_rest = nn.Parameter(gs._features_rest.detach().clone())
    gs_out._scaling = nn.Parameter(gs._scaling.detach() - params.log_s)
    gs_out._rotation = nn.Parameter(gs._rotation.detach().clone())
    gs_out._opacity = nn.Parameter(gs._opacity.detach().clone())
    gs_out._objects_dc = nn.Parameter(gs._objects_dc.detach().clone())
    gs_out.active_sh_degree = gs.active_sh_degree
    return gs_out


# ----------------------------
# 1) Synthetic pose sampler & renderer (import from render_synthetic)
# ----------------------------
def _get_synthetic_components():
    """Import from render_synthetic: orbit camera, gray-bg render, roi_crop."""
    from render_synthetic import SyntheticPoseSampler, SimpleRenderer, roi_crop_from_alpha
    return SyntheticPoseSampler, SimpleRenderer, roi_crop_from_alpha


# ----------------------------
# 2) CLIP wrapper
# ----------------------------
class CLIPWrapper:
    def __init__(self, device="cuda", model_name="ViT-L-14", pretrained="openai"):
        import open_clip
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device).eval()
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        self.image_size = self._resolve_image_size()

    def _resolve_image_size(self) -> int:
        size = getattr(self.model.visual, "image_size", CLIP_INPUT_SIZE)
        if isinstance(size, (tuple, list)):
            return int(size[0])
        return int(size)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        feat = self.model.encode_text(tokens)
        return feat / feat.norm(dim=-1, keepdim=True)

    def encode_image(self, rgb_bchw: torch.Tensor) -> torch.Tensor:
        x = rgb_bchw
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feat = self.model.encode_image(x)
        return feat / feat.norm(dim=-1, keepdim=True)


def _normalize_prompt_list(prompts: Any) -> List[str]:
    if prompts is None:
        return []
    if isinstance(prompts, str):
        vals = [prompts]
    elif isinstance(prompts, (list, tuple)):
        vals = list(prompts)
    else:
        vals = [str(prompts)]
    out: List[str] = []
    seen: set[str] = set()
    for val in vals:
        txt = str(val).strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


# ----------------------------
# 3) Utils
# ----------------------------
CLIP_INPUT_SIZE = 224


def rgba_to_white_bg_rgb(rgba: torch.Tensor) -> torch.Tensor:
    """Composite RGBA onto white so transparent splats become white."""
    rgb = rgba[:3].clamp(0, 1)
    if rgba.shape[0] < 4:
        return rgb
    alpha = rgba[3:4].clamp(0, 1)
    return (rgb * alpha + (1.0 - alpha)).clamp(0, 1)


def _clip_full_rgb(rgba: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(
        rgba_to_white_bg_rgb(rgba).unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )


def build_merged_model_once(gs_in: GaussianModel, gs_add: GaussianModel, device: str) -> GaussianModel:
    merged = GaussianModel(gs_in.max_sh_degree)
    merged._xyz = nn.Parameter(torch.cat([gs_in.get_xyz, gs_add.get_xyz], dim=0).detach().clone())
    merged._features_dc = nn.Parameter(torch.cat([gs_in._features_dc, gs_add._features_dc], dim=0).detach().clone())
    merged._features_rest = nn.Parameter(torch.cat([gs_in._features_rest, gs_add._features_rest], dim=0).detach().clone())
    merged._scaling = nn.Parameter(torch.cat([gs_in._scaling, gs_add._scaling], dim=0).detach().clone())
    merged._rotation = nn.Parameter(torch.cat([gs_in._rotation, gs_add._rotation], dim=0).detach().clone())
    merged._opacity = nn.Parameter(torch.cat([gs_in._opacity, gs_add._opacity], dim=0).detach().clone())
    merged._objects_dc = nn.Parameter(torch.cat([gs_in._objects_dc, gs_add._objects_dc], dim=0).detach().clone())
    merged.active_sh_degree = gs_in.active_sh_degree
    merged.max_radii2D = torch.zeros(merged.get_xyz.shape[0], device=device)
    return merged


def apply_gate_to_opacity_raw(opacity_raw: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(opacity_raw).squeeze(-1)
    p_eff = (p * gate).clamp(1e-6, 1 - 1e-6)
    return inverse_sigmoid(p_eff).unsqueeze(-1)


# ----------------------------
# 4) Gate optimization (synthetic poses only)
# ----------------------------
def optimize_gates(
    renderer: Any,
    pose_sampler: Any,
    clipm: CLIPWrapper,
    gs_in: GaussianModel,
    gs_add: GaussianModel,
    prompt_pos: Any = "a photo of a chair",
    prompt_neg: Optional[Any] = None,
    negative_weight: float = 1.0,
    iters: int = 1200,
    views_per_iter: int = 2,
    lr: float = 5e-2,
    tau_gate: float = 1.0,
    lam_sparse: float = 1e-4,
    gate_warm_start: float = 0.3,
    lam_sparse_warmup_iters: int = 100,
    debug_save_every: int = 0,
    debug_dir: Optional[str] = None,
    save_every: int = 0,
    save_dir: Optional[str] = None,
    save_name_prefix: str = "run",
    gate_thr: float = 0.5,
    norm_params: Optional[Any] = None,
    device: str = "cuda",
    roi_mask_thr: float = 0.15,
    roi_expand_ratio: float = 1.2,
    roi_fill: str = "background",
    roi_crop_fn: Any = None,
    log_every: int = 10,
    use_crop: bool = False,
    n_fixed_views: Optional[int] = None,
    pair_opposite_views: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Full-frame CLIP encodes only; each step samples 2 novel + 2 opposite cameras.
    # The crop-related and n_fixed_views arguments are accepted for compatibility and ignored.
    del roi_mask_thr, roi_expand_ratio, roi_fill, roi_crop_fn, use_crop, n_fixed_views
    def _gate_histogram_summary(gate_values: torch.Tensor) -> str:
        gv = gate_values.detach().reshape(-1).float()
        if gv.numel() == 0:
            return "gate_hist(empty)"
        c_00_03 = int(((gv >= 0.0) & (gv < 0.3)).sum().item())
        c_03_04 = int(((gv >= 0.3) & (gv < 0.4)).sum().item())
        c_04_05 = int(((gv >= 0.4) & (gv < 0.5)).sum().item())
        c_05_06 = int(((gv >= 0.5) & (gv < 0.6)).sum().item())
        c_06_07 = int(((gv >= 0.6) & (gv < 0.7)).sum().item())
        c_07_10 = int((gv >= 0.7).sum().item())
        return (
            "gate_hist "
            f"0-0.3:{c_00_03}, 0.3-0.4:{c_03_04}, 0.4-0.5:{c_04_05}, "
            f"0.5-0.6:{c_05_06}, 0.6-0.7:{c_06_07}, 0.7-1.0:{c_07_10}"
        )

    n_in = gs_in.get_xyz.shape[0]
    n_add = gs_add.get_xyz.shape[0]

    merged = build_merged_model_once(gs_in, gs_add, device)
    for attr in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "_objects_dc"]:
        getattr(merged, attr).requires_grad_(False)

    gate_logit = inverse_sigmoid(torch.tensor(gate_warm_start, device=device).clamp(1e-6, 1 - 1e-6))
    gate_logit = gate_logit.expand(n_add).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([gate_logit], lr=lr)

    pos_prompts = _normalize_prompt_list(prompt_pos)
    if not pos_prompts:
        pos_prompts = ["a photo of an object"]
    neg_prompts = _normalize_prompt_list(prompt_neg)
    with torch.no_grad():
        txt_pos = clipm.encode_text(pos_prompts)
        txt_neg = clipm.encode_text(neg_prompts) if neg_prompts else None

    base_op_add_raw = gs_add._opacity.detach().clone()
    attr_ema = torch.zeros((n_add,), device=device)

    n_step = max(int(views_per_iter), 1)
    use_pairs = bool(pair_opposite_views) and hasattr(pose_sampler, "sample_novel_and_opposite")
    n_novel_per_step = max(n_step // 2, 1) if use_pairs else n_step
    if use_pairs:
        n_step = n_novel_per_step * 2
    clip_size = int(getattr(clipm, "image_size", CLIP_INPUT_SIZE) or CLIP_INPUT_SIZE)
    pair_note = (
        f"; {n_novel_per_step} novel + {n_novel_per_step} opposite (az+180)" if use_pairs else ""
    )
    verbose = int(log_every) > 0
    if verbose:
        print(
            f"[clip-gate] {n_step}/step x {int(iters)} steps{pair_note}; "
            f"no IN cache; full-frame CLIP {clip_size}x{clip_size}",
            flush=True,
        )

    for it in range(1, iters + 1):
        opt.zero_grad(set_to_none=True)
        g = torch.sigmoid(gate_logit / tau_gate)

        op_add_raw_gated = apply_gate_to_opacity_raw(base_op_add_raw, g)
        op_add_activated = op_add_raw_gated.sigmoid()
        op_merged_override = torch.cat([merged._opacity[:n_in].sigmoid().detach(), op_add_activated], dim=0)

        if use_pairs:
            cams = pose_sampler.sample_novel_and_opposite(
                n_novel_per_step, gs_in, renderer.render_rgba
            )
        else:
            cams = pose_sampler.sample(n_step, gs_in, renderer.render_rgba)
        if len(cams) == 0:
            continue

        delta_s_list: List[torch.Tensor] = []
        rgb_full_batch: List[torch.Tensor] = []
        rgb_base_full_batch: List[torch.Tensor] = []
        debug_first = None
        keep_debug = debug_save_every > 0 and bool(debug_dir)
        for cam in cams:
            rgba_merged = renderer.render_rgba(cam, merged, override_opacity=op_merged_override)
            rgb_full = _clip_full_rgb(rgba_merged, clip_size)
            rgb_full_batch.append(rgb_full)
            with torch.no_grad():
                rgba_base = renderer.render_rgba(cam, gs_in)
                rgb_base_full_batch.append(_clip_full_rgb(rgba_base, clip_size))
            if keep_debug and debug_first is None:
                debug_first = {
                    "rgba_merged": rgba_merged,
                    "rgba_base": rgba_base,
                    "rgb_full": rgb_full,
                }

        if len(rgb_full_batch) == 0:
            continue

        feat_full = clipm.encode_image(torch.cat(rgb_full_batch, dim=0))
        s_pos_full = (feat_full @ txt_pos.t()).mean(dim=-1)
        with torch.no_grad():
            feat_base_full = clipm.encode_image(torch.cat(rgb_base_full_batch, dim=0))
            s_pos_base_full = (feat_base_full @ txt_pos.t()).mean(dim=-1)
            s_neg_base_full = (
                (feat_base_full @ txt_neg.t()).mean(dim=-1) if txt_neg is not None else None
            )
        d_full = s_pos_full - s_pos_base_full
        improvement = d_full

        L_clip = -improvement.mean()
        if txt_neg is not None and s_neg_base_full is not None:
            s_neg_full = (feat_full @ txt_neg.t()).mean(dim=-1)
            L_clip = L_clip + float(negative_weight) * (s_neg_full - s_neg_base_full).mean()
        delta_s_list = [d_full.detach()]

        if keep_debug and debug_save_every > 0 and it % debug_save_every == 0 and debug_first is not None:
            try:
                import torchvision
                os.makedirs(debug_dir, exist_ok=True)
                cam_idx = 0
                sim_full_float = float(s_pos_full[0].detach().item()) if s_pos_full.numel() > 0 else 0.0
                sim_str = f"sim{sim_full_float:.3f}"
                torchvision.utils.save_image(
                    rgba_to_white_bg_rgb(debug_first["rgba_merged"]).unsqueeze(0),
                    os.path.join(debug_dir, f"merged_white_i{it:05d}_cam{cam_idx}_{sim_str}.png"),
                )
                torchvision.utils.save_image(
                    debug_first["rgb_full"],
                    os.path.join(debug_dir, f"clip_full_{clip_size}_i{it:05d}_cam{cam_idx}_{sim_str}.png"),
                )
                torchvision.utils.save_image(
                    debug_first["rgba_base"][3:4].repeat(3, 1, 1).unsqueeze(0),
                    os.path.join(debug_dir, f"base_alpha_i{it:05d}_cam{cam_idx}.png"),
                )
            except Exception as e:
                print(f"[debug save] failed: {e}", file=sys.stderr)

        lam = lam_sparse * min(1.0, it / max(1, lam_sparse_warmup_iters))
        L_sparse = lam * g.sum()
        loss = L_clip + L_sparse
        loss.backward()
        opt.step()

        # Log every `log_every` iterations (and always at the last step). log_every<=0 disables.
        if verbose:
            log_step = max(1, int(log_every))
            if (it % log_step == 0) or (it == iters):
                with torch.no_grad():
                    sp = (g >= gate_thr).sum().item()
                    sp_ratio = float(sp) / float(max(1, n_add))
                    d_cos = float(torch.stack(delta_s_list).mean()) if delta_s_list else 0.0
                    impr = float((-L_clip).detach())
                    thr_tag = f"g>={gate_thr:.2f}"
                    g_min = float(g.min().item()) if g.numel() > 0 else 0.0
                    g_max = float(g.max().item()) if g.numel() > 0 else 0.0
                    g_std = float(g.std(unbiased=False).item()) if g.numel() > 0 else 0.0
                    g_med = float(torch.quantile(g, 0.50).item()) if g.numel() > 0 else 0.0
                    print(
                        f"[{it:4d}/{iters}] loss={float(loss):.4f} L_clip={float(L_clip):.4f} impr={impr:.4f} "
                        f"d_cos_full={d_cos:.4f} L_sparse={float(L_sparse):.4f} g_mean={float(g.mean()):.3f} "
                        f"g_std={g_std:.3f} g_min={g_min:.3f} g_med={g_med:.3f} "
                        f"g_q95={float(torch.quantile(g, 0.95)):.3f} g_max={g_max:.3f} "
                        f"#{thr_tag}={int(sp)}/{int(n_add)} ({sp_ratio:.3f})",
                        flush=True,
                    )
                    print(f"           {_gate_histogram_summary(g)}", flush=True)
            if it % 200 == 0 and hasattr(pose_sampler, "_reject_count"):
                print(f"  [pose] reject={pose_sampler._reject_count} accept={pose_sampler._accept_count}")

        if save_every > 0 and save_dir and it % save_every == 0:
            with torch.no_grad():
                n_sel = int((g >= gate_thr).sum().item())
                print(
                    f"[save iter {it}/{iters}] gate_thr={gate_thr} selected_gaussians={n_sel} (G_add, g>={gate_thr})",
                    flush=True,
                )
                gs_sel = threshold_select_gaussians(gs_add, g, thr=gate_thr)
                gs_ply = unnormalize_gaussians(gs_sel, norm_params, device) if norm_params is not None else gs_sel
                os.makedirs(save_dir, exist_ok=True)
                stem = save_name_prefix
                ply_path = os.path.join(save_dir, f"{stem}_iter{it:05d}_n{n_sel}_gadd.ply")
                gs_ply.save_ply(ply_path)
                png_in_add: Optional[str] = None
                png_add_only: Optional[str] = None
                cams_sv = pose_sampler.sample(1, gs_in, renderer.render_rgba)
                if len(cams_sv) > 0:
                    cam = cams_sv[0]
                    try:
                        import torchvision

                        # (1) G_in + G_add points with g>=gate_thr only (hard subset merge)
                        merged_sel = build_merged_model_once(gs_in, gs_sel, device)
                        for attr in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "_objects_dc"]:
                            getattr(merged_sel, attr).requires_grad_(False)
                        rgba_in_add = renderer.render_rgba(cam, merged_sel)
                        rgb_in_add = rgba_to_white_bg_rgb(rgba_in_add)
                        png_in_add = os.path.join(
                            save_dir, f"{stem}_iter{it:05d}_n{n_sel}_in_plus_add_thr.png"
                        )
                        torchvision.utils.save_image(
                            rgb_in_add.unsqueeze(0).cpu().clamp(0, 1), png_in_add
                        )

                        # (2) G_add only (same threshold), no base
                        if n_sel > 0:
                            rgba_add_only = renderer.render_rgba(cam, gs_sel)
                            rgb_add_only = rgba_to_white_bg_rgb(rgba_add_only)
                            png_add_only = os.path.join(
                                save_dir, f"{stem}_iter{it:05d}_n{n_sel}_add_only_thr.png"
                            )
                            torchvision.utils.save_image(
                                rgb_add_only.unsqueeze(0).cpu().clamp(0, 1), png_add_only
                            )
                        else:
                            print("[save render] add_only_thr skipped (n_sel=0)", flush=True)
                    except Exception as e:
                        print(f"[save render] failed: {e}", file=sys.stderr)
                else:
                    print("[save render] skipped (no camera sample)", flush=True)
                extra = []
                if png_in_add:
                    extra.append(f"in+add>thr={png_in_add}")
                if png_add_only:
                    extra.append(f"add_only>thr={png_add_only}")
                if extra:
                    print(f"[save] ply={ply_path} " + " ".join(extra), flush=True)
                else:
                    print(f"[save] ply={ply_path}", flush=True)

    with torch.no_grad():
        g_final = torch.sigmoid(gate_logit / tau_gate)
    return g_final, attr_ema


def refine_gaussian_object_features(
    gs_add: GaussianModel,
    gates: torch.Tensor,
    gate_thr: float,
    push_strength: float = 1.0,
) -> Tuple[int, int]:
    """
    Refine G_add object features from final gate score after warmup.

    - high gate (>= gate_thr): set feature to selected-cluster mean
    - low gate  (< gate_thr): push away from that mean direction
    """
    with torch.no_grad():
        g = gates.detach()
        high = g >= gate_thr
        low = ~high

        n_high = int(high.sum().item())
        n_low = int(low.sum().item())
        if n_high == 0:
            return 0, n_low

        obj = gs_add._objects_dc.detach().clone()
        obj_mean = obj[high].mean(dim=0, keepdim=True)

        # High-score Gaussians collapse to cluster mean feature.
        obj[high] = obj_mean.expand_as(obj[high])
        if n_low > 0:
            # Low-score Gaussians are moved to the opposite side of the mean.
            obj[low] = obj[low] + float(push_strength) * (obj[low] - obj_mean)

        gs_add._objects_dc = nn.Parameter(obj)
        return n_high, n_low


def keep_select_gaussians(gs_add: GaussianModel, keep: torch.Tensor) -> GaussianModel:
    """Copy of the Gaussians where ``keep`` is True (empty subset allowed)."""
    keep_t = keep.reshape(-1).detach().to(device=gs_add.get_xyz.device)
    if keep_t.dtype != torch.bool:
        keep_t = keep_t.bool()
    subset = GaussianModel(gs_add.max_sh_degree)
    if int(keep_t.sum().item()) == 0:
        dev = gs_add._xyz.device
        subset._xyz = nn.Parameter(torch.empty(0, 3, device=dev, dtype=gs_add._xyz.dtype))
        subset._features_dc = nn.Parameter(torch.empty(0, *gs_add._features_dc.shape[1:], device=dev, dtype=gs_add._features_dc.dtype))
        subset._features_rest = nn.Parameter(torch.empty(0, *gs_add._features_rest.shape[1:], device=dev, dtype=gs_add._features_rest.dtype))
        subset._scaling = nn.Parameter(torch.empty(0, 3, device=dev, dtype=gs_add._scaling.dtype))
        subset._rotation = nn.Parameter(torch.empty(0, 4, device=dev, dtype=gs_add._rotation.dtype))
        subset._opacity = nn.Parameter(torch.empty(0, 1, device=dev, dtype=gs_add._opacity.dtype))
        subset._objects_dc = nn.Parameter(torch.empty(0, *gs_add._objects_dc.shape[1:], device=dev, dtype=gs_add._objects_dc.dtype))
        subset.active_sh_degree = gs_add.active_sh_degree
        return subset
    subset._xyz = nn.Parameter(gs_add._xyz[keep_t].detach().clone())
    subset._features_dc = nn.Parameter(gs_add._features_dc[keep_t].detach().clone())
    subset._features_rest = nn.Parameter(gs_add._features_rest[keep_t].detach().clone())
    subset._scaling = nn.Parameter(gs_add._scaling[keep_t].detach().clone())
    subset._rotation = nn.Parameter(gs_add._rotation[keep_t].detach().clone())
    subset._opacity = nn.Parameter(gs_add._opacity[keep_t].detach().clone())
    subset._objects_dc = nn.Parameter(gs_add._objects_dc[keep_t].detach().clone())
    subset.active_sh_degree = gs_add.active_sh_degree
    return subset


def threshold_select_gaussians(gs_add: GaussianModel, g: torch.Tensor, thr: float = 0.6) -> GaussianModel:
    return keep_select_gaussians(gs_add, g >= thr)


# ----------------------------
# 5) main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="CLIP gate selection with synthetic orbit poses")
    parser.add_argument("-m", "--model_path", required=True, help="Model path for renderer")
    parser.add_argument("--filtered_ply", required=True, help="G_input PLY (filtered)")
    parser.add_argument("--roi_ply", required=True, help="G_add PLY (ROI candidates)")
    parser.add_argument("--out_ply", type=str, default=None)
    parser.add_argument("--normalize", action="store_true", help="Normalize xyz+scale before optimization")
    parser.add_argument("--iters", type=int, default=1200)
    parser.add_argument("--gate_thr", type=float, default=0.6)
    parser.add_argument("--prompt", type=str, default="a photo of a chair")
    parser.add_argument("--clip_model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--views_per_iter", type=int, default=2)
    parser.add_argument("--gate_warm_start", type=float, default=0.3)
    parser.add_argument("--lam_sparse", type=float, default=1e-4)
    parser.add_argument("--debug_save_every", type=int, default=0)
    parser.add_argument("--debug_dir", type=str, default=None)
    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
        help="Save G_add PLY (g>=gate_thr) + merged render PNG every N iters (0=off)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="Directory for periodic PLY/PNG saves and default final gated PLY",
    )
    parser.add_argument(
        "--save_name_prefix",
        type=str,
        default=None,
        help="Filename prefix for periodic saves (default: ROI ply stem)",
    )
    parser.add_argument("--no_reject", action="store_true", help="Accept all synthetic poses (no alpha rejection)")
    parser.add_argument("--radius_min", type=float, default=None, help="Optional synthetic orbit radius minimum")
    parser.add_argument("--radius_max", type=float, default=None, help="Optional synthetic orbit radius maximum")
    parser.add_argument("--roi_mask_thr", type=float, default=0.15, help="Alpha threshold for ROI crop")
    parser.add_argument("--roi_expand_ratio", type=float, default=1.2, help="ROI bbox expand ratio")
    parser.add_argument("--roi_fill", type=str, default="background", choices=["mean", "background"], help="Fill for ROI crop")
    parser.add_argument(
        "--feature_refine_push_strength",
        type=float,
        default=1.0,
        help="Push strength for low-score Gaussian object features away from cluster mean.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    safe_state(False)  # False = show prints (True silences stdout)

    if not os.path.isfile(args.filtered_ply) or not os.path.isfile(args.roi_ply):
        print("Error: PLY not found", file=sys.stderr)
        sys.exit(1)

    gs_in = GaussianModel(3)
    gs_in.load_ply(args.filtered_ply)
    gs_add = GaussianModel(3)
    gs_add.load_ply(args.roi_ply)
    norm_params = None

    if args.normalize:
        gs_in, gs_add, norm_params = normalize_gaussians(gs_in, gs_add, target_extent=0.5)
        print("[normalize] applied xyz+scale normalization", flush=True)

    print(f"[load] gs_in: {gs_in.get_xyz.shape[0]}, gs_add: {gs_add.get_xyz.shape[0]}", flush=True)

    SyntheticPoseSampler, SimpleRenderer, _roi_crop_unused = _get_synthetic_components()
    del _roi_crop_unused
    renderer = SimpleRenderer(model_path=os.path.abspath(args.model_path), device=device)
    renderer.set_background("white")
    scale_mod = 1.0
    renderer.set_scale_modifier(scale_mod)
    radius_fixed = None
    if args.radius_min is not None or args.radius_max is not None:
        if args.radius_min is None or args.radius_max is None:
            print("Error: --radius_min and --radius_max must be provided together", file=sys.stderr)
            sys.exit(2)
        if args.radius_min <= 0 or args.radius_max <= 0:
            print("Error: --radius_min/--radius_max must be > 0", file=sys.stderr)
            sys.exit(2)
        if args.radius_min > args.radius_max:
            print("Error: --radius_min cannot be greater than --radius_max", file=sys.stderr)
            sys.exit(2)
        radius_fixed = (float(args.radius_min), float(args.radius_max))

    pose_sampler = SyntheticPoseSampler(
        width=224,
        height=224,
        fov_deg=50.0,
        device=device,
        no_reject=bool(getattr(args, "no_reject", False)),
        radius_fixed=radius_fixed,
    )
    print(f"[synthetic] orbit poses + white bg + full-frame CLIP {CLIP_INPUT_SIZE}x{CLIP_INPUT_SIZE}", flush=True)
    if radius_fixed is not None:
        print(f"[synthetic] fixed radius range: [{radius_fixed[0]:.4f}, {radius_fixed[1]:.4f}]", flush=True)

    try:
        clipm = CLIPWrapper(device=device, model_name=args.clip_model_name, pretrained=args.clip_pretrained)
        print(
            f"[CLIP] model={args.clip_model_name} pretrained={args.clip_pretrained} "
            f"native_input={clipm.image_size} requested_input={CLIP_INPUT_SIZE} "
            f"(supports multi-positive / optional negative prompts)",
            flush=True,
        )
    except Exception as e:
        print(f"[CLIP] pip install open_clip_torch. Error: {e}", file=sys.stderr)
        sys.exit(1)

    debug_dir = args.debug_dir
    if args.debug_save_every > 0 and debug_dir is None:
        debug_dir = os.path.join(os.path.dirname(args.roi_ply), "gate_selected", "clip_samples")
        print(f"[debug] Saving CLIP sample images every {args.debug_save_every} iters -> {debug_dir}")

    save_dir_abs = os.path.abspath(args.save_dir)
    save_prefix = args.save_name_prefix or os.path.splitext(os.path.basename(args.roi_ply))[0]
    if args.save_every > 0:
        print(
            f"[save] Every {args.save_every} iters -> {save_dir_abs} "
            f"(G_add with g>={args.gate_thr}, prefix={save_prefix})",
            flush=True,
        )

    g, attr_ema = optimize_gates(
        renderer=renderer,
        pose_sampler=pose_sampler,
        clipm=clipm,
        gs_in=gs_in,
        gs_add=gs_add,
        prompt_pos=args.prompt,
        iters=args.iters,
        views_per_iter=args.views_per_iter,
        gate_warm_start=args.gate_warm_start,
        lam_sparse=args.lam_sparse,
        debug_save_every=args.debug_save_every,
        debug_dir=debug_dir,
        save_every=args.save_every,
        save_dir=save_dir_abs if args.save_every > 0 else None,
        save_name_prefix=save_prefix,
        gate_thr=args.gate_thr,
        norm_params=norm_params,
        device=device,
    )

    n_high, n_low = refine_gaussian_object_features(
        gs_add=gs_add,
        gates=g,
        gate_thr=args.gate_thr,
        push_strength=float(args.feature_refine_push_strength),
    )
    print(
        f"[feature refine] applied after gate optimization "
        f"(iters={args.iters}, high={n_high}, low={n_low}, gate_thr={args.gate_thr}, push={args.feature_refine_push_strength})",
        flush=True,
    )

    gs_selected = threshold_select_gaussians(gs_add, g, thr=args.gate_thr)
    n_add = gs_add.get_xyz.shape[0]
    n_selected = gs_selected.get_xyz.shape[0]
    print(f"[gate select] G_add: {n_add} candidates -> {n_selected} new gaussians added (threshold={args.gate_thr})", flush=True)

    if norm_params is not None:
        gs_selected = unnormalize_gaussians(gs_selected, norm_params, device)

    out_path = args.out_ply
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.roi_ply))[0]
        os.makedirs(save_dir_abs, exist_ok=True)
        out_path = os.path.join(save_dir_abs, f"{base}_gated.ply")

    gs_selected.save_ply(out_path)
    print(f"[save] {out_path} ({n_selected} gaussians)", flush=True)


if __name__ == "__main__":
    main()
