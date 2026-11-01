"""Scene-graph-driven intra-/inter-node refinement during VLSplat training.

Every refinement round (see ``run_scene_graph_refinement_cycle``):
  1. re-fit ``bbox_tight`` of every target node with the ROI recipe of the graph construction
     (cosine filter, anisotropic grouping, spatial crop);
  2. intra-node: per target node, a CLIP gate (text_prompt) over the node's own Gaussians picks the ones to
     keep and pushes the others away from the node's codebook entry;
  3. inter-node: the same for neighbouring leftover Gaussians inside the node's expanded box.

The CLIP gate and the feature push come from clip_refine/gaussian_object_completion_clip.py. Only the
Gaussians' object features and opacities change, never their geometry.
The step after the last iteration lives in finalize_scene_graph.py.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from scene_graph_builder.graph_io import expand_bbox, node_group_ids

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)
_CLIP_REFINE = os.path.join(_REPO_ROOT, "clip_refine")

_goc_clip_refine = None
_clip_refine_import_path_ready = False
_clip_wrapper_cache: Dict[Tuple[str, str, str], Any] = {}
_renderer_cache: Dict[Tuple[str, str], Any] = {}



def _ensure_clip_refine_import_path() -> None:
    """Ensure clip_refine Python modules are importable."""
    global _clip_refine_import_path_ready
    if _clip_refine_import_path_ready:
        return
    sys.path.insert(0, _CLIP_REFINE)
    _clip_refine_import_path_ready = True


def _load_goc_clip_refine():
    global _goc_clip_refine
    if _goc_clip_refine is not None:
        return _goc_clip_refine
    _ensure_clip_refine_import_path()
    path = os.path.join(_CLIP_REFINE, "gaussian_object_completion_clip.py")
    spec = importlib.util.spec_from_file_location("goc_clip_refine", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["goc_clip_refine"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _goc_clip_refine = mod
    return mod


def _sg_quiet(cfg: Optional[Dict[str, Any]] = None) -> bool:
    return bool((cfg or {}).get("quiet_node_refine", False))


def _sg_print(cfg: Optional[Dict[str, Any]], msg: str) -> None:
    if _sg_quiet(cfg):
        return
    print(msg, flush=True)


def _get_clip_wrapper(cfg: Dict[str, Any], device: str):
    goc = _load_goc_clip_refine()
    CLIPWrapper = goc.CLIPWrapper
    model_name = str(cfg["clip_model"])
    pretrained = str(cfg["clip_pretrained"])
    key = (str(device), model_name, pretrained)
    clipm = _clip_wrapper_cache.get(key)
    if clipm is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="These pretrained weights were trained with QuickGELU activation",
            )
            clipm = CLIPWrapper(device=device, model_name=model_name, pretrained=pretrained)
        _clip_wrapper_cache[key] = clipm
    return clipm


def _get_renderer(model_path: str, device: str):
    _ensure_clip_refine_import_path()
    from render_synthetic import SimpleRenderer

    key = (os.path.abspath(model_path), str(device))
    renderer = _renderer_cache.get(key)
    if renderer is None:
        renderer = SimpleRenderer(model_path=os.path.abspath(model_path), device=device)
        renderer.set_background("white")
        _renderer_cache[key] = renderer
    else:
        renderer.set_background("white")
    return renderer


def compute_group_per_point(gaussians, code_book: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (pred_class_id (N,), group_id (N)) on same device as gaussians."""
    n = int(gaussians.get_xyz.shape[0])
    f = gaussians._objects_dc.detach().reshape(n, -1)
    f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
    cb = code_book.detach().reshape(int(code_book.shape[0]), -1)
    logits = f @ cb.T
    pred_id = logits.argmax(dim=-1).reshape(-1).long()
    return pred_id, pred_id


def _filter_indices_by_bbox(
    xyz: torch.Tensor,
    idx: torch.Tensor,
    bbox: Optional[Dict[str, float]],
) -> torch.Tensor:
    """Return subset of idx whose xyz is inside bbox. If bbox missing, return idx as-is."""
    idx = idx.reshape(-1).long()
    if idx.numel() == 0 or not isinstance(bbox, dict):
        return idx
    pts = xyz[idx].detach().cpu()
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    inside = (
        (x >= float(bbox["xmin"]))
        & (x <= float(bbox["xmax"]))
        & (y >= float(bbox["ymin"]))
        & (y <= float(bbox["ymax"]))
        & (z >= float(bbox["zmin"]))
        & (z <= float(bbox["zmax"]))
    )
    return idx[inside.to(device=idx.device)]


def _bbox_tight(node: Dict[str, Any]) -> Optional[Dict[str, float]]:
    b = node.get("bbox_tight")
    return b if isinstance(b, dict) else None


def _bbox_expanded_from_tight(node: Dict[str, Any], margin_fraction: float) -> Optional[Dict[str, float]]:
    """inter-refine ADD volume: expand ``bbox_tight`` by ``margin_fraction`` on each side."""
    tight = _bbox_tight(node)
    return None if tight is None else expand_bbox(tight, float(margin_fraction))


def codebook_similarity_split(
    gaussians,
    code_book: torch.Tensor,
    gpp: torch.Tensor,
    idx_group: torch.Tensor,
    thr: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split indices by cosine similarity to the codebook row of each Gaussian's own group.
    IN: sim >= thr, ADD: sim < thr.
    """
    idx_group = idx_group.reshape(-1).long()
    if idx_group.numel() == 0:
        return idx_group, idx_group

    obj = gaussians._objects_dc[idx_group].detach().reshape(idx_group.shape[0], -1)
    obj = obj / (obj.norm(dim=-1, keepdim=True) + 1e-12)
    rows = code_book.detach().reshape(int(code_book.shape[0]), -1)[gpp[idx_group].long()]
    rows = rows / (rows.norm(dim=-1, keepdim=True) + 1e-12)
    sim = (obj * rows).sum(dim=-1)
    m_in = sim >= float(thr)
    return idx_group[m_in], idx_group[~m_in]


def indices_in_bbox_except_group_ids(
    xyz: torch.Tensor,
    gpp: torch.Tensor,
    group_id: int,
    bbox: Dict[str, float],
    exclude_group_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    # Prevent accidental huge broadcasting by normalizing shapes first.
    xyz = xyz.reshape(-1, 3)
    gpp = gpp.reshape(-1).long()
    n = min(int(xyz.shape[0]), int(gpp.shape[0]))
    xyz = xyz[:n]
    gpp = gpp[:n]

    xmin, xmax = bbox["xmin"], bbox["xmax"]
    ymin, ymax = bbox["ymin"], bbox["ymax"]
    zmin, zmax = bbox["zmin"], bbox["zmax"]
    # Compute mask on CPU to keep temporary memory small.
    xyz_cpu = xyz.detach().cpu()
    gpp_cpu = gpp.detach().cpu()
    x, y, z = xyz_cpu[:, 0], xyz_cpu[:, 1], xyz_cpu[:, 2]
    inside = (
        (x >= xmin)
        & (x <= xmax)
        & (y >= ymin)
        & (y <= ymax)
        & (z >= zmin)
        & (z <= zmax)
    )
    exclude_set = {int(group_id)}
    if exclude_group_ids:
        exclude_set.update(int(x) for x in exclude_group_ids)
    not_excluded = torch.ones_like(gpp_cpu, dtype=torch.bool)
    for gid in sorted(exclude_set):
        not_excluded &= gpp_cpu != int(gid)
    idx = torch.nonzero(inside & not_excluded, as_tuple=False).reshape(-1)
    return idx.to(device=xyz.device, dtype=torch.long)


def _node_prompt(node: Dict[str, Any]) -> str:
    """The CLIP prompt of a node: its VLM ``text_prompt`` (empty when the node has none)."""
    return str(node.get("text_prompt") or "").strip()


def _subset_gaussian(gaussians, idx: torch.Tensor):
    goc = _load_goc_clip_refine()
    GaussianModel = goc.GaussianModel
    idx = idx.long()
    out = GaussianModel(gaussians.max_sh_degree)
    out._xyz = nn.Parameter(gaussians._xyz[idx].detach().clone())
    out._features_dc = nn.Parameter(gaussians._features_dc[idx].detach().clone())
    out._features_rest = nn.Parameter(gaussians._features_rest[idx].detach().clone())
    out._scaling = nn.Parameter(gaussians._scaling[idx].detach().clone())
    out._rotation = nn.Parameter(gaussians._rotation[idx].detach().clone())
    out._opacity = nn.Parameter(gaussians._opacity[idx].detach().clone())
    out._objects_dc = nn.Parameter(gaussians._objects_dc[idx].detach().clone())
    out.active_sh_degree = gaussians.active_sh_degree
    return out


def _new_feature_dir_buffer(gaussians) -> Dict[str, torch.Tensor]:
    """Accumulate proposed ADD feature directions / opacities; apply once at the end."""
    obj = gaussians._objects_dc
    n = int(obj.shape[0])
    d = int(obj.reshape(n, -1).shape[1])
    return {
        "sum_dir": torch.zeros((n, d), device=obj.device, dtype=obj.dtype),
        "count": torch.zeros((n,), device=obj.device, dtype=obj.dtype),
        "sum_opacity": torch.zeros_like(gaussians._opacity.data),
        "op_count": torch.zeros(
            (n, 1), device=gaussians._opacity.device, dtype=gaussians._opacity.dtype
        ),
    }


def _buffer_add_from_subset(buf: Dict[str, torch.Tensor], idx: torch.Tensor, gs_part) -> None:
    """Add this node's proposed ADD features as unit directions (and opacities)."""
    idx = idx.reshape(-1).long()
    if idx.numel() == 0:
        return
    flat = gs_part._objects_dc.detach().reshape(idx.numel(), -1)
    eps = 1e-12
    dirs = flat / flat.norm(dim=-1, keepdim=True).clamp_min(eps)
    ones = torch.ones((idx.numel(),), device=idx.device, dtype=buf["count"].dtype)
    buf["sum_dir"].index_add_(0, idx, dirs.to(dtype=buf["sum_dir"].dtype))
    buf["count"].index_add_(0, idx, ones)
    op = gs_part._opacity.detach()
    if op.ndim == 1:
        op = op.unsqueeze(-1)
    buf["sum_opacity"].index_add_(0, idx, op.to(dtype=buf["sum_opacity"].dtype))
    buf["op_count"].index_add_(
        0, idx, torch.ones((idx.numel(), 1), device=idx.device, dtype=buf["op_count"].dtype)
    )


def _apply_mean_dir_to_objects(
    objects_dc: torch.Tensor, buf: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Return objects_dc with averaged unit directions where the buffer has votes."""
    out = objects_dc.detach().clone()
    count = buf["count"]
    mask = count > 0
    if int(mask.sum().item()) == 0:
        return out
    eps = 1e-12
    flat = out.reshape(out.shape[0], -1)
    mean_dir = buf["sum_dir"][mask] / count[mask].unsqueeze(-1)
    mean_dir = mean_dir / mean_dir.norm(dim=-1, keepdim=True).clamp_min(eps)
    mag = flat[mask].norm(dim=-1, keepdim=True).clamp_min(eps)
    flat[mask] = mean_dir.to(dtype=flat.dtype) * mag
    return flat.reshape_as(out)


def _apply_feature_dir_buffer(gaussians, buf: Dict[str, torch.Tensor]) -> int:
    """One write: set feature to the mean proposed direction; average proposed opacity."""
    count = buf["count"]
    mask = count > 0
    n_upd = int(mask.sum().item())
    if n_upd == 0:
        return 0
    with torch.no_grad():
        gaussians._objects_dc.data.copy_(
            _apply_mean_dir_to_objects(gaussians._objects_dc.data, buf)
        )
        op_count = buf["op_count"].reshape(-1)
        op_mask = op_count > 0
        if int(op_mask.sum().item()) > 0:
            mean_op = buf["sum_opacity"][op_mask] / op_count[op_mask].reshape(-1, 1)
            gaussians._opacity.data[op_mask] = mean_op.to(dtype=gaussians._opacity.dtype)
    return n_upd


def _apply_codebook_feature_refine(
    gs_add,
    gates: torch.Tensor,
    gate_thr: float,
    codebook_feature: torch.Tensor,
    alpha_attract: float,
    cos_cap: float,
    cos_floor: float,
    repel_base: float,
    repel_scale: float,
) -> None:
    """Move ADD features toward (attract) or away from (repel) this node's codebook row.

    Attract (gate >= gate_thr):
        cos_new = cos_before + alpha_attract * (cos_cap - cos_before)
        rho = (cos_new - cos_before) / (1 - cos_before)

    Repel (gate < gate_thr):
        beta = repel_base + repel_scale * max(0, gate_thr - gate)
        cos_new = cos_before - beta * (cos_before - cos_floor)
        rho = (cos_before - cos_new) / (1 + cos_before)

    rho is clamped to [-0.5, 0.5].
    """
    with torch.no_grad():
        obj = gs_add._objects_dc.detach().clone()
        if obj.numel() == 0:
            return

        eps = 1e-12
        flat = obj.reshape(obj.shape[0], -1)
        norms = flat.norm(dim=-1, keepdim=True).clamp_min(eps)
        feat_dir = flat / norms

        c = codebook_feature.detach().reshape(-1).to(device=flat.device, dtype=flat.dtype)
        c = c / c.norm().clamp_min(eps)

        g = gates.detach().reshape(-1).to(device=flat.device)
        high = g >= float(gate_thr)
        low = ~high
        cos_before_all = feat_dir @ c
        rho_max = 0.5
        updated_dir = feat_dir.clone()

        if int(high.sum()) > 0:  # attract
            cb = cos_before_all[high].unsqueeze(-1)
            cos_new = (cb + alpha_attract * (cos_cap - cb)).clamp(max=cos_cap)
            rho = ((cos_new - cb) / (1.0 - cb).clamp_min(eps)).clamp(-rho_max, rho_max)
            mixed = (1.0 - rho) * feat_dir[high] + rho * c.unsqueeze(0)
            updated_dir[high] = mixed / mixed.norm(dim=-1, keepdim=True).clamp_min(eps)

        if int(low.sum()) > 0:  # repel
            cb = cos_before_all[low].unsqueeze(-1)
            g_low = g[low].unsqueeze(-1)
            beta = repel_base + repel_scale * (float(gate_thr) - g_low).clamp_min(0.0)
            cos_new = (cb - beta * (cb - cos_floor)).clamp(min=cos_floor)
            rho = ((cb - cos_new) / (1.0 + cb).clamp_min(eps)).clamp(-rho_max, rho_max)
            mixed = (1.0 - rho) * feat_dir[low] + rho * (-c).unsqueeze(0)
            updated_dir[low] = mixed / mixed.norm(dim=-1, keepdim=True).clamp_min(eps)

        gs_add._objects_dc = nn.Parameter((updated_dir * norms).reshape_as(obj))


def _run_clip_gate_and_refine(
    gs_in,
    gs_add,
    model_path: str,
    prompt: str,
    cfg: Dict[str, Any],
    device: str,
    codebook_feature: torch.Tensor,
) -> None:
    """Optimize the CLIP gates of ``gs_add`` against ``prompt``, push their features, copy the result back."""
    goc = _load_goc_clip_refine()
    goc.CLIP_INPUT_SIZE = int(cfg["clip_input_size"])

    _ensure_clip_refine_import_path()
    from render_synthetic import SyntheticPoseSampler

    gs_in_n, gs_add_n, norm_params = goc.normalize_gaussians(gs_in, gs_add, target_extent=0.5)
    renderer = _get_renderer(model_path=os.path.abspath(model_path), device=device)
    pose_sampler = SyntheticPoseSampler(
        width=int(cfg["render_width"]),
        height=int(cfg["render_height"]),
        fov_deg=50.0,
        device=device,
        no_reject=True,
        radius_fixed=(float(cfg["radius_min"]), float(cfg["radius_max"])),
    )
    clipm = _get_clip_wrapper(cfg, device)

    # train_vlsplat calls refinement inside a no_grad section; optimize_gates() needs gradients.
    with torch.enable_grad():
        gates, _ = goc.optimize_gates(
            renderer=renderer,
            pose_sampler=pose_sampler,
            clipm=clipm,
            gs_in=gs_in_n,
            gs_add=gs_add_n,
            prompt_pos=[prompt],
            prompt_neg=[],
            negative_weight=1.0,
            iters=int(cfg["clip_iters"]),
            views_per_iter=int(cfg["views_per_iter"]),
            gate_warm_start=float(cfg["gate_warm_start"]),
            lam_sparse=0.0,
            debug_save_every=0,
            save_every=0,
            gate_thr=float(cfg["gate_thr"]),
            norm_params=norm_params,
            device=device,
            pair_opposite_views=True,
            log_every=0 if bool(cfg["quiet_node_refine"]) else 10,
        )
        _apply_codebook_feature_refine(
            gs_add=gs_add_n,
            gates=gates,
            gate_thr=float(cfg["gate_thr"]),
            codebook_feature=codebook_feature,
            alpha_attract=float(cfg["alpha_attract"]),
            cos_cap=float(cfg["cos_cap"]),
            cos_floor=float(cfg["cos_floor"]),
            repel_base=float(cfg["repel_base"]),
            repel_scale=float(cfg["repel_scale"]),
        )

    gs_add_world = goc.unnormalize_gaussians(gs_add_n, norm_params, device=device)
    with torch.no_grad():
        gs_add._opacity.data.copy_(gs_add_world._opacity.data)
        gs_add._objects_dc.data.copy_(gs_add_world._objects_dc.data)


def _node_prototype_group(node: dict, gpp: torch.Tensor) -> int:
    """Codebook group whose row is the node's prototype for ADD attract / repel: the group with the most Gaussians."""
    gids = node_group_ids(node)
    if not gids:
        raise ValueError(f"node.group_id missing/non-numeric: {node}")
    return int(max(gids, key=lambda g: int((gpp == int(g)).sum().item())))


def _roi_tight_box(gaussians, idx: torch.Tensor, cfg: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Tight box of the ROI of ``idx``: anisotropic grouping, then the spatial crop of the graph construction."""
    from utils.general_utils import build_rotation

    from scene_graph_builder.build_scene_graph_nodes import (
        build_covariance_from_rotation_scale,
        group_gaussians,
        spatial_roi_crop_from_labels,
    )

    idx = idx.reshape(-1).long()
    if idx.numel() == 0:
        return None
    R = build_rotation(gaussians._rotation[idx]).detach().cpu().numpy().astype(np.float64)
    scale = gaussians.get_scaling[idx].detach().cpu().numpy().astype(np.float64)
    mu = gaussians.get_xyz[idx].detach().cpu().numpy().astype(np.float64)
    eps = float(cfg["roi_eps"])
    Sigma = build_covariance_from_rotation_scale(R, scale, eps=eps)
    try:
        _, _, labels, _ = group_gaussians(
            mu=mu, Sigma=Sigma, k=int(cfg["roi_k"]), eps=eps,
            overlap_tolerance=float(cfg["roi_overlap_tolerance"]),
        )
        _, meta = spatial_roi_crop_from_labels(
            mu,
            np.asarray(labels, dtype=np.int64),
            top_group_fraction=float(cfg["roi_top_group_fraction"]),
            margin_fraction=float(cfg["roi_margin_fraction"]),
            restrict_to_selected_groups=False,
        )
    except Exception:
        return None
    return {k: float(meta["bbox_tight"][k]) for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}


def _init_style_roi_seed_indices(
    gaussians,
    gpp: torch.Tensor,
    code_book: torch.Tensor,
    group_id: int,
    cosine_threshold: float,
    min_count: int,
    cross_group_union_min_count: int,
) -> torch.Tensor:
    """
    Same ROI seed as ``2_init_scene_graph.sh`` /
    ``build_scene_graph_nodes.py`` for a target node:

    this group UNION other groups with > ``cross_group_union_min_count`` points
    whose cosine to ``codebook[group_id]`` exceeds ``cosine_threshold``, then
    keep only seed points that pass that cosine filter. Empty if count <= ``min_count``.
    """
    device = gpp.device
    gpp = gpp.reshape(-1)
    gid = int(group_id)
    n_all = int(gpp.numel())
    if gid < 0 or gid >= int(code_book.shape[0]) or n_all == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    obj = gaussians._objects_dc.detach().reshape(n_all, -1)
    obj = obj / (obj.norm(dim=-1, keepdim=True) + 1e-12)
    c = code_book[gid].detach().reshape(-1)
    c = c / (c.norm() + 1e-12)
    high = ((obj * c.unsqueeze(0)).sum(dim=-1) > float(cosine_threshold)).reshape(-1)

    in_src = torch.zeros(n_all, dtype=torch.bool, device=device)
    other_high = high & (gpp != gid)
    if bool(other_high.any()):
        uniq, counts = torch.unique(gpp[other_high], return_counts=True)
        src_ok = uniq[counts > int(cross_group_union_min_count)]
        for sg in src_ok.tolist():
            in_src |= gpp == int(sg)

    seed = (gpp == gid) | in_src
    idx = torch.nonzero(seed & high, as_tuple=False).reshape(-1).long()
    if int(idx.numel()) <= int(min_count):
        return torch.zeros(0, dtype=torch.long, device=device)
    return idx


def _union_boxes(boxes: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        f"{a}{m}": float((min if m == "min" else max)(b[f"{a}{m}"] for b in boxes))
        for a in "xyz"
        for m in ("min", "max")
    }


def refresh_target_node_bbox_tight(sg_doc: dict, gaussians, code_book: torch.Tensor, gpp: torch.Tensor, cfg: Dict[str, Any]) -> int:
    """Re-fit ``bbox_tight`` of every target node with the ROI recipe of the graph construction.

    A merged node takes the union of the ROI boxes of all its groups. A node whose ROI is empty keeps its
    previous box. Returns the number of updated nodes."""
    updated = 0
    for node in sg_doc.get("nodes") or []:
        if not bool(node.get("target", False)):
            continue
        boxes = []
        for gid in node_group_ids(node):
            idx = _init_style_roi_seed_indices(
                gaussians,
                gpp,
                code_book,
                int(gid),
                float(cfg["codebook_sim_threshold"]),
                int(cfg["min_count"]),
                int(cfg["cross_group_union_min_count"]),
            )
            if idx.numel() == 0:
                continue
            box = _roi_tight_box(gaussians, idx, cfg)
            if box is not None:
                boxes.append(box)
        if boxes:
            node["bbox_tight"] = _union_boxes(boxes)
            updated += 1
    return updated


def _cap_target_nodes_per_class(target_nodes: List[dict], gpp: torch.Tensor, cfg: Dict[str, Any]) -> List[dict]:
    """Heuristic speed cap: refine only the ``max_refine_nodes_per_class`` (default 5) largest nodes of each
    class, ranked by Gaussian count. Smaller nodes stay in the graph but are skipped by intra/inter-refine.
    ``max_refine_nodes_per_class <= 0`` disables the cap."""
    cap = int(cfg.get("max_refine_nodes_per_class", 5))
    if cap <= 0:
        return target_nodes
    by_class: Dict[str, List[Tuple[int, dict]]] = {}
    for node in target_nodes:
        gids = node_group_ids(node)
        size = int(sum(int((gpp == int(g)).sum().item()) for g in gids)) if gids else 0
        by_class.setdefault(str(node.get("class_name", "")), []).append((size, node))
    keep_ids = set()
    for cls, items in by_class.items():
        items.sort(key=lambda t: (-t[0], int(t[1].get("id", 0))))
        keep_ids.update(id(n) for _, n in items[:cap])
        if len(items) > cap:
            skipped = ", ".join(f"node {n.get('id')} ({sz} Gaussians)" for sz, n in items[cap:])
            _sg_print(cfg, f"[scene_graph] {cls}: {len(items)} nodes > {cap}; refining the {cap} largest, skipping {skipped}")
    return [n for n in target_nodes if id(n) in keep_ids]


def _node_refine_pbar(nodes: List[dict], phase: str, iteration: int):
    """Nested bar under the 15k→30k training tqdm (position 0)."""
    return tqdm(
        nodes,
        total=len(nodes),
        desc=f"{phase} [{int(iteration)}]",
        unit="node",
        leave=False,
        position=1,
        dynamic_ncols=True,
        miniters=1,
        file=sys.stderr,
    )


def _excluded_group_ids_for_inter(sg_doc: dict) -> List[int]:
    """Groups that must not be inter-refine ADD: every group that belongs to a scene-graph node."""
    out: set = set()
    for node in sg_doc.get("nodes") or []:
        out.update(node_group_ids(node))
    return sorted(out)


def _refine_nodes(
    phase: str,
    target_nodes: List[dict],
    iteration: int,
    gaussians,
    code_book: torch.Tensor,
    gpp: torch.Tensor,
    model_path: str,
    cfg: Dict[str, Any],
    feat_buf: Dict[str, torch.Tensor],
    excluded_group_ids: List[int],
) -> int:
    """One refinement phase over every target node. Returns the number of nodes that were refined.

    A node is refined as a whole: its Gaussians are those of all its groups, each compared with its own group's
    codebook row, and ADD features move toward the row of the node's prototype group (the largest one).
    intra: IN = the node's Gaussians inside bbox_tight with codebook cosine >= tau_f; ADD = the node's rest.
    inter: IN as above; ADD = Gaussians of other (node-less) groups inside bbox_tight expanded by the margin."""
    thr = float(cfg["feat_sim_threshold"])
    device = gaussians._xyz.device
    expand_margin = float(cfg["bbox_margin_fraction"])
    n_ok = 0
    pbar = _node_refine_pbar(target_nodes, f"{phase}-refine", iteration)
    for node in pbar:
        gids = node_group_ids(node)
        gid = _node_prototype_group(node, gpp)
        name = (node.get("class_name") or "").strip() or f"node_{int(node.get('id', -1))}"
        pbar.set_postfix_str(f"{name} g={gids}", refresh=False)
        prompt = _node_prompt(node)
        if not prompt:
            continue
        idx_group = torch.nonzero(torch.isin(gpp, torch.as_tensor(gids, device=gpp.device)), as_tuple=False).reshape(-1).long()
        if phase == "intra" and idx_group.numel() == 0:
            continue
        bbox_tight = _bbox_tight(node)
        idx_in, idx_add = codebook_similarity_split(
            gaussians, code_book, gpp, _filter_indices_by_bbox(gaussians.get_xyz.detach(), idx_group, bbox_tight), thr
        )
        if phase == "inter":
            bbox_add = _bbox_expanded_from_tight(node, expand_margin)
            if bbox_tight is None or bbox_add is None:
                continue
            idx_add = indices_in_bbox_except_group_ids(
                gaussians.get_xyz.detach(), gpp, int(gid), bbox_add, exclude_group_ids=excluded_group_ids
            )
        if idx_in.numel() == 0 or idx_add.numel() == 0:
            continue
        gs_in = _subset_gaussian(gaussians, idx_in)
        gs_add = _subset_gaussian(gaussians, idx_add)
        _run_clip_gate_and_refine(gs_in, gs_add, model_path, prompt, cfg, device, codebook_feature=code_book[gid])
        _buffer_add_from_subset(feat_buf, idx_add, gs_add)
        n_ok += 1
        del gs_in, gs_add
    pbar.close()
    return n_ok


def run_scene_graph_refinement_cycle(
    iteration: int,
    gaussians,
    code_book: torch.Tensor,
    sg_doc: dict,
    model_path: str,
    cfg: Dict[str, Any],
) -> str:
    """One refinement round: refresh the target boxes, then intra-refine and inter-refine every target node.

    ``sg_doc`` is updated in place (target ``bbox_tight`` only; ``group_id``, ``text_prompt`` and class stay
    fixed). Features and opacities are written to ``gaussians`` once at the end, so that all nodes see the same
    raw features during the round. Returns a one-line summary."""
    target_nodes = [n for n in sg_doc.get("nodes") or [] if bool(n.get("target", False))]
    if not target_nodes:
        return "skip: no target nodes"

    _, gpp = compute_group_per_point(gaussians, code_book)
    target_nodes = _cap_target_nodes_per_class(target_nodes, gpp, cfg)
    feat_buf = _new_feature_dir_buffer(gaussians)
    n_bbox = refresh_target_node_bbox_tight(sg_doc, gaussians, code_book, gpp, cfg)
    excluded = _excluded_group_ids_for_inter(sg_doc)

    n_intra = _refine_nodes("intra", target_nodes, iteration, gaussians, code_book, gpp, model_path, cfg, feat_buf, excluded)
    n_inter = _refine_nodes("inter", target_nodes, iteration, gaussians, code_book, gpp, model_path, cfg, feat_buf, excluded)
    n_feat = _apply_feature_dir_buffer(gaussians, feat_buf)
    return (
        f"iter {iteration}: target_nodes={len(target_nodes)} bbox_updated={n_bbox} "
        f"intra_ok={n_intra} inter_ok={n_inter} feat_updated={n_feat}"
    )
