#!/usr/bin/env python3
"""Stage 1 of the scene graph: codebook groups -> nodes (class from SAM3 votes, anisotropic ROI, boxes, edges).

Also holds the geometry helpers shared with the later stages and with refinement: anisotropic Gaussian
grouping, spatial ROI crop, upright oriented boxes, and the checkpoint / codebook loaders."""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import pickle
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRotation

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.local_runtime_utils import ensure_abs, load_codebook as load_codebook_torch, normalize_label

# Anisotropic grouping of 3D Gaussians (directional projected overlap).


OverlapMode = Literal["soft", "hard"]

# Overlap tolerance u ∈ [0, 1] maps linearly to the gap scale α ∈ [MIN, MAX].
OVERLAP_ALPHA_MIN = 0.32
OVERLAP_ALPHA_MAX = 2.05
# Hard affinity ∈ {0, 1}: an edge is kept iff its affinity exceeds this threshold.
AFFINITY_EDGE_THRESHOLD = 0.0
# κ of the ellipsoid test used to prune candidate pairs when ``k == 0``.
SPATIAL_CANDIDATE_KAPPA = 9.0


Edge = Tuple[int, int, float]


def build_covariance_from_rotation_scale(
    rotations: np.ndarray, scales: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Build Gaussian covariance matrices from rotation and scale.

    Args:
        rotations: Array of shape (N, 3, 3) containing rotation matrices.
        scales: Array of shape (N, 3) containing principal-axis std scales.
        eps: Minimum variance floor for numerical stability.

    Returns:
        Sigma: Array of shape (N, 3, 3), where
            Sigma_i = R_i @ diag(s_i^2) @ R_i^T
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"rotations must have shape (N,3,3), got {rotations.shape}")
    if scales.ndim != 2 or scales.shape[1] != 3 or scales.shape[0] != rotations.shape[0]:
        raise ValueError(f"scales must have shape (N,3), got {scales.shape}")

    s2 = np.maximum(scales * scales, eps)
    diag_s2 = np.zeros((scales.shape[0], 3, 3), dtype=np.float64)
    diag_s2[:, 0, 0] = s2[:, 0]
    diag_s2[:, 1, 1] = s2[:, 1]
    diag_s2[:, 2, 2] = s2[:, 2]

    sigma = rotations @ diag_s2 @ np.transpose(rotations, (0, 2, 1))
    sigma = 0.5 * (sigma + np.transpose(sigma, (0, 2, 1)))
    sigma += np.eye(3)[None, :, :] * eps
    return sigma


def spectral_extent_kappa(Sigma: np.ndarray, kappa: float, eps: float) -> float:
    """Outer sphere radius sqrt(kappa * lambda_max(Sigma)) containing the kappa-ellipsoid."""
    Sigma = np.asarray(Sigma, dtype=np.float64)
    eig = np.linalg.eigvalsh(Sigma)
    lam = max(float(np.max(eig)), float(eps))
    return float(np.sqrt(kappa * lam))


def iter_spatial_ellipsoid_candidate_pairs(
    mu: np.ndarray,
    Sigma: np.ndarray,
    kappa: float,
    eps: float,
) -> List[Tuple[int, int]]:
    """All pairs (i, j) with i < j whose kappa-ellipsoids may intersect."""
    mu = np.asarray(mu, dtype=np.float64)
    n = mu.shape[0]
    if n < 2:
        return []
    extents = np.array(
        [spectral_extent_kappa(Sigma[i], kappa, eps) for i in range(n)],
        dtype=np.float64,
    )
    max_e = float(np.max(extents))
    tree = cKDTree(mu)
    out: List[Tuple[int, int]] = []
    pad = float(max(eps, 1e-12))
    for i in range(n):
        rad = float(extents[i] + max_e + pad)
        cand = tree.query_ball_point(mu[i], r=rad)
        for j in cand:
            jj = int(j)
            if jj <= i:
                continue
            d = float(np.linalg.norm(mu[i] - mu[jj]))
            if d <= extents[i] + extents[jj] + pad:
                out.append((i, jj))
    return out


def iter_knn_candidate_pairs(mu: np.ndarray, k: int) -> List[Tuple[int, int]]:
    """Undirected candidate edges from Euclidean kNN on centers (each edge once, i < j)."""
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 2 or mu.shape[1] != 3:
        raise ValueError(f"mu must have shape (N,3), got {mu.shape}")
    n = mu.shape[0]
    if n < 2 or k <= 0:
        return []
    k_eff = min(int(k), n - 1)
    tree = cKDTree(mu)
    _, idx = tree.query(mu, k=k_eff + 1)
    neighbors = idx[:, 1:].astype(np.int64)
    seen: Set[Tuple[int, int]] = set()
    out: List[Tuple[int, int]] = []
    for i in range(n):
        for j in neighbors[i]:
            jj = int(j)
            if jj == i:
                continue
            a, b = (i, jj) if i < jj else (jj, i)
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, b))
    return out


def select_candidate_pairs(
    mu: np.ndarray,
    Sigma: np.ndarray,
    k: int,
    iou_kappa: float,
    eps: float,
) -> List[Tuple[int, int]]:
    """If ``k > 0``, kNN(center) pairs only; else all spatially plausible κ-ellipsoid pairs."""
    if k > 0:
        return iter_knn_candidate_pairs(mu, k)
    return iter_spatial_ellipsoid_candidate_pairs(mu, Sigma, float(iou_kappa), eps)


def alpha_from_overlap_tolerance(u: float) -> float:
    """Map user overlap tolerance ``u ∈ [0, 1]`` to gap scaling ``α`` (clamped).

    Linear ``α = α_min + u·(α_max - α_min)`` on ``gap = dist − α(σ_i + σ_j)``, with
    ``α_min = 0.32`` and ``α_max = 2.05``, so a small ``u`` prunes dense kNN edges strictly.
    """
    uu = float(np.clip(float(u), 0.0, 1.0))
    return float(
        OVERLAP_ALPHA_MIN + uu * (OVERLAP_ALPHA_MAX - OVERLAP_ALPHA_MIN)
    )


def compute_projected_overlap_affinity(
    mu_i: np.ndarray,
    Sigma_i: np.ndarray,
    mu_j: np.ndarray,
    Sigma_j: np.ndarray,
    alpha: float = 1.0,
    sigma: float = 0.05,
    eps: float = 1e-8,
    mode: OverlapMode = "soft",
    return_debug: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """Directional geometry affinity between two anisotropic Gaussians.

    Let ``d = μ_j - μ_i``, ``dist = ||d||_2``, and ``d_hat = d / (dist + ε)`` (avoids division by zero).

    Along ``d_hat``, the marginal variance is ``v_i = d_hatᵀ Σ_i d_hat`` (length²). For a length
    comparable to ``dist``, use the **directional standard deviation** (effective radius along the
    center–center line)::

        σ_i = sqrt(max(0, v_i)),   σ_j = sqrt(max(0, v_j)).

    **Gap** (overlap tolerance scaled by ``α``)::

        gap = dist - α * (σ_i + σ_j).

    Using ``v_i`` instead of ``σ_i`` here would mix length with length² and make the gate far too
    strict for typical splat covariances. Debug fields ``r_i``, ``r_j`` store ``σ_i``, ``σ_j``.

    **Soft** mode (default)::

        affinity = exp( - max(0, gap)² / (σ² + ε) ),

    so ``gap ≤ 0 ⇒ affinity = 1``, and positive gaps decay smoothly.

    **Hard** mode::

        affinity = 1 if gap ≤ 0 else 0.

    Coincident centers (``dist < ε``): returns ``affinity = 1`` (fully overlapping).

    Returns:
        If ``return_debug`` is False: scalar in ``[0, 1]``.
        If True: ``(affinity, {"dist", "r_i", "r_j", "gap"})``.
    """
    mu_i = np.asarray(mu_i, dtype=np.float64).reshape(3)
    mu_j = np.asarray(mu_j, dtype=np.float64).reshape(3)
    Sigma_i = np.asarray(Sigma_i, dtype=np.float64).reshape(3, 3)
    Sigma_j = np.asarray(Sigma_j, dtype=np.float64).reshape(3, 3)
    if Sigma_i.shape != (3, 3) or Sigma_j.shape != (3, 3):
        raise ValueError("Sigma_i, Sigma_j must be (3,3)")

    Sigma_i = 0.5 * (Sigma_i + Sigma_i.T)
    Sigma_j = 0.5 * (Sigma_j + Sigma_j.T)

    d = mu_j - mu_i
    dist = float(np.linalg.norm(d))
    if dist < float(eps):
        dbg = {"dist": 0.0, "r_i": 0.0, "r_j": 0.0, "gap": 0.0}
        return (1.0, dbg) if return_debug else 1.0

    d_hat = d / (dist + float(eps))
    vi = float(max(0.0, d_hat @ Sigma_i @ d_hat))
    vj = float(max(0.0, d_hat @ Sigma_j @ d_hat))
    ri = float(np.sqrt(vi + float(eps) ** 2))
    rj = float(np.sqrt(vj + float(eps) ** 2))
    gap = float(dist - float(alpha) * (ri + rj))

    if mode == "hard":
        score = 1.0 if gap <= 0.0 else 0.0
    elif mode == "soft":
        g = max(0.0, gap)
        denom = float(sigma) ** 2 + float(eps)
        score = float(np.exp(-(g * g) / denom))
    else:
        raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")

    score = float(np.clip(score, 0.0, 1.0))
    dbg = {"dist": dist, "r_i": ri, "r_j": rj, "gap": gap}
    return (score, dbg) if return_debug else score


def compute_pair_affinity(
    mu_i: np.ndarray,
    Sigma_i: np.ndarray,
    mu_j: np.ndarray,
    Sigma_j: np.ndarray,
    eps: float,
    overlap_tolerance: float,
) -> float:
    """Hard affinity ∈ {0,1} for one candidate pair."""
    alpha = alpha_from_overlap_tolerance(overlap_tolerance)
    return float(
        compute_projected_overlap_affinity(
            mu_i,
            Sigma_i,
            mu_j,
            Sigma_j,
            alpha=float(alpha),
            sigma=0.05,
            eps=float(eps),
            mode="hard",
            return_debug=False,
        )
    )


def build_gaussian_graph(
    mu: np.ndarray,
    Sigma: np.ndarray,
    k: int = 0,
    eps: float = 1e-6,
    overlap_tolerance: float = 0.5,
) -> Tuple[List[Edge], np.ndarray]:
    """Build an undirected graph from hard affinity edges (weight = affinity).

    ``k > 0``: kNN(center) candidates; ``k == 0``: κ-ellipsoid spatial candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    if mu.ndim != 2 or mu.shape[1] != 3:
        raise ValueError(f"mu must have shape (N,3), got {mu.shape}")
    if Sigma.ndim != 3 or Sigma.shape[1:] != (3, 3) or Sigma.shape[0] != mu.shape[0]:
        raise ValueError(f"Sigma must have shape (N,3,3), got {Sigma.shape}")

    n = mu.shape[0]
    pairs = select_candidate_pairs(mu, Sigma, int(k), float(SPATIAL_CANDIDATE_KAPPA), eps)

    edge_w: Dict[Tuple[int, int], float] = {}
    edge_threshold = float(AFFINITY_EDGE_THRESHOLD)
    for a, b in pairs:
        affinity = compute_pair_affinity(
            mu[a],
            Sigma[a],
            mu[b],
            Sigma[b],
            eps=eps,
            overlap_tolerance=float(overlap_tolerance),
        )
        keep = affinity > edge_threshold
        w = float(affinity)

        if keep:
            prev = edge_w.get((a, b))
            if prev is None or w > prev:
                edge_w[(a, b)] = float(w)

    edges: List[Edge] = [(i, j, float(w)) for (i, j), w in edge_w.items()]
    node_scores = np.zeros(n, dtype=np.float64)
    for i, j, w in edges:
        node_scores[i] += w
        node_scores[j] += w

    return edges, node_scores


def connected_components_from_edges(
    num_nodes: int, edges: Sequence[Edge], keep_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute component labels from undirected edges.

    Nodes excluded by keep_mask are assigned label -1.
    """
    labels = np.full(num_nodes, -1, dtype=np.int64)
    if num_nodes == 0:
        return labels

    if keep_mask is None:
        keep_mask = np.ones(num_nodes, dtype=bool)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    kept = np.where(keep_mask)[0]
    if kept.size == 0:
        return labels

    old_to_new = np.full(num_nodes, -1, dtype=np.int64)
    old_to_new[kept] = np.arange(kept.size, dtype=np.int64)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for i, j, w in edges:
        if keep_mask[i] and keep_mask[j]:
            ii, jj = int(old_to_new[i]), int(old_to_new[j])
            rows.extend([ii, jj])
            cols.extend([jj, ii])
            data.extend([w, w])

    if len(data) == 0:
        labels[kept] = np.arange(kept.size, dtype=np.int64)
        return labels

    graph: csr_matrix = coo_matrix((data, (rows, cols)), shape=(kept.size, kept.size)).tocsr()
    _, local_labels = connected_components(graph, directed=False, return_labels=True)
    labels[kept] = local_labels.astype(np.int64)
    return labels


def group_gaussians(
    mu: np.ndarray,
    Sigma: np.ndarray,
    k: int = 0,
    eps: float = 1e-6,
    overlap_tolerance: float = 0.5,
) -> Tuple[List[Edge], np.ndarray, np.ndarray, np.ndarray]:
    """Build affinity graph and connected-component labels (no pruning)."""
    edges, node_scores = build_gaussian_graph(
        mu,
        Sigma,
        k=k,
        eps=eps,
        overlap_tolerance=overlap_tolerance,
    )
    n = mu.shape[0]
    keep_mask = np.ones(n, dtype=bool)
    labels = connected_components_from_edges(n, edges, keep_mask=keep_mask)
    return edges, node_scores, labels, keep_mask


def load_gaussians_from_ply(
    ply_path: Path, scales_are_log: bool = True, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Load Gaussian centers/covariances from a PLY with x,y,z, scale_*, rot_* fields.

    Expected rotation field format is quaternion in scalar-first convention:
      (rot_0, rot_1, rot_2, rot_3) == (w, x, y, z).
    """
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("plyfile is required to read .ply. Install via `pip install plyfile`.") from exc

    ply = PlyData.read(str(ply_path))
    v = ply.elements[0]

    mu = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float64)

    scale_keys = sorted([p.name for p in v.properties if p.name.startswith("scale_")], key=lambda s: int(s.split("_")[-1]))
    if len(scale_keys) != 3:
        raise ValueError(f"Expected 3 scale_* properties, found {len(scale_keys)}")
    scales = np.stack([np.asarray(v[k]) for k in scale_keys], axis=1).astype(np.float64)
    if scales_are_log:
        scales = np.exp(scales)
    scales = np.maximum(scales, eps)

    rot_keys = sorted([p.name for p in v.properties if p.name.startswith("rot_")], key=lambda s: int(s.split("_")[-1]))
    if len(rot_keys) != 4:
        raise ValueError(f"Expected 4 rot_* properties (quaternion), found {len(rot_keys)}")
    q_wxyz = np.stack([np.asarray(v[k]) for k in rot_keys], axis=1).astype(np.float64)
    q_norm = np.linalg.norm(q_wxyz, axis=1, keepdims=True)
    q_wxyz = q_wxyz / np.maximum(q_norm, eps)
    q_xyzw = np.stack([q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]], axis=1)
    rotations = SciRotation.from_quat(q_xyzw).as_matrix()

    Sigma = build_covariance_from_rotation_scale(rotations, scales, eps=eps)
    return mu, Sigma


# Spatial ROI from KNN cluster labels: mass-based group selection → gate AABB → crop mask.
#
# Used by the scene-graph builders and training refinement
# (``spatial_roi_crop_from_labels``).
#
# Steps:
# 1. Sort connected components (label ≥ 0) by Gaussian count descending.
# 2. Take largest groups until cumulative count ≥ ``ceil(top_group_fraction × N_valid)``.
# 3. Tight AABB on those groups → expand by ``margin_fraction`` per side → ``keep`` mask
#    (optionally intersect with selected labels only).
#
# ``bbox_tight`` / ``bbox_expanded`` in the returned ``meta`` are the hull of **kept** centers
# and that hull expanded again by ``margin_fraction``.

def group_counts_descending(labels: np.ndarray) -> List[Tuple[int, int]]:
    """[(label, count), ...] sorted by count descending, then label ascending."""
    valid = labels >= 0
    if not np.any(valid):
        return []
    uniq, cnt = np.unique(labels[valid], return_counts=True)
    pairs = list(zip(uniq.astype(np.int64).tolist(), cnt.astype(np.int64).tolist()))
    pairs.sort(key=lambda t: (-t[1], t[0]))
    return pairs


def select_top_group_labels_mass(
    pairs: Sequence[Tuple[int, int]], n_valid: int, mass_fraction: float
) -> Tuple[set, int, int]:
    """Largest-first prefix until cumulative count >= ceil(mass_fraction * n_valid); at least one group."""
    g = len(pairs)
    if g == 0 or n_valid <= 0:
        return set(), 0, 0
    target = max(1, int(math.ceil(float(mass_fraction) * float(n_valid))))
    cum = 0
    selected: set = set()
    n_keep = 0
    for lab, c in pairs:
        selected.add(int(lab))
        cum += int(c)
        n_keep += 1
        if cum >= target:
            break
    return selected, g, n_keep


def axis_aligned_bbox(
    xyz: np.ndarray, mask: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    """Return xmin, xmax, ymin, ymax, zmin, zmax."""
    if not np.any(mask):
        raise ValueError("empty mask for bbox")
    p = xyz[mask]
    xmin, ymin, zmin = p.min(axis=0)
    xmax, ymax, zmax = p.max(axis=0)
    return float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax)


def expand_bbox_margin(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    margin_fraction: float,
) -> Tuple[float, float, float, float, float, float]:
    """Add margin_fraction * span on each side per axis."""
    m = float(margin_fraction)
    sx = xmax - xmin
    sy = ymax - ymin
    sz = zmax - zmin
    xmin -= m * sx
    xmax += m * sx
    ymin -= m * sy
    ymax += m * sy
    zmin -= m * sz
    zmax += m * sz
    return xmin, xmax, ymin, ymax, zmin, zmax


def mask_inside_bbox(
    xyz: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> np.ndarray:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return (
        (x >= xmin)
        & (x <= xmax)
        & (y >= ymin)
        & (y <= ymax)
        & (z >= zmin)
        & (z <= zmax)
    )


def spatial_roi_crop_from_labels(
    xyz: np.ndarray,
    labels: np.ndarray,
    top_group_fraction: float = 0.8,
    margin_fraction: float = 0.1,
    restrict_to_selected_groups: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Mass-based group selection → tight AABB on selected labels → margin → spatial ``keep`` mask.

    Returns ``(keep, meta)``. ``bbox_tight`` / ``bbox_expanded`` in ``meta`` describe the **ROI
    output** (Gaussians with ``keep``): tight AABB of kept centers, then the same per-side
    ``margin_fraction`` expansion of **that** tight box (not the pre-crop gate box).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = xyz.shape[0]
    if labels.shape[0] != n:
        raise ValueError(f"labels length {labels.shape[0]} != xyz rows {n}")

    if not (0.0 < float(top_group_fraction) <= 1.0):
        raise ValueError("top_group_fraction must be in (0, 1]")
    if float(margin_fraction) < 0.0:
        raise ValueError("margin_fraction must be >= 0")

    n_valid = int(np.sum(labels >= 0))
    pairs = group_counts_descending(labels)
    if not pairs:
        raise ValueError("No labels >= 0; cannot fit ROI bbox")

    selected, g_all, n_keep = select_top_group_labels_mass(
        pairs, n_valid, top_group_fraction
    )
    target_mass = max(1, int(math.ceil(float(top_group_fraction) * float(n_valid))))

    mask_fit = np.isin(labels, list(selected)) & (labels >= 0)
    if not np.any(mask_fit):
        raise RuntimeError("no Gaussians in selected groups")

    xmin, xmax, ymin, ymax, zmin, zmax = axis_aligned_bbox(xyz, mask_fit)
    xmin2, xmax2, ymin2, ymax2, zmin2, zmax2 = expand_bbox_margin(
        xmin, xmax, ymin, ymax, zmin, zmax, margin_fraction
    )

    in_box = mask_inside_bbox(xyz, xmin2, xmax2, ymin2, ymax2, zmin2, zmax2)
    sel = np.isin(labels, list(selected)) & (labels >= 0)
    if restrict_to_selected_groups:
        keep = in_box & sel
    else:
        keep = in_box
    n_out = int(np.sum(keep))
    n_sel = int(np.sum(sel))
    n_in_box = int(np.sum(in_box))

    if not np.any(keep):
        raise RuntimeError("ROI spatial crop produced an empty keep mask")

    rxmin, rxmax, rymin, rymax, rzmin, rzmax = axis_aligned_bbox(xyz, keep)
    exmin, exmax, eymin, eymax, ezmin, ezmax = expand_bbox_margin(
        rxmin, rxmax, rymin, rymax, rzmin, rzmax, margin_fraction
    )

    meta = {
        "n_vertices_in": n,
        "n_vertices_out": n_out,
        "num_groups_total": g_all,
        "num_groups_used_for_bbox": n_keep,
        "selection": "mass",
        "n_vertices_labels_ge_0": n_valid,
        "target_gaussian_count_mass": target_mass,
        "top_group_fraction": float(top_group_fraction),
        "margin_fraction": float(margin_fraction),
        "restrict_to_selected_groups": bool(restrict_to_selected_groups),
        "spatial_roi_crop": not bool(restrict_to_selected_groups),
        "n_vertices_in_top_groups": n_sel,
        "n_vertices_inside_expanded_bbox": n_in_box,
        "group_counts_desc": [{"label": int(a), "count": int(b)} for a, b in pairs],
        "selected_labels": sorted(selected),
        "bbox_tight": {
            "xmin": rxmin,
            "xmax": rxmax,
            "ymin": rymin,
            "ymax": rymax,
            "zmin": rzmin,
            "zmax": rzmax,
        },
        "bbox_expanded": {
            "xmin": exmin,
            "xmax": exmax,
            "ymin": eymin,
            "ymax": eymax,
            "zmin": ezmin,
            "zmax": ezmax,
        },
    }
    return keep, meta


# ---- PLY object features + classifier weights -> argmax codebook class ids ----
def read_object_features_from_ply(ply_path: str, num_objects: int = 16) -> np.ndarray:
    plydata = PlyData.read(ply_path)
    n = len(plydata.elements[0].data)
    obj_dc = np.zeros((n, num_objects), dtype=np.float32)
    for i in range(num_objects):
        obj_dc[:, i] = np.asarray(plydata.elements[0].data[f"obj_dc_{i}"])
    return obj_dc


def load_codebook(classifier_path: str, num_objects: int = 16, device: str = "cpu"):
    try:
        ckpt = torch.load(classifier_path, map_location=device, weights_only=True)
    except (TypeError, pickle.UnpicklingError, RuntimeError):
        ckpt = torch.load(classifier_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        w = ckpt.get("weight", list(ckpt.values())[0])
        w = w.squeeze()
        if w.dim() == 4:
            w = w.squeeze()
        return w
    return ckpt.squeeze()


def compute_class_ids_np(features: np.ndarray, codebook, normalize: bool = True) -> np.ndarray:
    cb = np.asarray(codebook, dtype=np.float64)
    if normalize:
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
        features = features / norms
    logits = features @ cb.T
    return np.argmax(logits, axis=1)


SEMANTIC_TEXT_LABELS = [
    "wall",
    "floor",
    "cabinet",
    "ceiling",
    "bed",
    "chair",
    "sofa",
    "door",
    "window",
    "picture",
    "monitor",
    "desk",
    "curtain",
]
SEMANTIC_TEXT_LABEL_SET = set(SEMANTIC_TEXT_LABELS)
# Image-edge margin applies only to these "thing" classes.
THING_SEMANTIC_LABELS = (
    "cabinet",
    "bed",
    "chair",
    "desk",
    "sofa",
    "monitor",
)
# Stuff classes vote on the full rendered region (no margin crop).
STUFF_SEMANTIC_LABELS = (
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "picture",
    "curtain",
)
THING_SEMANTIC_LABEL_SET = set(THING_SEMANTIC_LABELS)
STUFF_SEMANTIC_LABEL_SET = set(STUFF_SEMANTIC_LABELS)
# Classes that get their own target node and text prompt.
TARGET_TEXT_LABELS = {"cabinet", "bed", "chair", "sofa", "picture", "desk"}
# Remaining SAM3 classes become non-target nodes.
NON_TARGET_SEMANTIC_LABELS = SEMANTIC_TEXT_LABEL_SET - TARGET_TEXT_LABELS


def _is_json_label_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        for fn in os.listdir(path):
            if fn.endswith(".json"):
                return True
    except Exception:
        return False
    return False


def _resolve_label_json_root(source_path: str, object_path: str, label_json_root: str = "") -> str:
    source_abs = ensure_abs(source_path)
    obj_abs = ensure_abs(object_path) if os.path.isabs(object_path) else os.path.join(source_abs, object_path)
    candidates: List[str] = []

    def _push(path: str) -> None:
        p = ensure_abs(path)
        if p not in candidates:
            candidates.append(p)

    explicit = str(label_json_root).strip()
    if explicit:
        explicit_abs = ensure_abs(explicit)
        if os.path.isfile(explicit_abs) and explicit_abs.endswith(".json"):
            _push(os.path.dirname(explicit_abs))
        else:
            _push(explicit_abs)

    _push(obj_abs)
    _push(os.path.join(obj_abs, "vocab"))

    base = os.path.basename(os.path.normpath(obj_abs))
    parent = os.path.dirname(obj_abs)
    for suffix in ("_instance_masks", "_overlay_masks"):
        if base.endswith(suffix):
            prefix = base[: -len(suffix)]
            _push(os.path.join(parent, f"{prefix}_confidence_json"))
    if base.endswith("_confidence_json"):
        _push(obj_abs)

    # Also check common sibling directories under scene root.
    scene_root = source_abs
    try:
        for entry in os.listdir(scene_root):
            full = os.path.join(scene_root, entry)
            if not os.path.isdir(full):
                continue
            low = entry.lower()
            if (
                low.endswith("_confidence_json")
                or low.endswith("_semantic_json")
                or low.endswith("_label_json")
                or low.endswith("object_mask_json")
                or low == "object_mask_json"
                or low == "merge_mask_json"
            ):
                _push(full)
            if "vocab" in low:
                _push(full)
    except Exception:
        pass

    for cand in candidates:
        if _is_json_label_dir(cand):
            return cand
    details = []
    for cand in candidates:
        if not os.path.exists(cand):
            details.append(f"{cand} (missing)")
        elif not os.path.isdir(cand):
            details.append(f"{cand} (not a directory)")
        else:
            try:
                n_json = sum(1 for fn in os.listdir(cand) if fn.endswith(".json"))
            except Exception as exc:
                details.append(f"{cand} (listdir failed: {exc})")
                continue
            details.append(f"{cand} (dir, {n_json} json files)")
    print(
        "[warn] Could not find label JSON directory. Proceeding without semantic JSON labels. "
        "Each node then gets class_name=none, so target node n=0 and views/description are skipped. "
        "Need a folder of <image_stem>.json files (ScanNet++: object_mask_json, ScanNetV2: object_mask_json or merge_mask_json). Checked: "
        + "; ".join(details),
        file=sys.stderr,
    )
    return ""


def _read_label_json(label_json_root: str, stem: str) -> Dict[int, str]:
    path = os.path.join(label_json_root, f"{stem}.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    out: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            sid = int(k)
        except Exception:
            continue
        if sid <= 0:
            continue

        label_text: Optional[str] = None
        if isinstance(v, dict):
            for lk in ("label", "class_name", "name", "category", "text"):
                if v.get(lk) is not None:
                    label_text = str(v[lk])
                    break
        else:
            label_text = str(v)

        if label_text is None:
            continue
        normalized = _canonicalize_semantic_label(label_text)
        if normalized:
            out[int(sid)] = normalized
    return out


def _discover_labels(label_json_root: str) -> List[str]:
    if not os.path.isdir(label_json_root):
        return list(SEMANTIC_TEXT_LABELS)
    return list(SEMANTIC_TEXT_LABELS)


def _canonicalize_semantic_label(label_text: str) -> str:
    normalized = normalize_label(label_text)
    if not normalized:
        return ""
    if "sam2" in normalized:
        return "none"
    if normalized in {"none", "background", "unknown", "unlabeled"}:
        return "none"
    if normalized in SEMANTIC_TEXT_LABEL_SET:
        return normalized
    compact = normalized.replace("_", "")
    for allowed in SEMANTIC_TEXT_LABELS:
        if compact == allowed.replace("_", ""):
            return allowed
    return ""


def _build_scene(
    model_root: Path,
    source_path: str,
    object_path: str,
    render_point_cloud_path: Path,
    render_classifier_path: Path,
    images: str,
    resolution: int,
    num_objects: int,
) -> Tuple[GaussianModel, torch.Tensor, Sequence[Any], Any]:
    src_abs = ensure_abs(source_path)

    def _has_colmap_root(p: str) -> bool:
        return os.path.isdir(os.path.join(p, "sparse", "0")) or os.path.isdir(os.path.join(p, "colmap"))

    def _has_blender_root(p: str) -> bool:
        return os.path.isfile(os.path.join(p, "transforms_train.json"))

    # Normalize common user inputs to scene root:
    # - <scene>/images or <scene>/resized_undistorted_images
    # - <scene>/sparse or <scene>/colmap
    # - <scene>/sparse/0
    norm = os.path.normpath(src_abs)
    base = os.path.basename(norm)
    parent = os.path.dirname(norm)
    grand_parent = os.path.dirname(parent)

    candidates = [src_abs]
    if base == str(images):
        candidates.append(parent)
    if base == "0" and os.path.basename(parent) == "sparse":
        candidates.append(grand_parent)
    if base == "sparse":
        candidates.append(parent)
    if base == "colmap":
        candidates.append(parent)

    chosen = None
    for cand in candidates:
        if os.path.isdir(cand) and (_has_colmap_root(cand) or _has_blender_root(cand)):
            chosen = cand
            break
    if chosen is None:
        checked = ", ".join(candidates)
        raise FileNotFoundError(
            "Could not resolve scene root from source_path. Expected COLMAP root with 'sparse/0' or 'colmap' "
            "or Blender root with 'transforms_train.json'. Checked: " + checked
        )
    src_abs = chosen

    # Keep the same camera split as the training run.
    split_settings = {"eval": True, "train_split": True}
    cfg_path = Path(model_root) / "cfg_args"
    if cfg_path.is_file():
        cfg_expr = ast.parse(cfg_path.read_text(), mode="eval").body
        if isinstance(cfg_expr, ast.Call):
            for keyword in cfg_expr.keywords:
                if keyword.arg in split_settings:
                    split_settings[keyword.arg] = bool(ast.literal_eval(keyword.value))
    print(f"[scene_graph] camera split: {split_settings}", flush=True)

    dataset_args = types.SimpleNamespace(
        model_path=str(model_root),
        source_path=src_abs,
        images=str(images),
        eval=split_settings["eval"],
        object_path=str(object_path),
        n_views=100,
        random_init=False,
        train_split=split_settings["train_split"],
        sh_degree=3,
        num_classes=int(num_objects),
        white_background=False,
        resolution=int(resolution),
        data_device="cuda",
    )
    pipeline = types.SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    gaussians = GaussianModel(3)
    scene = Scene(dataset_args, gaussians, load_iteration=None, shuffle=False)
    gaussians.load_ply(str(render_point_cloud_path))
    cameras = scene.getTrainCameras()
    codebook = load_codebook_torch(str(render_classifier_path), device="cuda")
    return gaussians, codebook, cameras, pipeline


def _label_pixel_buckets(
    codes: torch.Tensor,
    render_cnt: int,
    code_to_label: Dict[int, str],
) -> Dict[str, int]:
    bucket: Dict[str, int] = {lab: 0 for lab in code_to_label.values()}
    if int(codes.numel()) > 0:
        uniq, ucnt = torch.unique(codes, return_counts=True)
        for c_t, n_t in zip(uniq, ucnt):
            lab = code_to_label.get(int(c_t.item()), "none")
            bucket[lab] = int(n_t.item())
    labeled = int(sum(v for k, v in bucket.items() if k != "none"))
    bucket["none"] = int(max(0, int(render_cnt) - labeled))
    return bucket


def _compute_semantic_votes(
    cameras: Sequence[Any],
    gaussians: GaussianModel,
    codebook_t: torch.Tensor,
    pipeline: Any,
    label_json_root: str,
    valid_gid_set: Set[int],
    target_labels: Sequence[str],
    min_sam_pixels: int,
    valid_mask_margin_pct: float = 0.0,
    min_render_pixels: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, int]]]:
    """
    Accumulate votes between rendered codebook groups and SAM masks.

      - A view is a *valid vote* for group g iff rendered pixels of g
        (inside margin when margin_pct>0) >= min_render_pixels.
      - Label pixels are counted in that same region.
      - That view casts exactly one vote in {each SAM3 class, sam2, none}.
      - Final assigned_label = argmax over view-votes.
      Call again with valid_mask_margin_pct=0 to rematch none nodes on the full image.
    """
    votes = defaultdict(int)  # (gid, label) -> pixels (valid views only)
    view_vote_counts = defaultdict(int)  # (gid, label) -> #views that chose this label
    group_render_pixels = defaultdict(int)  # all observed render px
    group_valid_render_pixels = defaultdict(int)  # render px only from valid-vote views
    group_observed_views = defaultdict(set)
    group_valid_vote_views = defaultdict(set)
    per_view_label_pixels: Dict[int, Dict[int, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    per_view_render_pixels: Dict[int, Dict[int, int]] = defaultdict(dict)
    per_view_assigned: Dict[int, Dict[int, str]] = defaultdict(dict)
    per_view_names: Dict[int, str] = {}
    n_train_views_total = 0
    bg = torch.zeros(3, device="cuda")
    sam3_labels = [str(x) for x in target_labels]
    margin_pct = float(valid_mask_margin_pct)
    min_render = int(min_render_pixels) if int(min_render_pixels) > 0 else int(min_sam_pixels)

    # Integer codes for the dense label maps.
    label_to_code = {"none": 0, "sam2": 1}
    code_to_label = {0: "none", 1: "sam2"}
    for code, lab in enumerate(sam3_labels, start=2):
        label_to_code[str(lab)] = code
        code_to_label[code] = str(lab)

    for vidx, cam in enumerate(cameras):
        if getattr(cam, "objects", None) is None:
            continue
        stem = os.path.splitext(str(getattr(cam, "image_name", f"view_{vidx}")))[0]
        vocab = _read_label_json(label_json_root, stem)
        if not vocab:
            continue
        n_train_views_total += 1
        per_view_names[int(vidx)] = str(stem)

        with torch.no_grad():
            pkg = render(cam, gaussians, pipeline, bg)
            obj_map = pkg["render_object"]
            cdim, h, w = int(obj_map.shape[0]), int(obj_map.shape[1]), int(obj_map.shape[2])
            pred = (codebook_t @ obj_map.view(cdim, -1)).argmax(dim=0).view(h, w)

        if margin_pct > 0.0:
            yy = torch.arange(h, device=pred.device).view(h, 1).expand(h, w)
            xx = torch.arange(w, device=pred.device).view(1, w).expand(h, w)
            x_lo = float(w) * margin_pct
            x_hi = float(w) * (1.0 - margin_pct)
            y_lo = float(h) * margin_pct
            y_hi = float(h) * (1.0 - margin_pct)
            valid_rect = (xx > x_lo) & (xx < x_hi) & (yy > y_lo) & (yy < y_hi)
        else:
            valid_rect = torch.ones((h, w), device=pred.device, dtype=torch.bool)

        pred_for_render_count = pred[valid_rect]
        gids_all, cnts_all = torch.unique(pred_for_render_count, return_counts=True)
        gid_to_render_cnt = {
            int(gt_.item()): int(ct.item())
            for gt_, ct in zip(gids_all, cnts_all)
            if int(gt_.item()) in valid_gid_set and int(ct.item()) > 0
        }
        for gid, cnt in gid_to_render_cnt.items():
            group_render_pixels[gid] += cnt
            group_observed_views[gid].add(int(vidx))
            per_view_render_pixels[gid][int(vidx)] = cnt

        gt = cam.objects.long()
        if gt.dim() == 3:
            gt = gt.squeeze(0)
        if int(gt.shape[0]) != h or int(gt.shape[1]) != w:
            gt = F.interpolate(
                gt.unsqueeze(0).unsqueeze(0).float(),
                size=(h, w),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()

        # Dense semantic map over the image (0=none).
        label_code = torch.zeros((h, w), device=pred.device, dtype=torch.long)
        for sid, raw_label in vocab.items():
            if int(sid) <= 0:
                continue
            raw_norm = normalize_label(str(raw_label))
            if "sam2" in raw_norm:
                lab = "sam2"
            else:
                lab = _canonicalize_semantic_label(raw_label)
            if lab not in label_to_code or lab == "none":
                continue
            sam_mask = gt == int(sid)
            if int(sam_mask.sum().item()) <= 0:
                continue
            label_code[sam_mask] = int(label_to_code[lab])

        for gid, render_cnt in gid_to_render_cnt.items():
            if int(render_cnt) < int(min_render):
                continue
            group_mask = (pred == int(gid)) & valid_rect
            codes = label_code[group_mask]
            if codes.numel() <= 0:
                continue
            bucket_px = _label_pixel_buckets(codes, int(render_cnt), code_to_label)

            winner, _ = max(
                bucket_px.items(),
                key=lambda kv: (int(kv[1]), str(kv[0]) != "none", str(kv[0]) != "sam2", str(kv[0])),
            )
            view_vote_counts[(int(gid), str(winner))] += 1
            group_valid_vote_views[int(gid)].add(int(vidx))
            group_valid_render_pixels[int(gid)] += int(render_cnt)
            per_view_assigned[int(gid)][int(vidx)] = str(winner)

            for lab, px in bucket_px.items():
                if int(px) <= 0:
                    continue
                votes[(int(gid), str(lab))] += int(px)
                cur = per_view_label_pixels[int(gid)][int(vidx)]
                if str(lab) != "none":
                    cur[str(lab)] = int(cur.get(str(lab), 0)) + int(px)

    rows: List[Dict[str, Any]] = []
    vote_by_group_label: Dict[int, Dict[str, int]] = {}
    for gid in sorted(valid_gid_set):
        label_vote = {label: int(votes.get((gid, label), 0)) for label in sam3_labels}
        sam2_pixels = int(votes.get((gid, "sam2"), 0))
        vote_by_group_label[int(gid)] = dict(label_vote)
        vote_by_group_label[int(gid)]["sam2"] = int(sam2_pixels)

        # Percentages and assignment use the valid-vote views only.
        total_render = int(group_valid_render_pixels.get(gid, 0))
        none_pixels = int(votes.get((gid, "none"), max(0, total_render - int(sum(label_vote.values()) + sam2_pixels))))
        view_buckets = {lab: int(view_vote_counts.get((int(gid), str(lab)), 0)) for lab in sam3_labels}
        view_buckets["sam2"] = int(view_vote_counts.get((int(gid), "sam2"), 0))
        view_buckets["none"] = int(view_vote_counts.get((int(gid), "none"), 0))
        n_vote_views = int(len(group_valid_vote_views.get(gid, set())))
        assigned_label = "none"
        assigned_view_votes = 0
        if n_vote_views > 0:
            assigned_label, assigned_view_votes = max(
                view_buckets.items(),
                key=lambda kv: (int(kv[1]), str(kv[0]) != "none", str(kv[0]) != "sam2", str(kv[0])),
            )
        assigned_pixels = int(votes.get((gid, assigned_label), 0)) if assigned_label != "none" else int(none_pixels)
        assign_buckets_pixels: Dict[str, int] = dict(label_vote)
        assign_buckets_pixels["sam2"] = int(sam2_pixels)
        assign_buckets_pixels["none"] = int(none_pixels)
        label_view_votes = {str(k): int(v) for k, v in view_buckets.items()}
        assigned_ratio = float(assigned_view_votes / max(1, n_vote_views))

        total_sam3 = int(sum(label_vote.values()))
        majority_label = "none"
        majority_vote = 0
        if total_sam3 > 0:
            majority_label, majority_vote = max(label_vote.items(), key=lambda kv: (int(kv[1]), str(kv[0])))
            if int(majority_vote) <= 0:
                majority_label = "none"
                majority_vote = 0

        vote_with_none = dict(assign_buckets_pixels)
        best_label = "none"
        best_vote = 0
        if vote_with_none:
            best_label, best_vote = max(
                vote_with_none.items(),
                key=lambda kv: (int(kv[1]), str(kv[0]) != "none", str(kv[0])),
            )
        best_label_ratio = float(best_vote / max(1, total_render))
        none_ratio = float(none_pixels / max(1, total_render)) if total_render > 0 else 1.0
        n_render_views = int(len(group_observed_views.get(gid, set())))

        bucket_ratios = {}
        for k, v in sorted(assign_buckets_pixels.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            bucket_ratios[str(k)] = {
                "pixels": int(v),
                "ratio": float(v / max(1, total_render)) if total_render > 0 else 0.0,
                "n_views": int(label_view_votes.get(str(k), 0)),
            }

        sorted_labels = sorted(
            vote_with_none.items(),
            key=lambda kv: (-int(kv[1]), str(kv[0]) != "none", str(kv[0])),
        )
        top2 = []
        for label, pixels in sorted_labels[:2]:
            top2.append(
                {
                    "label": str(label),
                    "pixels": int(pixels),
                    "ratio": float(pixels / max(1, total_render)),
                    "n_views": int(label_view_votes.get(str(label), 0)),
                }
            )

        per_view_rows: List[Dict[str, Any]] = []
        view_ids = sorted(
            set(per_view_render_pixels.get(int(gid), {}).keys())
            | set(per_view_label_pixels.get(int(gid), {}).keys())
            | set(per_view_assigned.get(int(gid), {}).keys())
        )
        for vv in view_ids:
            render_px = int(per_view_render_pixels.get(int(gid), {}).get(int(vv), 0))
            label_px = {
                str(k): int(v)
                for k, v in (per_view_label_pixels.get(int(gid), {}).get(int(vv), {}) or {}).items()
                if int(v) > 0
            }
            labeled_px = int(sum(label_px.values()))
            none_px = int(max(0, render_px - labeled_px))
            is_valid_vote = render_px >= int(min_render)
            if render_px <= 0 and labeled_px <= 0:
                continue
            per_view_rows.append(
                {
                    "view_idx": int(vv),
                    "image_name": str(per_view_names.get(int(vv), "")),
                    "render_pixels": int(render_px),
                    "valid_vote": bool(is_valid_vote),
                    "view_assigned_label": str(per_view_assigned.get(int(gid), {}).get(int(vv), "")),
                    "label_pixel_votes": label_px,
                    "none_pixels": int(none_px),
                }
            )

        rows.append(
            {
                "group_id": int(gid),
                "best_label": str(best_label),
                "best_label_pixels": int(best_vote),
                "best_label_ratio": float(best_label_ratio),
                "majority_label": str(majority_label),
                "majority_label_pixels": int(majority_vote),
                "assigned_label": str(assigned_label),
                "assigned_label_pixels": int(assigned_pixels),
                "assigned_label_ratio": float(assigned_ratio),
                "sam2_pixels": int(sam2_pixels),
                "sam2_ratio": float(sam2_pixels / max(1, total_render)) if total_render > 0 else 0.0,
                "none_pixels": int(none_pixels),
                "none_ratio": float(none_ratio),
                "num_valid_views": int(n_vote_views),
                "n_train_views_total": int(n_train_views_total),
                "n_render_views": int(n_render_views),
                "n_vote_views": int(n_vote_views),
                "min_render_pixels": int(min_render),
                "label_view_votes": dict(label_view_votes),
                "total_render_pixels": int(total_render),
                "per_view_votes": per_view_rows,
                "label_candidates": {
                    "top2": top2,
                    "none_ratio": float(none_ratio),
                    "label_votes": dict(label_vote),
                    "label_view_votes": dict(label_view_votes),
                    "assign_buckets": bucket_ratios,
                },
            }
        )
    rows.sort(key=lambda x: int(x["group_id"]))
    return rows, vote_by_group_label


def _safe_label_suffix(label: str) -> str:
    out = normalize_label(label)
    if not out or out not in SEMANTIC_TEXT_LABEL_SET:
        raise ValueError(f"scene_graph_node label must be one of {sorted(SEMANTIC_TEXT_LABEL_SET)}, got: {label}")
    return out


def _select_cross_group_union_sources_for_target(
    features: np.ndarray,
    group_per_point: np.ndarray,
    codebook_np: np.ndarray,
    target_gid: int,
    cosine_threshold: float,
    min_count: int,
) -> List[int]:
    """
    Return source group ids (excluding target_gid) whose number of points satisfying
    cosine(feature, codebook[target_gid]) > cosine_threshold is > min_count.
    """
    if target_gid < 0 or target_gid >= int(codebook_np.shape[0]):
        return []

    feat = np.asarray(features, dtype=np.float64)
    grp = np.asarray(group_per_point, dtype=np.int64)
    cb = np.asarray(codebook_np, dtype=np.float64)
    if feat.ndim != 2 or cb.ndim != 2 or feat.shape[1] != cb.shape[1]:
        return []

    feat = feat / np.clip(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12, None)
    c = cb[target_gid].reshape(1, -1)
    c = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
    sim = (feat * c).sum(axis=1)

    mask = (grp != int(target_gid)) & (sim > float(cosine_threshold))
    if not np.any(mask):
        return []
    src_groups, counts = np.unique(grp[mask], return_counts=True)
    picked = [int(g) for g, c in zip(src_groups.tolist(), counts.tolist()) if int(c) > int(min_count)]
    return sorted(picked)


# Node construction from a checkpoint: every codebook group that passes the eligibility test
# (at least --min-high-sim-count Gaussians with cosine >= --codebook-cosine-threshold to the
# group's codebook entry) becomes one node, with a class_name from the 13 semantic classes or "none".
NONE_CLASS_NAME = "none"
CLASS_NAME_ALIAS = {
    "table": "desk",
}


def _resolve_out_path(iter_dir: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else iter_dir / p


def _bbox_from_meta(meta: dict) -> tuple:
    return dict(meta["bbox_tight"]), dict(meta["bbox_expanded"])


def _filter_global_indices_by_codebook_cosine(
    features: np.ndarray,
    codebook: np.ndarray,
    global_idx: np.ndarray,
    class_id: int,
    threshold: float,
) -> np.ndarray:
    """Subset of global_idx where cosine(obj_dc, codebook[class_id]) > threshold (L2-normalized)."""
    global_idx = np.asarray(global_idx, dtype=np.int64).reshape(-1)
    if global_idx.size == 0:
        return global_idx
    cb = np.asarray(codebook, dtype=np.float64)
    cid = int(class_id)
    if cid < 0 or cid >= int(cb.shape[0]):
        return global_idx
    f = np.asarray(features[global_idx], dtype=np.float64)
    c = cb[cid]
    fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    cn = c / (np.linalg.norm(c) + 1e-8)
    sim = (fn * cn.reshape(1, -1)).sum(axis=1)
    return global_idx[sim > float(threshold)]


def _normalize_codebook(codebook_np: np.ndarray) -> np.ndarray:
    cb = np.asarray(codebook_np, dtype=np.float64)
    if cb.ndim > 2:
        cb = cb.reshape(cb.shape[0], -1)
    return cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-12)


def _eligible_groups_by_high_sim_pct(
    features: np.ndarray,
    codebook_np: np.ndarray,
    group_per_point: np.ndarray,
    min_count: int,
    sim_thresh: float,
    min_high_sim_pct: float,
) -> Tuple[List[int], Dict[int, float]]:
    """Groups with count > min_count and (cos>=sim_thresh) fraction > min_high_sim_pct/100."""
    cb = _normalize_codebook(codebook_np)
    obj = np.asarray(features, dtype=np.float64)
    obj = obj / (np.linalg.norm(obj, axis=1, keepdims=True) + 1e-12)
    gids = np.asarray(group_per_point, dtype=np.int64)
    pct_map: Dict[int, float] = {}
    kept: List[int] = []
    unique_gids, counts = np.unique(gids, return_counts=True)
    for gid, count in zip(unique_gids.tolist(), counts.tolist()):
        gid = int(gid)
        count = int(count)
        if count <= int(min_count):
            continue
        if gid < 0 or gid >= int(cb.shape[0]):
            continue
        idx = np.where(gids == gid)[0]
        cos = (obj[idx] * cb[gid].reshape(1, -1)).sum(axis=1)
        pct = float(100.0 * np.count_nonzero(cos >= float(sim_thresh)) / max(1, idx.size))
        pct_map[gid] = pct
        if pct > float(min_high_sim_pct):
            kept.append(gid)
    return sorted(kept), pct_map


def _eligible_groups_by_high_sim_count(features, codebook_np, group_per_point,
                                        sim_thresh=0.9, min_high_sim_count=100):
    """Eligibility depends only on the absolute high-cosine point count."""
    if min_high_sim_count < 1:
        raise ValueError("min_high_sim_count must be positive")
    cb = _normalize_codebook(codebook_np)
    obj = np.asarray(features, dtype=np.float64)
    obj = obj / (np.linalg.norm(obj, axis=1, keepdims=True) + 1e-12)
    gids = np.asarray(group_per_point, dtype=np.int64)
    counts = {}
    for gid in np.unique(gids):
        gid = int(gid)
        if 0 <= gid < len(cb):
            cos = (obj[gids == gid] * cb[gid]).sum(axis=1)
            counts[gid] = int(np.count_nonzero(cos >= float(sim_thresh)))
    return sorted(g for g, count in counts.items() if count >= min_high_sim_count), counts


def extra_normalized_core_groups(
    features: np.ndarray,
    codebook_np: np.ndarray,
    already: Sequence[int],
    min_count: int,
    sim_thresh: float,
    min_high_sim_pct: float,
) -> List[int]:
    """Extra group ids from L2-normalized codebook assignment.

    Used only after the original scene graph exists (eval / SAM3-only), so init
    ROI / target construction is unchanged.
    """
    cb = _normalize_codebook(codebook_np)
    obj = np.asarray(features, dtype=np.float64)
    obj = obj / (np.linalg.norm(obj, axis=1, keepdims=True) + 1e-12)
    gids_norm = np.argmax(obj @ cb.T, axis=1).astype(np.int64)
    already_set = set(int(g) for g in already)
    extra: List[int] = []
    unique_n, counts_n = np.unique(gids_norm, return_counts=True)
    abs_core_min = max(int(min_count), 10 * int(min_count))
    for gid, count in zip(unique_n.tolist(), counts_n.tolist()):
        gid = int(gid)
        count = int(count)
        if gid in already_set or count <= int(min_count) or gid < 0 or gid >= int(cb.shape[0]):
            continue
        idx = np.where(gids_norm == gid)[0]
        n_high = int(np.count_nonzero((obj[idx] * cb[gid].reshape(1, -1)).sum(axis=1) >= float(sim_thresh)))
        pct = float(100.0 * n_high / max(1, idx.size))
        if pct > float(min_high_sim_pct) or n_high > int(abs_core_min):
            extra.append(gid)
    return sorted(extra)


def _parse_class_list(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        for part in str(v).replace(",", " ").split():
            lab = _safe_label_suffix(part)
            if not lab or lab in seen:
                continue
            seen.add(lab)
            out.append(lab)
    return out


def _resolve_node_class_name(assigned_label: str) -> str:
    """Map a SAM vote label to a scene-graph class_name (13-class set or none)."""
    lab = normalize_label(str(assigned_label))
    lab = CLASS_NAME_ALIAS.get(lab, lab)
    if lab in SEMANTIC_TEXT_LABEL_SET:
        return lab
    return NONE_CLASS_NAME


def _cluster_bucket_and_class(
    group_id: int,
    semantic_by_gid: Dict[int, Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[str, str]:
    """
    Assign label by argmax over *view votes*.
    First pass: a view is valid iff this group's rendered pixels inside the
    image margin >= min_render_pixels. All classes use that same region.

    class_name is always one of the 13 semantic classes, or "none".
    Target only if that class_name is in --target-classes.
    """
    target_classes = set(_parse_class_list(getattr(args, "target_classes", []) or []))
    semantic_row = semantic_by_gid.get(int(group_id))
    if not semantic_row:
        return "non_target_node", NONE_CLASS_NAME

    assigned_label = str(semantic_row.get("assigned_label", NONE_CLASS_NAME))
    # Prefer view-vote count when available.
    view_votes = dict(semantic_row.get("label_view_votes") or {})
    assigned_view_votes = int(view_votes.get(assigned_label, 0))
    assigned_pixels = int(semantic_row.get("assigned_label_pixels", 0))
    if assigned_label in {"", NONE_CLASS_NAME, "sam2"} or (assigned_view_votes <= 0 and assigned_pixels <= 0):
        return "non_target_node", NONE_CLASS_NAME

    class_name = _resolve_node_class_name(assigned_label)
    if class_name in target_classes:
        return "target_node", class_name
    return "non_target_node", class_name


def _class_from_vote_row(semantic_row: Optional[Dict[str, Any]]) -> str:
    if not semantic_row:
        return NONE_CLASS_NAME
    assigned_label = str(semantic_row.get("assigned_label", NONE_CLASS_NAME))
    view_votes = dict(semantic_row.get("label_view_votes") or {})
    assigned_view_votes = int(view_votes.get(assigned_label, 0))
    assigned_pixels = int(semantic_row.get("assigned_label_pixels", 0))
    if assigned_label in {"", NONE_CLASS_NAME, "sam2"} or (assigned_view_votes <= 0 and assigned_pixels <= 0):
        return NONE_CLASS_NAME
    return _resolve_node_class_name(assigned_label)


def best_vote_row(semantic_by_gid: Dict[int, Dict[str, Any]], gids: Sequence[int]) -> Optional[Dict[str, Any]]:
    """Vote row of the node's member group with the most voting views.

    A merged node lists several groups; the first one can be a tiny fragment that renders in only a few views,
    while the main group carries the real evidence, so the first group's vote must not decide the node's class.
    """
    rows = [semantic_by_gid[int(g)] for g in gids if int(g) in semantic_by_gid]
    return max(rows, key=lambda r: int(r.get("n_vote_views", 0))) if rows else None


def _relabel_none_nodes(
    nodes: List[Dict[str, Any]],
    semantic_by_gid: Dict[int, Dict[str, Any]],
    target_classes: Sequence[str] = (),
) -> int:
    """Rematch none nodes without margin. Only none -> stuff is allowed."""
    del target_classes
    n_updated = 0
    for node in nodes:
        if bool(node.get("target", False)):
            continue
        old_name = str(node.get("class_name") or NONE_CLASS_NAME).strip() or NONE_CLASS_NAME
        if old_name != NONE_CLASS_NAME:
            continue
        raw_gid = node.get("group_id")
        try:
            gids = [int(g) for g in (raw_gid if isinstance(raw_gid, (list, tuple)) else [raw_gid])]
        except (TypeError, ValueError):
            continue
        row = best_vote_row(semantic_by_gid, gids)
        new_name = _class_from_vote_row(row)
        if new_name not in STUFF_SEMANTIC_LABEL_SET:
            continue
        node["class_name"] = new_name
        node["target"] = False
        node["sam3_vote"] = _sam3_vote_summary(row, rematch=True)
        n_updated += 1
    return n_updated


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build initial scene graph from checkpoint (ckpt + SAM + label JSON)."
    )
    p.add_argument(
        "iteration_dir",
        type=str,
        help="Directory with point_cloud.ply and classifier.pth (e.g. iteration_20000).",
    )
    p.add_argument("--source-path", type=str, required=True)
    p.add_argument("--object-path", type=str, required=True)
    p.add_argument("--label-json-root", type=str, default="")
    p.add_argument("--images", type=str, default="images")
    p.add_argument("--resolution", type=int, default=2)
    p.add_argument("--num-objects", type=int, default=16)
    p.add_argument("--roi-k", type=int, default=8)
    p.add_argument("--roi-overlap-tolerance", type=float, default=0.5)
    p.add_argument("--roi-top-group-fraction", type=float, default=0.9)
    p.add_argument("--roi-margin-fraction", type=float, default=0.1)
    p.add_argument("--roi-eps", type=float, default=1e-6)
    p.add_argument("--codebook-cosine-threshold", type=float, default=0.9)
    p.add_argument(
        "--cross-group-union-min-count",
        type=int,
        default=100,
        help="Target-only: include other groups with > this many high-cosine points in ROI seed.",
    )
    p.add_argument("--min-high-sim-count", type=int, default=100,
                   help="Require at least this many high-cosine assigned points per group.")
    p.add_argument(
        "--target-classes",
        nargs="+",
        default=["cabinet", "bed", "chair", "desk", "sofa", "monitor"],
        help=(
            "Class names that become target nodes after view-vote assignment. "
            "Example: --target-classes cabinet bed chair desk sofa monitor"
        ),
    )
    p.add_argument(
        "--min-render-pixels",
        type=int,
        default=1000,
        help=(
            "A train view is a valid vote for a group only if that group's full "
            "rendered pixel count is >= this. That view then votes sam3/sam2/none. "
            "Thing-class pixels still use --valid-mask-margin-pct; stuff does not."
        ),
    )
    p.add_argument(
        "--valid-mask-margin-pct",
        type=float,
        default=0.1,
        help=(
            "Fractional image-edge margin applied only to thing classes "
            "(cabinet/bed/chair/desk/sofa/monitor). Stuff classes "
            "(wall/floor/ceiling/door/window/picture/curtain) vote on the full image."
        ),
    )
    p.add_argument(
        "--out-json",
        type=str,
        default="initial_scene_graph.json",
        help="Output JSON (relative paths resolve under iteration_dir).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print warnings and errors.",
    )
    return p.parse_args()


def _sam3_vote_summary(row: Optional[Dict[str, Any]], rematch: bool = False) -> Dict[str, Any]:
    """Evidence behind a node's SAM3 class: used to decide whether the VLM has to look at it."""
    if not row:
        return {"label": NONE_CLASS_NAME, "n_vote_views": 0, "view_share": 0.0, "pixel_ratio": 0.0, "rematch": bool(rematch)}
    label = str(row.get("assigned_label", NONE_CLASS_NAME))
    n_views = int(row.get("n_vote_views", 0))
    votes = int((row.get("label_view_votes") or {}).get(label, 0))
    return {
        "label": _resolve_node_class_name(label) if label not in {"", NONE_CLASS_NAME, "sam2"} else NONE_CLASS_NAME,
        "n_vote_views": n_views,
        "view_votes": votes,
        "view_share": float(votes / max(1, n_views)),
        "pixel_ratio": float(row.get("assigned_label_ratio", 0.0)),
        "none_ratio": float(row.get("none_ratio", 0.0)),
        "rematch": bool(rematch),
    }


def _make_target_node(
    node_id: int,
    group_id: int,
    class_name: str,
    bbox_tight: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "id": int(node_id),
        "group_id": int(group_id),
        "target": True,
        "class_name": str(class_name),
        "text_prompt": "",
        "bbox_tight": {k: float(bbox_tight[k]) for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")},
    }


def _make_non_target_node(
    node_id: int,
    group_id: int,
    class_name: str,
    bbox_tight: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    node = {
        "id": int(node_id),
        "group_id": int(group_id),
        "target": False,
        "class_name": str(class_name) if str(class_name).strip() else NONE_CLASS_NAME,
    }
    if isinstance(bbox_tight, dict):
        keys = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
        if all(k in bbox_tight for k in keys):
            node["bbox_tight"] = {k: float(bbox_tight[k]) for k in keys}
    return node


def main() -> None:
    args = _parse_args()

    iter_dir = Path(args.iteration_dir).resolve()
    ply_path = iter_dir / "point_cloud.ply"
    classifier_path = iter_dir / "classifier.pth"
    if not iter_dir.is_dir():
        raise FileNotFoundError(f"iteration_dir not found: {iter_dir}")
    if not ply_path.is_file():
        raise FileNotFoundError(f"point_cloud.ply not found: {ply_path}")
    if not classifier_path.is_file():
        raise FileNotFoundError(f"classifier.pth not found: {classifier_path}")

    iteration_name = iter_dir.name
    if iteration_name.startswith("iteration_"):
        model_root = iter_dir.parent.parent
    else:
        model_root = iter_dir

    features = read_object_features_from_ply(str(ply_path), args.num_objects)
    codebook_np = load_codebook(str(classifier_path), args.num_objects)
    if hasattr(codebook_np, "detach"):
        codebook_np = codebook_np.detach().cpu().numpy()
    else:
        codebook_np = np.asarray(codebook_np)

    group_per_point = compute_class_ids_np(features, codebook_np, normalize=True).astype(np.int64)
    n_codebook_ids = int(np.unique(group_per_point).size)
    eligible_groups, high_sim_counts = _eligible_groups_by_high_sim_count(
        features=features,
        codebook_np=codebook_np,
        group_per_point=group_per_point,
        sim_thresh=float(args.codebook_cosine_threshold),
        min_high_sim_count=int(args.min_high_sim_count),
    )
    eligible_group_set = set(int(g) for g in eligible_groups)
    quiet = bool(getattr(args, "quiet", False))

    if not quiet:
        print(f"[nodes] original codebook id number: {n_codebook_ids}")
        print(
            f"[nodes] after filter "
            f"(cos>={float(args.codebook_cosine_threshold):.2f} "
            f"count>={int(args.min_high_sim_count)}): "
            f"{len(eligible_groups)}"
        )

    mu_all, sigma_all = load_gaussians_from_ply(ply_path, scales_are_log=True, eps=float(args.roi_eps))
    if mu_all.shape[0] != group_per_point.shape[0]:
        raise RuntimeError("point count mismatch between features and gaussian geometry")

    label_json_root = _resolve_label_json_root(
        args.source_path,
        args.object_path,
        label_json_root=str(args.label_json_root or ""),
    )
    target_labels = _discover_labels(label_json_root)
    semantic_by_gid: Dict[int, Dict[str, Any]] = {}
    vote_cameras = None
    vote_gaussians = None
    vote_codebook_t = None
    vote_pipeline = None
    if label_json_root:
        gaussians, codebook_t, cameras, pipeline = _build_scene(
            model_root=model_root,
            source_path=args.source_path,
            object_path=args.object_path,
            render_point_cloud_path=ply_path,
            render_classifier_path=classifier_path,
            images=args.images,
            resolution=int(args.resolution),
            num_objects=int(args.num_objects),
        )
        semantic_rows, _ = _compute_semantic_votes(
            cameras=cameras,
            gaussians=gaussians,
            codebook_t=codebook_t,
            pipeline=pipeline,
            label_json_root=label_json_root,
            valid_gid_set=eligible_group_set,
            target_labels=target_labels,
            min_sam_pixels=int(args.min_render_pixels),
            valid_mask_margin_pct=float(args.valid_mask_margin_pct),
            min_render_pixels=int(args.min_render_pixels),
        )
        semantic_by_gid = {int(r["group_id"]): r for r in semantic_rows}
        vote_cameras = cameras
        vote_gaussians = gaussians
        vote_codebook_t = codebook_t
        vote_pipeline = pipeline
    else:
        print("[warn] No label JSON root; groups will not get confident semantic classes.", file=sys.stderr)

    n_kept: Dict[int, int] = {}  # ROI Gaussian count per group, used by the merge stage
    raw_target: List[Dict[str, Any]] = []
    raw_non_target: List[Dict[str, Any]] = []

    for group_id in eligible_groups:
        group_id = int(group_id)
        idx = np.where(group_per_point == group_id)[0]
        if idx.size == 0:
            continue

        final_bucket, normalized_class = _cluster_bucket_and_class(
            group_id=group_id,
            semantic_by_gid=semantic_by_gid,
            args=args,
        )

        roi_seed_idx = idx
        if final_bucket == "target_node":
            src_groups = _select_cross_group_union_sources_for_target(
                features=features,
                group_per_point=group_per_point,
                codebook_np=codebook_np,
                target_gid=group_id,
                cosine_threshold=float(args.codebook_cosine_threshold),
                min_count=int(args.cross_group_union_min_count),
            )
            src_groups = [g for g in src_groups if g != group_id]
            if src_groups:
                union_groups = np.asarray([group_id] + [int(g) for g in src_groups], dtype=np.int64)
                roi_seed_idx = np.where(np.isin(group_per_point, union_groups))[0]
                if not quiet:
                    print(
                        f"[roi-union] target group={group_id} includes source groups={src_groups} "
                        f"(seed points={int(roi_seed_idx.size)})"
                    )

        idx_f = _filter_global_indices_by_codebook_cosine(
            features,
            codebook_np,
            roi_seed_idx,
            group_id,
            float(args.codebook_cosine_threshold),
        )
        if idx_f.size < int(args.min_high_sim_count):
            continue

        mu = mu_all[idx_f]
        sigma = sigma_all[idx_f]
        _, _, labels, _ = group_gaussians(
            mu=mu,
            Sigma=sigma,
            k=int(args.roi_k),
            eps=float(args.roi_eps),
            overlap_tolerance=float(args.roi_overlap_tolerance),
        )
        labels = np.asarray(labels, dtype=np.int64)
        keep_local, meta = spatial_roi_crop_from_labels(
            xyz=mu,
            labels=labels,
            top_group_fraction=float(args.roi_top_group_fraction),
            margin_fraction=float(args.roi_margin_fraction),
            restrict_to_selected_groups=False,
        )
        keep_local = np.asarray(keep_local, dtype=bool)
        if not np.any(keep_local):
            continue

        n_kept[group_id] = int(np.count_nonzero(keep_local))
        bbox_tight, bbox_expanded = _bbox_from_meta(meta)

        if final_bucket == "target_node":
            raw_target.append(
                {
                    "group_id": group_id,
                    "class_name": str(normalized_class),
                    "bbox_tight": bbox_tight,
                }
            )
        else:
            raw_non_target.append(
                {
                    "group_id": group_id,
                    "class_name": str(normalized_class) if str(normalized_class).strip() else NONE_CLASS_NAME,
                    "bbox_tight": bbox_tight,
                }
            )

    nodes: List[Dict[str, Any]] = []
    node_id = 1
    for item in raw_target:
        nodes.append(
            _make_target_node(
                node_id=node_id,
                group_id=item["group_id"],
                class_name=item["class_name"],
                bbox_tight=item["bbox_tight"],
            )
        )
        node_id += 1
    for item in raw_non_target:
        nodes.append(
            _make_non_target_node(
                node_id=node_id,
                group_id=item["group_id"],
                class_name=str(item.get("class_name") or NONE_CLASS_NAME),
                bbox_tight=item.get("bbox_tight") if isinstance(item.get("bbox_tight"), dict) else None,
            )
        )
        node_id += 1

    for node in nodes:
        node["n_gaussians"] = n_kept[int(node["group_id"])]
        node["sam3_vote"] = _sam3_vote_summary(semantic_by_gid.get(int(node["group_id"])))

    none_gids = [
        int(n["group_id"])
        for n in nodes
        if str(n.get("class_name") or NONE_CLASS_NAME) == NONE_CLASS_NAME and not bool(n.get("target", False))
    ]
    if none_gids and vote_cameras is not None:
        none_rows, _ = _compute_semantic_votes(
            cameras=vote_cameras,
            gaussians=vote_gaussians,
            codebook_t=vote_codebook_t,
            pipeline=vote_pipeline,
            label_json_root=label_json_root,
            valid_gid_set=set(none_gids),
            target_labels=target_labels,
            min_sam_pixels=int(args.min_render_pixels),
            valid_mask_margin_pct=0.0,
            min_render_pixels=int(args.min_render_pixels),
        )
        n_none_updated = _relabel_none_nodes(
            nodes,
            {int(r["group_id"]): r for r in none_rows},
            getattr(args, "target_classes", []) or [],
        )
        if n_none_updated and not quiet:
            print(f"[nodes] none-node rematch without margin: updated {n_none_updated}")

    doc = {"nodes": nodes}

    out_json = _resolve_out_path(iter_dir, args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    missing_class = [
        int(n["id"])
        for n in nodes
        if not str(n.get("class_name", "")).strip()
    ]
    if missing_class:
        raise RuntimeError(f"nodes missing class_name: {missing_class}")

    class_counts = Counter(str(n.get("class_name", NONE_CLASS_NAME)) for n in nodes)
    if not quiet:
        print(f"[nodes] left node number: {len(nodes)}")
        print(f"[nodes]   target classes: {_parse_class_list(getattr(args, 'target_classes', []) or [])}")
        print(f"[nodes]   semantic classes: {list(SEMANTIC_TEXT_LABELS)} (else class_name={NONE_CLASS_NAME})")
        print(f"[nodes]   target node n: {sum(bool(n['target']) for n in nodes)}")
        print(
            "[nodes]   class_name counts: "
            + ", ".join(f"{name}={count}" for name, count in sorted(class_counts.items()))
        )
        print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
