#!/usr/bin/env python3
"""Per-Gaussian 3D semantic eval against ScanNet v2 vertex labels.

Codebook argmax + scene graph → mIoU, mAcc, and optional PQ(Thing)/PQ(Stuff).

Two scopes (always both):
  VLSplat: every Gaussian, group_id → scene-graph class (unmapped = 0)
  VLSplat+ROI-thing: stuff = all group points; thing = group ∩ that node's bbox
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_CLASS_NAMES: Tuple[str, ...] = (
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
    "desk",
    "curtain",
    "monitor",
)
NUM_SUBSET_CLASSES = len(EVAL_CLASS_NAMES) + 1  # 0 ignore, 1..13 classes
NAME_ALIASES: Dict[str, str] = {
    "table": "desk",
    "table/desk": "desk",
    "desk/table": "desk",
    "shelves": "cabinet",
    "bookshelf": "cabinet",
    "bookshelves": "cabinet",
}
# cabinet, bed, chair, sofa, picture, desk, monitor
THING_SUBSET_IDS: frozenset[int] = frozenset({3, 5, 6, 7, 10, 11, 13})
ORPHAN_INSTANCE_OFFSET = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-Gaussian 3D semantic eval (mIoU / mAcc / PQ)")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--scene-graph-json", default="")
    parser.add_argument("--gaussian-ply", default="")
    parser.add_argument("--classifier-path", default="")
    parser.add_argument("--labels-ply", default="")
    parser.add_argument("--segs-json", default="")
    parser.add_argument("--aggregation-json", default="")
    parser.add_argument("--feature-prefix", default="obj_dc_")
    parser.add_argument("--xyz-atol", type=float, default=1e-5)
    parser.add_argument("--nn-max-dist", type=float, default=1e-5)
    parser.add_argument("--roi-bbox", choices=("tight", "expanded"), default="tight")
    parser.add_argument("--no-panoptic", action="store_true")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def class_name_to_subset_id(name: str) -> int:
    if not name:
        return 0
    n = str(name).strip().lower().replace("_", " ")
    n = NAME_ALIASES.get(n, n)
    try:
        return int(EVAL_CLASS_NAMES.index(n)) + 1
    except ValueError:
        return 0


def iter_group_ids(node: Mapping[str, Any]) -> List[int]:
    raw = node.get("group_id", node.get("group_ids"))
    out: List[int] = []
    if isinstance(raw, list):
        for item in raw:
            try:
                out.append(int(item))
            except Exception:
                continue
    elif raw is not None:
        try:
            out.append(int(raw))
        except Exception:
            pass
    canon = node.get("canonical_group_id")
    if canon is not None:
        try:
            out.append(int(canon))
        except Exception:
            pass
    return sorted(set(out))


SCANNET_ID_TO_CLASS: Dict[str, List[int]] = {  # ScanNet label id (labels.combined.tsv `id`) -> eval class
    "wall": [1, 128, 947, 1165, 1204, 1243],
    "floor": [3, 417, 1347],
    "cabinet": [7, 8, 18, 29, 75, 385, 815, 1164, 1173, 1283, 1316, 1322, 1335],
    "ceiling": [41, 1225],
    "bed": [11, 494, 786, 1191, 1349],
    "chair": [2, 10, 23, 74, 885, 1184, 1291, 1338],
    "sofa": [6, 1313],
    "door": [5, 161, 276, 569, 649, 1167, 1208, 1345],
    "window": [16],
    "picture": [15, 1188, 1218],
    "desk": [4, 9, 24, 44, 45, 108, 222, 1193, 1355],
    "curtain": [21],
    "monitor": [19],
}

NYU40_ID_TO_CLASS: Dict[str, List[int]] = {  # NYU40 id -> eval class
    "wall": [1],
    "floor": [2],
    "cabinet": [3, 10, 15],
    "ceiling": [22],
    "bed": [4],
    "chair": [5],
    "sofa": [6],
    "door": [8],
    "window": [9],
    "picture": [11],
    "desk": [7, 14],
    "curtain": [16],
}


def _class_table_to_lut(table: Mapping[str, Sequence[int]]) -> np.ndarray:
    lut = np.zeros(max(i for ids in table.values() for i in ids) + 1, dtype=np.int64)
    for name, ids in table.items():
        lut[list(ids)] = class_name_to_subset_id(name)
    return lut


def scannet_ids_to_subset_semantic(label_ids: np.ndarray, lut: np.ndarray) -> np.ndarray:
    ids = label_ids.astype(np.int64)
    out = np.zeros(ids.shape, dtype=np.int64)
    ok = (ids >= 0) & (ids < lut.size)
    out[ok] = lut[ids[ok]]
    return out


def calculate_iou_3d(preds: np.ndarray, targets: np.ndarray, num_classes: int):
    iou_per_class = np.zeros(num_classes, dtype=np.float64)
    acc_per_class = np.zeros(num_classes, dtype=np.float64)
    mask_per_class = np.zeros(num_classes, dtype=bool)
    for c in range(num_classes):
        pred_class = preds == c
        target_class = targets == c
        intersection = float((pred_class & target_class).sum())
        union = float((pred_class | target_class).sum())
        if union != 0.0:
            iou_per_class[c] = intersection / union
        target_count = float(target_class.sum())
        if target_count != 0.0:
            acc_per_class[c] = intersection / target_count
            mask_per_class[c] = True
    return iou_per_class, acc_per_class, mask_per_class


def resolve_pred_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and path.name.endswith(".ply"):
        return path.parent
    if (path / "point_cloud.ply").is_file():
        return path
    candidates = list(path.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(f"point_cloud.ply not found under {path}")

    def _iter_key(ply_path: Path) -> int:
        try:
            return int(ply_path.parent.name.split("_")[-1])
        except Exception:
            return -1

    candidates.sort(key=_iter_key)
    return candidates[-1].parent


def resolve_gt_dir(gt_dir: Path) -> Path:
    gt_dir = gt_dir.expanduser().resolve()
    if gt_dir.is_file():
        return gt_dir.parent
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT path not found: {gt_dir}")
    return gt_dir


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def collect_feature_names(dtype: np.dtype, prefix: str) -> List[str]:
    names = [x for x in (dtype.names or []) if x.startswith(prefix)]
    if not names:
        raise ValueError(f"No PLY fields with prefix {prefix!r}")
    names.sort(key=lambda x: int(x[len(prefix) :]))
    return names


def load_classifier_matrix(classifier_path: Path, feat_dim: int) -> np.ndarray:
    try:
        ckpt = torch.load(str(classifier_path), map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError, RuntimeError):
        ckpt = torch.load(str(classifier_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        w = ckpt.get("weight", list(ckpt.values())[0])
        w = w.squeeze()
        if w.dim() == 4:
            w = w.squeeze()
    else:
        w = ckpt
    w = w.detach().cpu().numpy().astype(np.float32)
    if w.ndim != 2:
        raise ValueError(f"classifier must be 2D, got {w.shape}")
    if w.shape[1] != feat_dim:
        raise ValueError(f"feature dim ply={feat_dim} vs classifier={w.shape[1]}")
    return w


def per_gaussian_gt_alignment(
    gaussian_vertex: np.ndarray,
    labels_vertex: np.ndarray,
    xyz_atol: float,
    nn_max_dist: float,
) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]:
    def stack_xyz(v: np.ndarray) -> np.ndarray:
        return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    xyz_g = stack_xyz(gaussian_vertex)
    xyz_l = stack_xyz(labels_vertex)
    labels_raw = np.asarray(labels_vertex["label"], dtype=np.int64)
    n_g = int(xyz_g.shape[0])
    n_l = int(xyz_l.shape[0])
    if n_g == n_l:
        diff = np.abs(xyz_g - xyz_l)
        if float(diff.max()) <= float(xyz_atol):
            stats = {
                "mode": "row_aligned",
                "n_pred": n_g,
                "n_gt_mesh": n_l,
                "n_unique_gt": n_g,
                "max_abs_diff": float(diff.max()),
                "unique_1to1": True,
            }
            return labels_raw.copy(), np.arange(n_l, dtype=np.int64), "row_aligned", stats
    if float(nn_max_dist) <= 0.0:
        raise ValueError(f"Gaussian N={n_g} vs labels N={n_l}; enable NN or align PLYs.")
    tree = cKDTree(xyz_l)
    dist, idx = tree.query(xyz_g, k=1, workers=-1)
    dist = np.asarray(dist, dtype=np.float64)
    max_d = float(np.max(dist))
    if max_d > float(nn_max_dist):
        raise ValueError(f"max NN label distance {max_d:.6e} > tol {nn_max_dist}")
    label_idx = idx.astype(np.int64)
    n_unique = int(np.unique(label_idx).size)
    stats = {
        "mode": "nearest_vertex",
        "n_pred": n_g,
        "n_gt_mesh": n_l,
        "n_unique_gt": n_unique,
        "n_gt_unused": int(n_l - n_unique),
        "n_pred_sharing_gt": int(n_g - n_unique),
        "nn_dist_min": float(dist.min()),
        "nn_dist_mean": float(dist.mean()),
        "nn_dist_max": max_d,
        "unique_1to1": bool(n_unique == n_g),
    }
    return labels_raw[label_idx], label_idx, "nearest_vertex", stats


def bbox_mask(xyz: np.ndarray, bbox: Mapping[str, Any]) -> np.ndarray:
    return (
        (xyz[:, 0] >= float(bbox["xmin"]))
        & (xyz[:, 0] <= float(bbox["xmax"]))
        & (xyz[:, 1] >= float(bbox["ymin"]))
        & (xyz[:, 1] <= float(bbox["ymax"]))
        & (xyz[:, 2] >= float(bbox["zmin"]))
        & (xyz[:, 2] <= float(bbox["zmax"]))
    )


def node_bbox(node: Mapping[str, Any], roi_bbox: str) -> Optional[dict]:
    roi = node.get("roi") if isinstance(node.get("roi"), dict) else {}
    bbox = None
    if isinstance(roi, dict):
        bbox = roi.get(f"bbox_{roi_bbox}")
    if bbox is None:
        bbox = node.get(f"bbox_{roi_bbox}") or node.get("bbox_tight")
    return bbox if isinstance(bbox, dict) else None


def vertex_instance_ids(seg_indices: np.ndarray, aggregation_path: Path) -> np.ndarray:
    agg = json.loads(aggregation_path.read_text(encoding="utf-8"))
    seg_to_obj: Dict[int, int] = {}
    for group in agg["segGroups"]:
        oid = int(group["objectId"])
        for seg in group["segments"]:
            seg_to_obj[int(seg)] = oid
    out = np.empty(seg_indices.shape[0], dtype=np.int64)
    for i, seg in enumerate(seg_indices.astype(np.int64)):
        oid = seg_to_obj.get(int(seg))
        out[i] = oid if oid is not None else ORPHAN_INSTANCE_OFFSET + int(seg)
    return out


def panoptic_evaluate_subset(
    pred_semantics: np.ndarray,
    pred_instances: np.ndarray,
    gt_semantics: np.ndarray,
    gt_instances: np.ndarray,
    *,
    thing: bool,
    allowed_subset_ids: set[int],
    thing_subset_ids: frozenset[int],
) -> Dict[str, Any]:
    pq = sq = rq = ap = 0.0
    num_valid_class = 0.0
    for cls_label in range(1, len(EVAL_CLASS_NAMES) + 1):
        if cls_label not in allowed_subset_ids:
            continue
        is_thing = cls_label in thing_subset_ids
        if thing and not is_thing:
            continue
        if not thing and is_thing:
            continue
        tp = fp = fn = 0
        iou_sum = 0.0
        pred_mask = pred_semantics == cls_label
        gt_mask = gt_semantics == cls_label
        unique_gt = np.unique(gt_instances[gt_mask])
        unique_pred = np.unique(pred_instances[pred_mask])
        upred = unique_pred.copy()
        for gt_inst in unique_gt:
            gt_inst_mask = gt_mask & (gt_instances == gt_inst)
            if int(gt_inst_mask.sum()) == 0:
                continue
            max_iou = 0.0
            matched_pred_inst = -1
            for pred_inst in upred:
                pred_inst_mask = pred_mask & (pred_instances == pred_inst)
                inter = int(np.sum(gt_inst_mask & pred_inst_mask))
                union = int(np.sum(gt_inst_mask | pred_inst_mask))
                iou = inter / union if union > 0 else 0.0
                if iou > max_iou:
                    max_iou = iou
                    matched_pred_inst = pred_inst
            if max_iou > 0.25:
                tp += 1
                iou_sum += max_iou
                upred = upred[upred != matched_pred_inst]
            else:
                fn += 1
        fp = len(upred)
        if (tp + fp + fn) > 0:
            pq += iou_sum / (tp + 0.5 * fp + 0.5 * fn)
            rq += tp / (tp + 0.5 * fp + 0.5 * fn)
            num_valid_class += 1.0
        if tp > 0:
            sq += iou_sum / tp
        if (tp + fp) > 0:
            ap += iou_sum / (tp + fp)
    if num_valid_class <= 0:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "AP": 0.0, "num_valid_classes": 0}
    inv = 1.0 / num_valid_class
    return {
        "PQ": pq * inv,
        "SQ": sq * inv,
        "RQ": rq * inv,
        "AP": ap * inv,
        "num_valid_classes": int(num_valid_class),
    }


def semantic_metrics(
    pred_sem: np.ndarray,
    gt_sem: np.ndarray,
    eval_mask: np.ndarray,
) -> Dict[str, Any]:
    pred = pred_sem[eval_mask]
    gt = gt_sem[eval_mask]
    ious, accs, masks = calculate_iou_3d(pred, gt, NUM_SUBSET_CLASSES)
    cls_range = np.zeros(NUM_SUBSET_CLASSES, dtype=bool)
    cls_range[1:NUM_SUBSET_CLASSES] = True
    use = masks & cls_range
    per_class = []
    for class_id in range(1, NUM_SUBSET_CLASSES):
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": EVAL_CLASS_NAMES[class_id - 1],
                "present_in_gt": bool(masks[class_id]),
                "gt_count": int((gt == class_id).sum()),
                "pred_count": int((pred == class_id).sum()),
                "iou": float(ious[class_id]),
                "acc": float(accs[class_id]),
            }
        )
    valid_names = [EVAL_CLASS_NAMES[i - 1] for i in range(1, NUM_SUBSET_CLASSES) if use[i]]
    absent_names = [EVAL_CLASS_NAMES[i - 1] for i in range(1, NUM_SUBSET_CLASSES) if not use[i]]
    n_ignore = int((gt == 0).sum())
    n_labeled = int((gt > 0).sum())
    n_pred_on_ignore = int(((gt == 0) & (pred > 0)).sum())
    n_eval = int(eval_mask.sum())
    return {
        "n_eval_gaussians": n_eval,
        "n_gt_labeled": n_labeled,
        "n_gt_ignore": n_ignore,
        "n_pred_on_ignore": n_pred_on_ignore,
        "mIoU": float(ious[use].mean()) if np.any(use) else float("nan"),
        "mAcc": float(accs[use].mean()) if np.any(use) else float("nan"),
        "valid_class_names": valid_names,
        "absent_class_names": absent_names,
        "per_class": per_class,
    }


def format_semantic_report(name: str, metrics: Mapping[str, Any]) -> str:
    lines = [
        f"[{name}] mIoU={metrics['mIoU']:.6f} mAcc={metrics['mAcc']:.6f} "
        f"n={metrics['n_eval_gaussians']}"
    ]
    if metrics.get("PQ_thing") is not None:
        lines[-1] += (
            f" PQ_T={metrics['PQ_thing']:.6f} PQ_S={metrics['PQ_stuff']:.6f}"
        )
        if int(metrics.get("n_stuff_pred", 1)) == 0:
            lines.append(
                "  note: PQ_S=0 because this mask has no stuff predictions "
                "(wall/floor/ceiling/door/window/curtain)."
            )
    n_eval = int(metrics["n_eval_gaussians"])
    n_ignore = int(metrics.get("n_gt_ignore", 0))
    pct = (100.0 * n_ignore / n_eval) if n_eval else 0.0
    lines.append(
        f"  ignore GT={n_ignore}/{n_eval} ({pct:.1f}%)  "
        f"labeled={int(metrics.get('n_gt_labeled', n_eval - n_ignore))}  "
        f"pred_on_ignore(FP)={int(metrics.get('n_pred_on_ignore', 0))}"
    )
    lines.append(
        "  ignore: not a scored class and not FN; "
        "predicting an eval class on ignore counts as FP"
    )
    lines.append("valid_classes=" + ", ".join(metrics["valid_class_names"]))
    absent = metrics.get("absent_class_names") or []
    if absent:
        lines.append("absent_candidates=" + ", ".join(absent) + " (no GT in this mask)")
    for row in metrics["per_class"]:
        if not row["present_in_gt"]:
            continue
        lines.append(
            f"  - {row['class_name']}: IoU={row['iou']:.6f} Acc={row['acc']:.6f} "
            f"GT={row['gt_count']} Pred={row['pred_count']}"
        )
    lines.append("")
    return "\n".join(lines)


def assign_full(group_ids: np.ndarray, nodes: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    group_to_node: Dict[int, Tuple[int, int]] = {}
    for node in nodes:
        nid = int(node.get("id", -1))
        sem = class_name_to_subset_id(str(node.get("class_name", "")))
        for gid in iter_group_ids(node):
            group_to_node[int(gid)] = (nid, sem)
    pred_sem = np.zeros(group_ids.shape[0], dtype=np.int64)
    pred_ins = np.zeros(group_ids.shape[0], dtype=np.int64)
    for i, gid in enumerate(group_ids.tolist()):
        mapped = group_to_node.get(int(gid))
        if mapped is None:
            continue
        pred_ins[i] = mapped[0]
        pred_sem[i] = mapped[1]
    return pred_sem, pred_ins


def assign_roi_thing_only(
    xyz: np.ndarray,
    group_ids: np.ndarray,
    nodes: Sequence[Mapping[str, Any]],
    roi_bbox: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """ROI-style mask, but bbox cropping applies only to thing nodes.

    Stuff: every Gaussian whose group_id belongs to a stuff node (no bbox).
    Thing: group_id AND inside that node's bbox; outside → dropped from eval.
    Unmapped groups stay in the eval as unlabeled (class 0) predictions; only
    out-of-bbox thing points are dropped.
    """
    pred_sem, pred_ins = assign_full(group_ids, nodes)
    n = int(group_ids.shape[0])
    stuff_keep = np.zeros(n, dtype=bool)
    thing_keep = np.zeros(n, dtype=bool)
    mapped = np.zeros(n, dtype=bool)
    thing_group = np.zeros(n, dtype=bool)
    for node in nodes:
        sem = class_name_to_subset_id(str(node.get("class_name", "")))
        if sem <= 0:
            continue
        gids = iter_group_ids(node)
        if not gids:
            continue
        in_group = np.isin(group_ids, np.asarray(gids, dtype=np.int32))
        mapped |= in_group
        if sem in THING_SUBSET_IDS:
            thing_group |= in_group
            bbox = node_bbox(node, roi_bbox)
            if bbox is None:
                continue
            thing_keep |= in_group & bbox_mask(xyz, bbox)
        else:
            stuff_keep |= in_group
    thing_oob = thing_group & ~thing_keep
    eval_mask = ~thing_oob
    unmapped = ~mapped
    stats: Dict[str, Any] = {
        "n_all": n,
        "n_kept": int(eval_mask.sum()),
        "n_kept_stuff": int(stuff_keep.sum()),
        "n_kept_thing": int(thing_keep.sum()),
        "n_removed": int((~eval_mask).sum()),
        "n_unmapped_kept_unlabeled": int(unmapped.sum()),
        "n_removed_thing_oob": int(thing_oob.sum()),
    }
    return pred_sem, pred_ins, eval_mask, stats


def maybe_panoptic(
    pred_sem: np.ndarray,
    pred_ins: np.ndarray,
    gt_sem: np.ndarray,
    gt_ins: Optional[np.ndarray],
    eval_mask: np.ndarray,
) -> Dict[str, Any]:
    if gt_ins is None:
        return {}
    allowed = set(range(1, NUM_SUBSET_CLASSES))
    thing_ids = THING_SUBSET_IDS
    thing_m = panoptic_evaluate_subset(
        pred_sem[eval_mask],
        pred_ins[eval_mask],
        gt_sem[eval_mask],
        gt_ins[eval_mask],
        thing=True,
        allowed_subset_ids=allowed,
        thing_subset_ids=thing_ids,
    )
    stuff_m = panoptic_evaluate_subset(
        pred_sem[eval_mask],
        pred_ins[eval_mask],
        gt_sem[eval_mask],
        gt_ins[eval_mask],
        thing=False,
        allowed_subset_ids=allowed,
        thing_subset_ids=thing_ids,
    )
    pred_e = pred_sem[eval_mask]
    stuff_pred = int(sum(int((pred_e == cid).sum()) for cid in allowed if cid not in thing_ids))
    return {
        "PQ_thing": float(thing_m["PQ"]),
        "PQ_stuff": float(stuff_m["PQ"]),
        "SQ_thing": float(thing_m["SQ"]),
        "SQ_stuff": float(stuff_m["SQ"]),
        "RQ_thing": float(thing_m["RQ"]),
        "RQ_stuff": float(stuff_m["RQ"]),
        "n_stuff_pred": stuff_pred,
        "n_thing_pred": int(sum(int((pred_e == cid).sum()) for cid in thing_ids)),
    }


def main() -> None:
    args = parse_args()
    pred_dir = resolve_pred_dir(Path(args.pred_dir))
    gt_dir = resolve_gt_dir(Path(args.gt_dir))
    gaussian_ply = Path(args.gaussian_ply).expanduser().resolve() if args.gaussian_ply else pred_dir / "point_cloud.ply"
    classifier_path = (
        Path(args.classifier_path).expanduser().resolve() if args.classifier_path else pred_dir / "classifier.pth"
    )
    scene_graph_json = (
        Path(args.scene_graph_json).expanduser().resolve() if args.scene_graph_json else pred_dir / "scene_graph.json"
    )
    label_candidates: List[Path] = []
    if args.labels_ply:
        label_candidates.append(Path(args.labels_ply).expanduser())
    label_candidates.append(gt_dir / f"{gt_dir.name.replace('gt_', '')}_vh_clean_2.labels.ply")
    label_candidates.extend(sorted(gt_dir.glob("*_vh_clean_2.labels.ply")))
    labels_ply = first_existing(label_candidates)
    if labels_ply is None:
        raise FileNotFoundError(f"labels.ply not found under {gt_dir}")
    scene = labels_ply.name.split("_vh_clean_2")[0]
    segs_candidates: List[Path] = []
    if args.segs_json:
        segs_candidates.append(Path(args.segs_json).expanduser())
    segs_candidates.append(gt_dir / f"{scene}_vh_clean_2.0.010000.segs.json")
    segs_json = first_existing(segs_candidates)
    agg_candidates: List[Path] = []
    if args.aggregation_json:
        agg_candidates.append(Path(args.aggregation_json).expanduser())
    agg_candidates.append(gt_dir / f"{scene}.aggregation.json")
    aggregation_json = first_existing(agg_candidates)

    ply_g = PlyData.read(str(gaussian_ply))
    vg = ply_g["vertex"].data
    ply_l = PlyData.read(str(labels_ply))
    vl = ply_l["vertex"].data
    xyz_g = np.stack([vg["x"], vg["y"], vg["z"]], axis=1).astype(np.float64)

    gt_ids_raw, gt_vertex_idx, align_mode, align_stats = per_gaussian_gt_alignment(
        vg, vl, float(args.xyz_atol), float(args.nn_max_dist)
    )
    max_lab = int(np.max(gt_ids_raw)) if gt_ids_raw.size else 0
    if max_lab <= 40:
        lut = _class_table_to_lut(NYU40_ID_TO_CLASS)
        gt_label_id_space = "nyu40id"
    else:
        lut = _class_table_to_lut(SCANNET_ID_TO_CLASS)
        gt_label_id_space = "scannet_id"
    gt_sem = scannet_ids_to_subset_semantic(gt_ids_raw, lut)

    gt_ins = None
    if not args.no_panoptic:
        if segs_json is None or aggregation_json is None:
            print("[warn] segs/aggregation missing; skip panoptic PQ", flush=True)
        else:
            seg_doc = json.loads(segs_json.read_text(encoding="utf-8"))
            seg_idx = np.array(seg_doc["segIndices"], dtype=np.int64)
            if seg_idx.shape[0] != vl.shape[0]:
                raise ValueError(
                    f"segIndices length {seg_idx.shape[0]} != label vertices {vl.shape[0]}"
                )
            gt_ins = vertex_instance_ids(seg_idx[gt_vertex_idx], aggregation_json)

    feat_names = collect_feature_names(vg.dtype, str(args.feature_prefix))
    feats = np.stack([vg[n] for n in feat_names], axis=1).astype(np.float32)
    codebook = load_classifier_matrix(classifier_path, feats.shape[1])
    group_ids = np.argmax(feats @ codebook.T, axis=1).astype(np.int32)

    nodes = json.loads(scene_graph_json.read_text(encoding="utf-8")).get("nodes") or []
    pred_full_sem, pred_full_ins = assign_full(group_ids, nodes)
    full_mask = np.ones(group_ids.shape[0], dtype=bool)
    metrics_full = semantic_metrics(pred_full_sem, gt_sem, full_mask)
    metrics_full.update(maybe_panoptic(
        pred_full_sem, pred_full_ins, gt_sem, gt_ins, full_mask
    ))

    pred_roi_th_sem, pred_roi_th_ins, roi_thing_mask, roi_thing_stats = assign_roi_thing_only(
        xyz_g, group_ids, nodes, str(args.roi_bbox)
    )
    if int(roi_thing_mask.sum()) == 0:
        raise RuntimeError("VLSplat+ROI-thing eval mask is empty.")
    metrics_roi_thing = semantic_metrics(pred_roi_th_sem, gt_sem, roi_thing_mask)
    metrics_roi_thing.update(maybe_panoptic(
        pred_roi_th_sem, pred_roi_th_ins, gt_sem, gt_ins, roi_thing_mask
    ))

    one_to_one = "yes" if align_stats.get("unique_1to1") else "no"
    proto_text = (
        "[protocol] 13 eval classes; table merged→desk; tv/television ignored; "
        "other GT → ignore(0)\n"
        "  candidates=" + ", ".join(EVAL_CLASS_NAMES) + "\n"
        "  mIoU/mAcc average only classes present in this GT mask\n"
        "  ignore has no class score and no FN; pred-on-ignore is FP for that class\n\n"
    )
    align_text = (
        f"[xyz-match] {align_mode}  pred={align_stats['n_pred']} gt_mesh={align_stats['n_gt_mesh']}\n"
        f"  unique_1to1={one_to_one}  unique_gt={align_stats['n_unique_gt']}  "
        f"unused_gt={align_stats.get('n_gt_unused', 0)}\n"
    )
    if align_stats["mode"] == "nearest_vertex":
        align_text += (
            f"  nn_dist min/mean/max="
            f"{align_stats['nn_dist_min']:.3e}/{align_stats['nn_dist_mean']:.3e}/{align_stats['nn_dist_max']:.3e}  "
            f"(fail if max>{args.nn_max_dist:g})\n"
        )
    align_text += (
        f"  gt_labels={labels_ply.name} id_space={gt_label_id_space}  "
        f"table+desk merged→desk  tv/television ignored\n\n"
    )

    def _metric_or_dash(metrics: Mapping[str, Any], key: str) -> str:
        value = metrics.get(key)
        if value is None:
            return "-"
        return f"{float(value):.6f}"

    summary_text = (
        "\n[summary]\n"
        f"{'method':<20} {'mIoU':>10} {'mAcc':>10} {'PRQ(T)':>10} {'PRQ(S)':>10}\n"
        f"{'VLSplat':<20} "
        f"{_metric_or_dash(metrics_full, 'mIoU'):>10} "
        f"{_metric_or_dash(metrics_full, 'mAcc'):>10} "
        f"{_metric_or_dash(metrics_full, 'PQ_thing'):>10} "
        f"{_metric_or_dash(metrics_full, 'PQ_stuff'):>10}\n"
        f"{'VLSplat+ROI-thing':<20} "
        f"{_metric_or_dash(metrics_roi_thing, 'mIoU'):>10} "
        f"{_metric_or_dash(metrics_roi_thing, 'mAcc'):>10} "
        f"{_metric_or_dash(metrics_roi_thing, 'PQ_thing'):>10} "
        f"{_metric_or_dash(metrics_roi_thing, 'PQ_stuff'):>10}\n"
    )
    report = (
        proto_text
        + align_text
        + "[VLSplat] all Gaussians, scene-graph group→class (unmapped=0)\n"
        + format_semantic_report("VLSplat", metrics_full)
        + (
            f"[VLSplat+ROI-thing] stuff=all group points (no bbox); "
            f"thing=group ∩ bbox_{args.roi_bbox} (outside dropped)\n"
            f"  n={int(roi_thing_mask.sum())}/{int(full_mask.sum())}\n"
            f"  kept_stuff={roi_thing_stats['n_kept_stuff']}  "
            f"kept_thing={roi_thing_stats['n_kept_thing']}\n"
            f"  removed={roi_thing_stats['n_removed']}  "
            f"unmapped(kept as unlabeled)={roi_thing_stats['n_unmapped_kept_unlabeled']}  "
            f"thing_oob={roi_thing_stats['n_removed_thing_oob']}\n"
        )
        + format_semantic_report("VLSplat+ROI-thing", metrics_roi_thing)
        + summary_text
    )
    print(report, end="")


if __name__ == "__main__":
    main()
