#!/usr/bin/env python3
"""Shared loading and rendering helpers for the render-level eval."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render
from utils.local_runtime_utils import load_codebook as load_codebook_torch, normalize_label, quiet_scene_io
from scene import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos

TARGET_CLASSES: List[str] = [
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
NUM_CLASSES = len(TARGET_CLASSES)
VOID_ID = 255
TARGET_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
CLASS_NAME_BY_ID = {idx: name for idx, name in enumerate(TARGET_CLASSES)}
PRED_CLASS_ALIAS: Dict[str, str] = {
    "table": "desk",
}
CANONICAL_TARGET_REMAP = np.arange(NUM_CLASSES, dtype=np.uint8)
LABEL_RGB_BY_ID: Dict[int, Tuple[int, int, int]] = {
    0: (36, 72, 118),
    1: (17, 119, 51),
    2: (102, 51, 0),
    3: (146, 43, 33),
    4: (204, 102, 0),
    5: (102, 0, 153),
    6: (153, 51, 102),
    7: (128, 64, 0),
    8: (0, 102, 153),
    9: (153, 0, 0),
    10: (0, 128, 128),
    11: (102, 76, 0),
    12: (85, 26, 139),
    VOID_ID: (70, 70, 70),
}

@dataclass(frozen=True)

class SceneGraphNode:
    node_id: int
    class_name: str
    class_id: int
    group_ids: Tuple[int, ...]

@dataclass

class ExpEvalContext:
    exp_dir: Path
    exp_name: str
    scene_root: Path
    split: str
    cameras: List[Any]
    gaussians: GaussianModel
    codebook: torch.Tensor
    codebook_norm: torch.Tensor
    group_to_class_id: np.ndarray
    group_to_node_id: np.ndarray
    scene_graph_nodes: List[SceneGraphNode]

def normalize_pred_class_name(name: str) -> str:
    out = normalize_label(name).replace("_", " ")
    out = PRED_CLASS_ALIAS.get(out, out)
    return out

def target_id_from_name(name: str) -> int:
    normalized = normalize_pred_class_name(name)
    return TARGET_ID.get(normalized, -1)

def canonicalize_target_label_map(label_map: np.ndarray, *, void_id: int = VOID_ID) -> np.ndarray:
    arr = np.asarray(label_map)
    out = arr.astype(np.uint8, copy=True)
    valid = (out >= 0) & (out < NUM_CLASSES)
    if np.any(valid):
        out[valid] = CANONICAL_TARGET_REMAP[out[valid]]
    out[~valid & (out != int(void_id))] = np.uint8(void_id)
    return out

def load_label_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)

def resize_label_map_nearest(label_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    src = np.asarray(label_map, dtype=np.uint8)
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if src.shape == (target_h, target_w):
        return src.copy()
    pil = Image.fromarray(src, mode="L")
    resized = pil.resize((target_w, target_h), resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.uint8)

def parse_group_ids(node: Dict[str, Any]) -> List[int]:
    values: List[int] = []
    for key in ("group_id", "group_ids"):
        raw = node.get(key)
        if isinstance(raw, list):
            for item in raw:
                try:
                    values.append(int(item))
                except Exception:
                    continue
        elif raw is not None:
            try:
                values.append(int(raw))
            except Exception:
                pass
    canon = node.get("canonical_group_id")
    if canon is not None:
        try:
            values.append(int(canon))
        except Exception:
            pass
    return sorted(set(values))

def load_split_names(scene_root: Path, split: str) -> List[str]:
    doc = json.loads((scene_root / "train_test_lists.json").read_text(encoding="utf-8"))
    if split not in doc:
        raise KeyError(f"Split '{split}' not found in {scene_root / 'train_test_lists.json'}")
    return [Path(str(name)).stem for name in doc[split]]

def build_cameras(
    scene_root: Path,
    split: str,
    *,
    images_dir: str = "resized_undistorted_images",
    object_dir: str = "gt_semantics",
    view_names: Optional[Sequence[str]] = None,
    resolution: int = 1,
    data_device: str = "cuda",
) -> List[Any]:
    scene_info = sceneLoadTypeCallbacks["Colmap"](
        str(scene_root),
        images_dir,
        False,
        object_dir,
        n_views=100,
        random_init=False,
        train_split=False,
    )
    wanted = set(load_split_names(scene_root, split))
    if view_names is not None:
        wanted &= {Path(str(name)).stem for name in view_names}
    cam_infos = [cam for cam in scene_info.train_cameras if str(cam.image_name) in wanted]
    cam_infos.sort(key=lambda cam: str(cam.image_name))
    cam_args = SimpleNamespace(
        resolution=int(resolution),
        data_device=str(data_device),
    )
    return cameraList_from_camInfos(cam_infos, 1.0, cam_args)

def load_scene_graph(exp_dir: Path, num_groups: int) -> Tuple[List[SceneGraphNode], np.ndarray, np.ndarray]:
    path = exp_dir / "merged_scene_graph_all.json"
    if not path.is_file():
        raise FileNotFoundError(f"Scene graph JSON not found: {path}")
    return load_scene_graph_from_path(path, num_groups)

def load_scene_graph_from_path(path: Path, num_groups: int) -> Tuple[List[SceneGraphNode], np.ndarray, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Scene graph JSON not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))

    group_to_class_id = np.full((num_groups,), VOID_ID, dtype=np.uint8)
    group_to_node_id = np.full((num_groups,), -1, dtype=np.int32)
    nodes: List[SceneGraphNode] = []

    for idx, raw_node in enumerate(doc.get("nodes") or [], start=1):
        group_ids = parse_group_ids(raw_node)
        if not group_ids:
            continue
        class_id = target_id_from_name(str(raw_node.get("class_name", "")).strip())
        if class_id < 0:
            continue
        node_id = int(raw_node.get("id", idx))
        class_name = TARGET_CLASSES[class_id]
        for gid in group_ids:
            if 0 <= gid < num_groups:
                group_to_class_id[gid] = np.uint8(class_id)
                group_to_node_id[gid] = np.int32(node_id)
        nodes.append(
            SceneGraphNode(
                node_id=node_id,
                class_name=class_name,
                class_id=class_id,
                group_ids=tuple(int(gid) for gid in group_ids if 0 <= gid < num_groups),
            )
        )

    return nodes, group_to_class_id, group_to_node_id

def build_exp_eval_context(
    *,
    exp_dir: Path,
    scene_root: Path,
    split: str,
    view_names: Optional[Sequence[str]] = None,
    scene_graph_json: Optional[Path] = None,
    object_dir: str = "gt_semantics",
    resolution: int = 1,
    data_device: str = "cuda",
    images_dir: str = "resized_undistorted_images",
) -> ExpEvalContext:
    exp_dir = exp_dir.expanduser().resolve()
    scene_root = scene_root.expanduser().resolve()
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    gaussians = GaussianModel(3)
    gaussians.load_ply(str(exp_dir / "point_cloud.ply"))
    codebook = load_codebook_torch(str(exp_dir / "classifier.pth"), device=data_device)
    codebook = codebook.detach().clone()
    codebook_norm = F.normalize(codebook, dim=1)
    if scene_graph_json is None:
        scene_graph_nodes, group_to_class_id, group_to_node_id = load_scene_graph(exp_dir, int(codebook.shape[0]))
    else:
        scene_graph_nodes, group_to_class_id, group_to_node_id = load_scene_graph_from_path(
            Path(scene_graph_json),
            int(codebook.shape[0]),
        )
    with quiet_scene_io():
        cameras = build_cameras(
            scene_root=scene_root,
            split=split,
            images_dir=images_dir,
            object_dir=str(object_dir),
            view_names=view_names,
            resolution=resolution,
            data_device=data_device,
        )
    return ExpEvalContext(
        exp_dir=exp_dir,
        exp_name=exp_dir.name,
        scene_root=scene_root,
        split=split,
        cameras=cameras,
        gaussians=gaussians,
        codebook=codebook,
        codebook_norm=codebook_norm,
        group_to_class_id=group_to_class_id,
        group_to_node_id=group_to_node_id,
        scene_graph_nodes=scene_graph_nodes,
    )

def render_semantic_prediction(
    camera: Any,
    context: ExpEvalContext,
    *,
    bg_mode: str = "black",
) -> Dict[str, Any]:
    if bg_mode == "white":
        bg = torch.ones(3, device=context.gaussians.get_xyz.device)
    else:
        bg = torch.zeros(3, device=context.gaussians.get_xyz.device)
    pipe = SimpleNamespace(
        debug=False,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    )

    with torch.no_grad():
        pkg = render(camera, context.gaussians, pipe, bg)
        rgb_t = pkg["render"].detach().clamp(0.0, 1.0)
        obj_t = pkg["render_object"].detach()

    cdim, height, width = int(obj_t.shape[0]), int(obj_t.shape[1]), int(obj_t.shape[2])
    flat_obj = obj_t.view(cdim, -1).transpose(0, 1)
    obj_norm = torch.linalg.norm(flat_obj, dim=1)
    valid = obj_norm > 1e-8
    pred_group = torch.full((flat_obj.shape[0],), -1, dtype=torch.int64, device=flat_obj.device)
    if int(valid.sum().item()) > 0:
        flat_obj_valid = F.normalize(flat_obj[valid], dim=1)
        logits = flat_obj_valid @ context.codebook_norm.T
        pred_group[valid] = torch.argmax(logits, dim=1)

    pred_group_np = pred_group.view(height, width).cpu().numpy().astype(np.int32)
    valid_np = valid.view(height, width).cpu().numpy()
    rgb_np = (rgb_t.permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)

    class_map = np.full((height, width), VOID_ID, dtype=np.uint8)
    node_map = np.full((height, width), -1, dtype=np.int32)
    valid_group_mask = valid_np & (pred_group_np >= 0) & (pred_group_np < context.group_to_class_id.shape[0])
    if np.any(valid_group_mask):
        class_map[valid_group_mask] = context.group_to_class_id[pred_group_np[valid_group_mask]]
        node_map[valid_group_mask] = context.group_to_node_id[pred_group_np[valid_group_mask]]
    class_map[~valid_group_mask] = VOID_ID
    node_map[~valid_group_mask] = -1
    visible_node_ids = sorted(int(v) for v in np.unique(node_map[node_map >= 0]).tolist())

    return {
        "rgb": rgb_np,
        "pred_group": pred_group_np,
        "class_map": class_map,
        "node_map": node_map,
        "visible_node_ids": visible_node_ids,
    }

def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    arr = np.asarray(label_map)
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    for class_id, color in LABEL_RGB_BY_ID.items():
        rgb[arr == class_id] = color
    return rgb

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_label_png(path: Path, label_map: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(label_map.astype(np.uint8), mode="L").save(path)

def save_rgb_png(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)

def compute_semantic_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    void_id: int = VOID_ID,
    class_subset_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    gt_arr = canonicalize_target_label_map(np.asarray(gt, dtype=np.uint8), void_id=void_id).astype(np.int32)
    pred_arr = canonicalize_target_label_map(np.asarray(pred, dtype=np.uint8), void_id=void_id).astype(np.int32)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Shape mismatch: gt {gt_arr.shape} vs pred {pred_arr.shape}")

    valid = gt_arr != int(void_id)
    subset_ids: Optional[np.ndarray] = None
    if class_subset_ids is not None:
        subset_ids = np.asarray(sorted({int(class_id) for class_id in class_subset_ids}), dtype=np.int32)
        if subset_ids.size > 0:
            valid &= np.isin(gt_arr, subset_ids)
    gt_valid = gt_arr[valid]
    pred_valid = pred_arr[valid]

    pred_bucket = np.full(pred_valid.shape, NUM_CLASSES, dtype=np.int32)
    pred_known = (pred_valid >= 0) & (pred_valid < NUM_CLASSES)
    if subset_ids is None:
        pred_bucket[pred_known] = pred_valid[pred_known]
    else:
        pred_allowed = pred_known & np.isin(pred_valid, subset_ids)
        pred_bucket[pred_allowed] = pred_valid[pred_allowed]

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES + 1), dtype=np.int64)
    for gt_id in range(NUM_CLASSES):
        mask = gt_valid == gt_id
        if not np.any(mask):
            continue
        bincount = np.bincount(pred_bucket[mask], minlength=NUM_CLASSES + 1)
        confusion[gt_id] = bincount[: NUM_CLASSES + 1]

    per_class: List[Dict[str, Any]] = []
    gt_counts = confusion.sum(axis=1)
    pred_counts = confusion[:, :NUM_CLASSES].sum(axis=0)
    for class_id in range(NUM_CLASSES):
        tp = int(confusion[class_id, class_id])
        gt_count = int(gt_counts[class_id])
        pred_count = int(pred_counts[class_id])
        union = gt_count + pred_count - tp
        iou = float(tp / union) if union > 0 else 0.0
        acc = float(tp / gt_count) if gt_count > 0 else 0.0
        precision = float(tp / pred_count) if pred_count > 0 else 0.0
        per_class.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAME_BY_ID[class_id],
                "present_in_gt": bool(gt_count > 0),
                "gt_count": gt_count,
                "pred_count": pred_count,
                "tp": tp,
                "union": int(union),
                "iou": iou,
                "acc": acc,
                "precision": precision,
            }
        )

    valid_class_ids = [row["class_id"] for row in per_class if row["present_in_gt"]]
    miou = float(np.mean([per_class[idx]["iou"] for idx in valid_class_ids])) if valid_class_ids else 0.0
    macc = float(np.mean([per_class[idx]["acc"] for idx in valid_class_ids])) if valid_class_ids else 0.0
    mprecision = (
        float(np.mean([per_class[idx]["precision"] for idx in valid_class_ids])) if valid_class_ids else 0.0
    )
    total_valid = int(valid.sum())
    overall_acc = float(np.mean(gt_valid == pred_valid)) if total_valid > 0 else 0.0

    return {
        "num_pixels_total": int(gt_arr.size),
        "num_pixels_valid_gt": total_valid,
        "num_pixels_gt_void": int((gt_arr == int(void_id)).sum()),
        "num_pixels_pred_void_on_valid_gt": int((pred_bucket == NUM_CLASSES).sum()),
        "overall_pixel_accuracy": overall_acc,
        "mIoU": miou,
        "mAcc": macc,
        "mPrecision": mprecision,
        "metric_scope_class_ids": None if subset_ids is None else subset_ids.tolist(),
        "valid_class_ids": valid_class_ids,
        "valid_class_names": [CLASS_NAME_BY_ID[idx] for idx in valid_class_ids],
        "per_class": per_class,
        "confusion_matrix_gt_by_pred": confusion.tolist(),
        "confusion_pred_labels": list(range(NUM_CLASSES)) + [int(void_id)],
    }

def aggregate_metric_dicts(metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {
            "overall_pixel_accuracy": 0.0,
            "mIoU": 0.0,
            "mAcc": 0.0,
            "mPrecision": 0.0,
            "per_class": [],
            "valid_class_ids": [],
            "valid_class_names": [],
            "confusion_matrix_gt_by_pred": np.zeros((NUM_CLASSES, NUM_CLASSES + 1), dtype=np.int64).tolist(),
            "confusion_pred_labels": list(range(NUM_CLASSES)) + [int(VOID_ID)],
            "num_views": 0,
        }

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES + 1), dtype=np.int64)
    for metric in metrics:
        confusion += np.asarray(metric["confusion_matrix_gt_by_pred"], dtype=np.int64)

    total_valid = int(confusion.sum())
    gt_counts = confusion.sum(axis=1)
    pred_counts = confusion[:, :NUM_CLASSES].sum(axis=0)
    per_class: List[Dict[str, Any]] = []
    valid_class_ids: List[int] = []
    for class_id in range(NUM_CLASSES):
        tp = int(confusion[class_id, class_id])
        gt_count = int(gt_counts[class_id])
        pred_count = int(pred_counts[class_id])
        union = gt_count + pred_count - tp
        iou = float(tp / union) if union > 0 else 0.0
        acc = float(tp / gt_count) if gt_count > 0 else 0.0
        precision = float(tp / pred_count) if pred_count > 0 else 0.0
        if gt_count > 0:
            valid_class_ids.append(class_id)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAME_BY_ID[class_id],
                "present_in_gt": bool(gt_count > 0),
                "gt_count": gt_count,
                "pred_count": pred_count,
                "tp": tp,
                "union": int(union),
                "iou": iou,
                "acc": acc,
                "precision": precision,
            }
        )

    miou = float(np.mean([per_class[idx]["iou"] for idx in valid_class_ids])) if valid_class_ids else 0.0
    macc = float(np.mean([per_class[idx]["acc"] for idx in valid_class_ids])) if valid_class_ids else 0.0
    mprecision = (
        float(np.mean([per_class[idx]["precision"] for idx in valid_class_ids])) if valid_class_ids else 0.0
    )
    overall_acc = float(np.trace(confusion[:, :NUM_CLASSES]) / max(1, total_valid))

    return {
        "overall_pixel_accuracy": overall_acc,
        "mIoU": miou,
        "mAcc": macc,
        "mPrecision": mprecision,
        "per_class": per_class,
        "valid_class_ids": valid_class_ids,
        "valid_class_names": [CLASS_NAME_BY_ID[idx] for idx in valid_class_ids],
        "confusion_matrix_gt_by_pred": confusion.tolist(),
        "confusion_pred_labels": list(range(NUM_CLASSES)) + [int(VOID_ID)],
        "num_views": len(metrics),
        "num_pixels_valid_gt": total_valid,
    }

def make_diff_overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = np.asarray(rgb, dtype=np.float32).copy()
    gt_valid = gt != VOID_ID
    correct = gt_valid & (gt == pred)
    wrong = gt_valid & (gt != pred)
    void_pred = gt_valid & (pred == VOID_ID)
    base[correct] = 0.65 * base[correct] + 0.35 * np.array([0.0, 255.0, 0.0], dtype=np.float32)
    base[wrong] = 0.55 * base[wrong] + 0.45 * np.array([255.0, 0.0, 0.0], dtype=np.float32)
    base[void_pred] = 0.55 * base[void_pred] + 0.45 * np.array([255.0, 255.0, 0.0], dtype=np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)

