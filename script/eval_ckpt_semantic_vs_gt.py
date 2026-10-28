#!/usr/bin/env python3
"""Render-level semantic eval: rendered semantic maps vs GT index PNGs (mIoU / mAcc, per-class table)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.exp_eval_common import (
    ExpEvalContext,
    TARGET_CLASSES,
    VOID_ID,
    aggregate_metric_dicts,
    build_exp_eval_context,
    canonicalize_target_label_map,
    colorize_label_map,
    compute_semantic_metrics,
    ensure_dir,
    load_label_png,
    make_diff_overlay,
    render_semantic_prediction,
    resize_label_map_nearest,
    save_label_png,
    save_rgb_png,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render 2D semantic maps from a 3DGS+codebook checkpoint and compare against GT PNGs."
    )
    p.add_argument("--exp-dir", required=True, help="Checkpoint dir with point_cloud.ply and classifier.pth")
    p.add_argument("--scene-root", required=True, help="Colmap scene root (images + sparse + train_test_lists.json)")
    p.add_argument("--gt-dir", required=True, help="Directory of GT semantic index PNGs")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--scene-graph-json", default="", help="Scene-graph JSON. Default: <exp-dir>/scene_graph.json")
    p.add_argument("--split", default="test")
    p.add_argument("--resolution", type=int, default=1)
    p.add_argument("--data-device", default="cuda")
    p.add_argument("--tile-width", type=int, default=360)
    return p.parse_args()


def _resolve_scene_graph_json(exp_dir: Path, requested: str) -> Path:
    if str(requested).strip():
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Scene graph JSON not found: {path}")
        return path
    fallback = exp_dir / "scene_graph.json"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Scene graph JSON not found: {fallback}")


def _load_gt_map(gt_dir: Path, image_name: str, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    stem = Path(str(image_name)).stem
    gt_path = gt_dir / f"{stem}.png"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing GT semantic PNG: {gt_path}")
    gt_map = canonicalize_target_label_map(load_label_png(gt_path))
    if target_shape is not None and gt_map.shape != tuple(target_shape):
        gt_map = canonicalize_target_label_map(resize_label_map_nearest(gt_map, target_shape))
    return gt_map


def _align_pred_to_gt(pred_map: np.ndarray, gt_map: np.ndarray) -> np.ndarray:
    pred_map = canonicalize_target_label_map(pred_map)
    if pred_map.shape != gt_map.shape:
        pred_map = canonicalize_target_label_map(resize_label_map_nearest(pred_map, gt_map.shape))
    return pred_map


def _compact_view_metrics(metrics: Mapping[str, Any], image_name: str) -> Dict[str, Any]:
    return {
        "image_name": image_name,
        "overall_pixel_accuracy": float(metrics["overall_pixel_accuracy"]),
        "mIoU": float(metrics["mIoU"]),
        "mAcc": float(metrics["mAcc"]),
        "mPrecision": float(metrics["mPrecision"]),
        "valid_class_names": list(metrics["valid_class_names"]),
        "confusion_matrix_gt_by_pred": metrics["confusion_matrix_gt_by_pred"],
        "confusion_pred_labels": metrics["confusion_pred_labels"],
    }


def format_metric_report(doc: Mapping[str, Any]) -> str:
    summary = doc["summary"]
    lines = [
        f"{doc['experiment']}  {doc['split']}/{doc['num_views']}   "
        f"mIoU {summary['mIoU']:.3f}   mAcc {summary['mAcc']:.3f}",
        f"{'class':<10}{'IoU':>8}{'Acc':>8}",
        f"{'-'*10}{'':->8}{'':->8}",
    ]
    for row in summary["per_class"]:
        if not row["present_in_gt"]:
            continue
        lines.append(f"{row['class_name']:<10}{row['iou']:8.3f}{row['acc']:8.3f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir).expanduser().resolve()
    scene_root = Path(args.scene_root).expanduser().resolve()
    gt_dir = Path(args.gt_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    scene_graph_json = _resolve_scene_graph_json(exp_dir, str(args.scene_graph_json))

    context: ExpEvalContext = build_exp_eval_context(
        exp_dir=exp_dir,
        scene_root=scene_root,
        split=str(args.split),
        scene_graph_json=scene_graph_json,
        object_dir=str(gt_dir),
        resolution=int(args.resolution),
        data_device=str(args.data_device),
    )
    if not context.cameras:
        raise RuntimeError(f"No cameras found for split={args.split} under {scene_root}")
    if not context.scene_graph_nodes:
        raise RuntimeError(
            f"No labeled scene-graph nodes mapped to target classes in {scene_graph_json}"
        )

    pred_dir = ensure_dir(out_dir / "pred_labels")
    pred_viz_dir = ensure_dir(out_dir / "pred_viz")
    gt_viz_dir = ensure_dir(out_dir / "gt_viz")
    rgb_dir = ensure_dir(out_dir / "rgb")
    diff_dir = ensure_dir(out_dir / "diff_overlay")

    per_view_rows: List[Dict[str, Any]] = []
    view_names: List[str] = []

    for camera in context.cameras:
        image_name = str(camera.image_name)
        render_out = render_semantic_prediction(camera, context, bg_mode="black")
        pred_raw = canonicalize_target_label_map(render_out["class_map"])
        gt_map = _load_gt_map(gt_dir, image_name)
        pred_map = _align_pred_to_gt(pred_raw, gt_map)
        rgb = np.asarray(render_out["rgb"], dtype=np.uint8)
        if rgb.shape[:2] != gt_map.shape:
            rgb = np.asarray(
                Image.fromarray(rgb, mode="RGB").resize((gt_map.shape[1], gt_map.shape[0]), resample=Image.BILINEAR),
                dtype=np.uint8,
            )

        all_metrics = compute_semantic_metrics(gt_map, pred_map)
        per_view_rows.append(_compact_view_metrics(all_metrics, image_name))

        pred_viz = colorize_label_map(pred_map)
        gt_viz = colorize_label_map(gt_map)
        diff = make_diff_overlay(rgb, gt_map, pred_map)

        save_label_png(pred_dir / f"{image_name}.png", pred_map)
        save_rgb_png(pred_viz_dir / f"{image_name}.png", pred_viz)
        save_rgb_png(gt_viz_dir / f"{image_name}.png", gt_viz)
        save_rgb_png(rgb_dir / f"{image_name}.png", rgb)
        save_rgb_png(diff_dir / f"{image_name}.png", diff)
        view_names.append(image_name)

    summary = aggregate_metric_dicts(per_view_rows)
    doc: Dict[str, Any] = {
        "experiment": context.exp_name,
        "exp_dir": str(exp_dir),
        "scene_root": str(scene_root),
        "gt_dir": str(gt_dir),
        "scene_graph_json": str(scene_graph_json),
        "split": str(args.split),
        "num_views": len(view_names),
        "view_names": view_names,
        "target_classes": TARGET_CLASSES,
        "void_id": int(VOID_ID),
        "pred_dir": str(pred_dir),
        "summary": summary,
        "per_view": per_view_rows,
    }
    print(format_metric_report(doc), end="")


if __name__ == "__main__":
    main()
