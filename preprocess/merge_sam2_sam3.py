"""
SAM3 + SAM2 mask merge: a SAM2 mask is dropped when overlap / SAM2_area > 0.3.

  python merge_sam2_sam3.py --image-dir /path/to/<scene>/images

Reads and writes under the scene directory (parent of images/):
  sam3_* , sam2_*  →  object_mask / object_mask_json / object_mask_vis
  then removes intermediate sam2_* and sam3_* folders.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from mask_utils import (
    Instance,
    find_image,
    repaint_mask,
    render_overlay,
)

SAM3_MASK_SUBDIR = "sam3_masks"
SAM3_JSON_SUBDIR = "sam3_json"
SAM3_VIS_SUBDIR = "sam3_vis"
SAM2_MASK_SUBDIR = "sam2_masks"
SAM2_JSON_SUBDIR = "sam2_json"
SAM2_VIS_SUBDIR = "sam2_vis"

# Intermediate dirs removed after a successful merge.
INTERMEDIATE_SUBDIRS = (
    SAM3_MASK_SUBDIR,
    SAM3_JSON_SUBDIR,
    SAM3_VIS_SUBDIR,
    SAM2_MASK_SUBDIR,
    SAM2_JSON_SUBDIR,
    SAM2_VIS_SUBDIR,
)

OVERLAP_THRESHOLD = 0.3
MIN_SAM2_AREA = 200


def cleanup_intermediates(scene_dir: Path) -> None:
    for name in INTERMEDIATE_SUBDIRS:
        path = scene_dir / name
        if path.is_dir():
            shutil.rmtree(path)


def normalize_label(label: str) -> str:
    """Final output: treat 'table' as 'desk'."""
    return "desk" if label.strip().lower() == "table" else label


def load_sam3_instances(mask_path: Path, json_path: Path) -> List[Instance]:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.uint32)

    with open(json_path) as f:
        label_map = json.load(f)

    instances = []
    for k, v in label_map.items():
        iid = int(k)
        region = mask == iid
        if not region.any():
            continue
        instances.append(Instance(
            region=region,
            label=v.get("label", "unknown"),
            score=float(v.get("score", 0.0)),
        ))
    return instances


def load_sam2_instances(mask_path: Path, json_path: Path) -> List[Instance]:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.uint32)

    with open(json_path) as f:
        label_map = json.load(f)

    instances = []
    for k, v in label_map.items():
        iid = int(k)
        region = mask == iid
        if not region.any():
            continue
        score = float(v.get("stability_score", v.get("score", 0.0)))
        instances.append(Instance(region=region, label="sam2", score=score))
    return instances


def filter_sam2_by_sam2_area(
    sam2_instances: List[Instance],
    sam3_occupied: np.ndarray,
) -> List[Instance]:
    kept = []
    for inst in sam2_instances:
        area = int(inst.region.sum())
        if area < MIN_SAM2_AREA:
            continue
        overlap = int(np.logical_and(inst.region, sam3_occupied).sum())
        if overlap / area <= OVERLAP_THRESHOLD:
            kept.append(inst)
    return kept


def save_result(
    stem: str,
    all_instances: List[Instance],
    mask_shape: tuple,
    mask_dir: Path,
    json_dir: Path,
    overlay_dir: Path,
    image_dir: Optional[Path],
) -> None:
    out_mask, final_instances = repaint_mask(all_instances, mask_shape)
    for inst in final_instances:
        inst.label = normalize_label(inst.label)

    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_mask.astype(np.uint16)).save(mask_dir / f"{stem}.png")

    json_dir.mkdir(parents=True, exist_ok=True)
    out_label_map = {
        str(inst.new_id): {"label": inst.label, "score": inst.score}
        for inst in final_instances if inst.new_id > 0
    }
    with open(json_dir / f"{stem}.json", "w") as f:
        json.dump(out_label_map, f)

    if image_dir is not None:
        src_path = find_image(image_dir, stem)
        if src_path:
            src_rgb = np.array(Image.open(src_path).convert("RGB"))
            overlay_img = render_overlay(src_rgb, out_mask, final_instances)
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_img.save(overlay_dir / f"{stem}.png")


def process_image(
    stem: str,
    sam3_mask_dir: Path,
    sam3_json_dir: Path,
    sam2_mask_dir: Path,
    sam2_json_dir: Path,
    image_dir: Optional[Path],
    mask_dir: Path,
    json_dir: Path,
    overlay_dir: Path,
) -> None:
    s3m = sam3_mask_dir / f"{stem}.png"
    s3j = sam3_json_dir / f"{stem}.json"
    s2m = sam2_mask_dir / f"{stem}.png"
    s2j = sam2_json_dir / f"{stem}.json"
    if not (s3m.exists() and s3j.exists() and s2m.exists() and s2j.exists()):
        return

    sam3_instances = load_sam3_instances(s3m, s3j)
    sam2_instances = load_sam2_instances(s2m, s2j)
    mask_shape = np.array(Image.open(s3m)).shape[:2]

    sam3_occupied = np.zeros(mask_shape, dtype=bool)
    for inst in sam3_instances:
        sam3_occupied |= inst.region

    sam2_kept = filter_sam2_by_sam2_area(sam2_instances, sam3_occupied)
    save_result(
        stem, sam3_instances + sam2_kept, mask_shape,
        mask_dir, json_dir, overlay_dir, image_dir,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--image-dir",
        required=True,
        help="RGB image directory (.../<scene>/images)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).resolve()
    scene_dir = image_dir.parent

    print(f"images:  {image_dir}")
    print(f"scene:   {scene_dir}")

    mask_dir = scene_dir / "object_mask"
    json_dir = scene_dir / "object_mask_json"
    overlay_dir = scene_dir / "object_mask_vis"

    sam3_mask_dir = scene_dir / SAM3_MASK_SUBDIR
    sam3_json_dir = scene_dir / SAM3_JSON_SUBDIR
    sam2_mask_dir = scene_dir / SAM2_MASK_SUBDIR
    sam2_json_dir = scene_dir / SAM2_JSON_SUBDIR

    if not sam3_mask_dir.is_dir():
        print(f"[error] missing {sam3_mask_dir}")
        return
    if not sam2_mask_dir.is_dir():
        print(f"[error] missing {sam2_mask_dir}")
        return

    stems = sorted(p.stem for p in sam3_mask_dir.glob("*.png"))
    if not stems:
        print(f"no masks under {sam3_mask_dir}")
        return
    print(f"{len(stems)} images\n")

    print("=== merge SAM3 + SAM2 ===")
    for stem in tqdm(stems, desc="merge", unit="img"):
        process_image(
            stem=stem,
            sam3_mask_dir=sam3_mask_dir,
            sam3_json_dir=sam3_json_dir,
            sam2_mask_dir=sam2_mask_dir,
            sam2_json_dir=sam2_json_dir,
            image_dir=image_dir if image_dir.is_dir() else None,
            mask_dir=mask_dir,
            json_dir=json_dir,
            overlay_dir=overlay_dir,
        )

    cleanup_intermediates(scene_dir)
    print(f"\ndone -> {mask_dir}")


if __name__ == "__main__":
    main()
