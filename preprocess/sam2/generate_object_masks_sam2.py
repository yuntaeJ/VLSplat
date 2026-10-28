"""
SAM2 automatic mask generation script

Uses SAM2 (Segment Anything Model 2) AutomaticMaskGenerator to detect
objects in an image and produce per-instance masks. Operates without
text prompts and assigns a unique instance ID (1..N) to each detected region.

Usage:
    python generate_object_masks_sam2.py --image-dir <scene>/images

Output folders (next to the scene `images/` directory):
    sam2_masks/   - uint16 PNG (0=background, 1..N=instance ID)
    sam2_json/    - JSON { instance_id: {score, stability_score} }

Defaults favor speed (points=16, crop=0, max_long_side=1024).
"""

import json
import os
import random
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
# Local SAM2 checkout.
_SAM2_SRC = _SCRIPT_DIR / "sam2"
if _SAM2_SRC.is_dir() and str(_SAM2_SRC) not in sys.path:
    sys.path.insert(0, str(_SAM2_SRC))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2_hiera_l_builder import build_sam2_hiera_large

# Architecture: sam2_hiera_l_builder.py | weights: sam2/weights/
SAM2_CKPT = str(_SCRIPT_DIR / "weights" / "sam2.1_hiera_large.pt")

MASK_SUBDIR = "sam2_masks"
JSON_SUBDIR = "sam2_json"


# ─────────────────────────────────────────────────────────────────────────────
# NMS — duplicate mask removal
# ─────────────────────────────────────────────────────────────────────────────

def _mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2):
    """
    Mask-level NMS.
    - iou_thr   : if IoU between two masks exceeds this, remove lower score
    - score_thr : remove masks with score below this threshold
    - inner_thr : threshold for removing masks that are mostly inside another
    """
    scores, idx = scores.sort(0, descending=True)
    num_masks   = idx.shape[0]
    masks_ord   = masks[idx.view(-1), :]
    masks_area  = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix       = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)

    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union        = torch.sum(torch.logical_or(masks_ord[i],  masks_ord[j]), dtype=torch.float)
            iou_matrix[i, j] = intersection / union
            # case: mask i is mostly contained in mask j
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou_matrix[i, j] = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
            # case: mask j is mostly contained in mask i
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou_matrix[j, i] = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])

    iou_matrix.triu_(diagonal=1)
    iou_max,  _ = iou_matrix.max(dim=0)
    iu_max_u, _ = torch.triu(inner_iou_matrix, diagonal=1).max(dim=0)
    iu_max_l, _ = torch.tril(inner_iou_matrix, diagonal=1).max(dim=0)

    keep         = iou_max <= iou_thr
    keep_conf    = scores > score_thr
    keep_inner_u = iu_max_u <= 1 - inner_thr
    keep_inner_l = iu_max_l <= 1 - inner_thr

    # avoid extreme case where all masks are filtered out: always keep top 3
    for arr in (keep_conf, keep_inner_u, keep_inner_l):
        if arr.sum() == 0:
            arr[scores.topk(3).indices, 0] = True

    keep = keep * keep_conf * keep_inner_u * keep_inner_l
    return idx[keep]


def _filter(keep: torch.Tensor, masks_result):
    keep_set = set(keep.int().cpu().numpy().tolist())
    return [m for i, m in enumerate(masks_result) if i in keep_set]


def masks_update(masks, **kwargs):
    """Apply NMS to SAM2 `generate()` result list to remove duplicate masks."""
    seg  = torch.from_numpy(np.stack([m["segmentation"]    for m in masks], axis=0))
    iou  = torch.from_numpy(np.stack([m["predicted_iou"]   for m in masks], axis=0))
    stab = torch.from_numpy(np.stack([m["stability_score"] for m in masks], axis=0))
    keep = _mask_nms(seg, stab * iou, **kwargs)
    return _filter(keep, masks)


# ─────────────────────────────────────────────────────────────────────────────
# Instance mask generation
# ─────────────────────────────────────────────────────────────────────────────

def masks_to_inst_map(masks, orig_shape: Tuple[int, int]) -> np.ndarray:
    """Convert mask list to uint16 instance map (0=background, 1..N=instance ID)."""
    seg_map = np.zeros(orig_shape, dtype=np.uint16)
    for i, m in enumerate(masks):
        seg = m["segmentation"]
        # if prediction was on a resized image, restore to original resolution
        if seg.shape != orig_shape:
            seg = cv2.resize(seg.astype(np.uint8), (orig_shape[1], orig_shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        seg_map[seg] = i + 1
    return seg_map


# ─────────────────────────────────────────────────────────────────────────────
# Single image processing
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    image_path: Path,
    mask_generator: SAM2AutomaticMaskGenerator,
    mask_dir: Path,
    json_dir: Path,
    max_long_side: int,
    extra_nms: bool,
) -> None:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        tqdm.write(f"[warn] failed to load: {image_path}")
        return

    orig_h, orig_w = bgr.shape[:2]
    long_side = max(orig_h, orig_w)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        proc_bgr = cv2.resize(
            bgr, (int(round(orig_w * scale)), int(round(orig_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        proc_bgr = bgr

    image_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
    proc_h, proc_w = image_rgb.shape[:2]

    with torch.inference_mode():
        masks_raw = mask_generator.generate(image_rgb)

    if len(masks_raw) == 0:
        masks = []
    elif extra_nms:
        # Slow O(n^2) custom NMS — optional; AMG already applies box NMS.
        masks = masks_update(masks_raw, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    else:
        masks = masks_raw

    # Build instance map at inference resolution, then one nearest upsample.
    inst_map = masks_to_inst_map(masks, (proc_h, proc_w))
    if (proc_h, proc_w) != (orig_h, orig_w):
        inst_map = cv2.resize(
            inst_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.uint16)

    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(inst_map).save(mask_dir / f"{image_path.stem}.png")

    confidence_map: Dict[int, Dict] = {}
    for i, m in enumerate(masks):
        confidence_map[i + 1] = {
            "score": round(float(np.clip(m.get("predicted_iou", 0.0), 0.0, 1.0)), 6),
            "stability_score": round(
                float(np.clip(m.get("stability_score", 0.0), 0.0, 1.0)), 6
            ),
        }
    json_dir.mkdir(parents=True, exist_ok=True)
    with open(json_dir / f"{image_path.stem}.json", "w") as f:
        json.dump(confidence_map, f)


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

def get_image_paths(root: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG"}
    return sorted(p for p in Path(root).rglob("*") if p.suffix in exts and p.is_file())


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # deterministic=True slows cudnn a lot; keep off for throughput
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAM2 automatic masks.")
    parser.add_argument(
        "--image-dir",
        required=True,
        help="RGB image directory (.../<scene>/images)",
    )
    parser.add_argument(
        "--sam2-ckpt",
        default=SAM2_CKPT,
        help="Path to SAM2.1 Hiera-L checkpoint .pt (architecture is in code).",
    )
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda or cpu.")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip images whose mask PNG and metadata JSON already exist.",
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=16,
        help="Grid density (default 16 → 256 points).",
    )
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=128,
        help="Points decoded per GPU batch (higher = faster if VRAM allows).",
    )
    parser.add_argument(
        "--crop-n-layers",
        type=int,
        default=0,
        help="Extra crop passes (1 was ~2-3x slower). Keep 0 for speed.",
    )
    parser.add_argument(
        "--max-long-side",
        type=int,
        default=1024,
        help="Downscale so max(H,W) <= this before AMG (default 1024).",
    )
    parser.add_argument(
        "--extra-nms",
        action="store_true",
        help="Run slow custom O(n^2) NMS on top of AMG NMS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(42)
    torch.set_default_dtype(torch.float32)
    # Optional CUDA extension missing — harmless; keep tqdm clean.
    warnings.filterwarnings(
        "ignore",
        message="cannot import name '_C'",
        category=UserWarning,
    )

    image_dir = Path(args.image_dir).resolve()
    scene_dir = image_dir.parent

    print(f"images:  {image_dir}")
    print(f"scene:   {scene_dir}")

    image_paths = get_image_paths(str(image_dir))
    if not image_paths:
        print(f"no images under {image_dir}")
        return
    print(f"{len(image_paths)} images\n")

    ckpt_path = Path(args.sam2_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint missing: {ckpt_path}")
    sam2_model = build_sam2_hiera_large(str(ckpt_path), device=args.device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    mask_dir = scene_dir / MASK_SUBDIR
    json_dir = scene_dir / JSON_SUBDIR

    print("=== SAM2 inference ===")
    for img_path in tqdm(image_paths, desc="SAM2", unit="img"):
        if (args.skip_existing and (mask_dir / f"{img_path.stem}.png").is_file()
                and (json_dir / f"{img_path.stem}.json").is_file()):
            continue
        process_image(
            image_path=img_path,
            mask_generator=mask_generator,
            mask_dir=mask_dir,
            json_dir=json_dir,
            max_long_side=args.max_long_side,
            extra_nms=args.extra_nms,
        )

    print(f"\ndone -> {mask_dir}")


if __name__ == "__main__":
    main()
