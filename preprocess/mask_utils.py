import colorsys
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BACKGROUND_LABELS: Set[str] = {"wall", "ceiling", "floor"}
OVERLAY_ALPHA = 0.30
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG"}


@dataclass
class Instance:
    region: np.ndarray
    label: str
    score: float
    new_id: int = 0


def canonical_class_key(label: str) -> str:
    return label.strip().lower()


def load_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint32)


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def extract_instances(mask: np.ndarray, label_map: Dict) -> List[Instance]:
    instances: List[Instance] = []
    for k, v in label_map.items():
        iid = int(k)
        region = mask == iid
        if not region.any():
            continue
        instances.append(
            Instance(
                region=region,
                label=v.get("label", "unknown"),
                score=float(v.get("score", 0.0)),
            )
        )
    return instances


def merge_background_per_type(instances: List[Instance]) -> List[Instance]:
    fg: List[Instance] = []
    buckets: Dict[str, List[Instance]] = {t: [] for t in BACKGROUND_LABELS}
    for inst in instances:
        key = canonical_class_key(inst.label)
        if key in buckets:
            buckets[key].append(inst)
        else:
            fg.append(inst)

    merged_bg: List[Instance] = []
    for typ in ("wall", "ceiling", "floor"):
        group = buckets[typ]
        if not group:
            continue
        if len(group) == 1:
            g0 = group[0]
            merged_bg.append(Instance(region=g0.region.copy(), label=typ, score=g0.score))
            continue
        merged_region = group[0].region.copy()
        for b in group[1:]:
            merged_region |= b.region
        merged_score = max(b.score for b in group)
        merged_bg.append(Instance(region=merged_region, label=typ, score=merged_score))
    return fg + merged_bg


def find_image(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in _IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def repaint_mask(
    instances: List[Instance], shape: Tuple[int, int]
) -> Tuple[np.ndarray, List[Instance]]:
    """
    Paint instances onto a uint16 map with ids 1..N.
    Lower score is painted first so higher score wins on overlaps.
    Regions are clipped to pixels that remain after overwrites.
    """
    h, w = shape
    sorted_insts = sorted(
        instances,
        key=lambda x: (x.score, int(x.region.sum())),
    )
    out_mask = np.zeros((h, w), dtype=np.uint16)
    for new_id, inst in enumerate(sorted_insts, start=1):
        out_mask[inst.region] = new_id
        inst.new_id = new_id

    final: List[Instance] = []
    for inst in sorted_insts:
        actual = out_mask == inst.new_id
        if actual.any():
            inst.region = actual
            final.append(inst)
    return out_mask, final


def _instance_color(inst_id: int) -> Tuple[int, int, int]:
    hue = (inst_id * 137.508) % 360
    sat = 0.7 + (inst_id % 3) * 0.1
    val = 0.8 + (inst_id % 2) * 0.2
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))


def _contrasting_text_color(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = bg
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 160 else (255, 255, 255)


def _get_font(size: int = 13) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def render_overlay(image_rgb: np.ndarray, out_mask: np.ndarray, instances: List[Instance]) -> Image.Image:
    base = image_rgb.astype(np.float32)
    overlay = base.copy()

    for inst in instances:
        region = out_mask == inst.new_id
        if not region.any():
            continue
        color = np.array(_instance_color(inst.new_id), dtype=np.float32)
        overlay[region] = OVERLAY_ALPHA * color + (1.0 - OVERLAY_ALPHA) * overlay[region]

    result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    font = _get_font(13)

    for inst in instances:
        region = out_mask == inst.new_id
        if not region.any():
            continue
        ys, xs = np.where(region)
        cx, cy = int(xs.mean()), int(ys.mean())
        score = float(np.clip(inst.score, 0.0, 1.0))
        text = f"{inst.label} {score:.2f}"
        bg_color = _instance_color(inst.new_id)
        txt_color = _contrasting_text_color(bg_color)
        bbox = draw.textbbox((cx, cy), text, font=font, anchor="mm")
        pad = 3
        draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=bg_color)
        draw.text((cx, cy), text, font=font, fill=txt_color, anchor="mm")

    return result
