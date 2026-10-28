"""
SAM3 postprocess helpers used by generate_object_masks_sam3.py.

Pipeline (in-memory):
  extract_instances → merge_background_per_type → dedup_cross_class_bbox
  → paint_instance_mask_uint16 → min-area filter → (re-paint if needed)
"""

from typing import List, Set, Tuple

import numpy as np

from mask_utils import Instance, canonical_class_key

# SAM3 masks with an area below this many pixels are removed.
MIN_AREA = 500
DESK_TABLE_MASK_IOU_THRESH = 0.5
MASK_OVERLAP_FRAC_THRESH = 0.5
DESK_TABLE_LABELS = frozenset({"desk", "table"})

MASK_OUT_SUBDIR = "sam3_masks"
OVERLAY_OUT_SUBDIR = "sam3_vis"
JSON_OUT_SUBDIR = "sam3_json"

BACKGROUND_LABELS = {"wall", "ceiling", "floor"}


def paint_instance_mask_uint16(
    instances: List[Instance], shape: Tuple[int, int]
) -> Tuple[np.ndarray, List[Instance]]:
    """Paint instances 1..N; higher score overwrites lower on overlap."""
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


def _mask_iou(region_a: np.ndarray, region_b: np.ndarray) -> float:
    inter = int(np.logical_and(region_a, region_b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(region_a, region_b).sum())
    return float(inter) / float(union) if union > 0 else 0.0


def _is_desk_table_cross_pair(label_i: str, label_j: str) -> bool:
    ki = canonical_class_key(label_i)
    kj = canonical_class_key(label_j)
    return ki in DESK_TABLE_LABELS and kj in DESK_TABLE_LABELS and ki != kj


def _dedup_desk_table_mask_iou(
    instances: List[Instance],
    iou_thresh: float,
) -> Tuple[List[Instance], int]:
    n = len(instances)
    drop: Set[int] = set()

    for i in range(n):
        if i in drop:
            continue
        if canonical_class_key(instances[i].label) in BACKGROUND_LABELS:
            continue
        for j in range(i + 1, n):
            if j in drop:
                continue
            if canonical_class_key(instances[j].label) in BACKGROUND_LABELS:
                continue
            if not _is_desk_table_cross_pair(instances[i].label, instances[j].label):
                continue

            iou = _mask_iou(instances[i].region, instances[j].region)
            if iou < iou_thresh:
                continue

            si, sj = instances[i].score, instances[j].score
            if si > sj:
                drop.add(j)
            elif sj > si:
                drop.add(i)
            else:
                ai = int(instances[i].region.sum())
                aj = int(instances[j].region.sum())
                drop.add(j if ai >= aj else i)

    kept = [inst for idx, inst in enumerate(instances) if idx not in drop]
    return kept, n - len(kept)


def _dedup_cross_class_mask_overlap_fraction(
    instances: List[Instance],
    frac_thresh: float = MASK_OVERLAP_FRAC_THRESH,
    max_passes: int = 32,
) -> List[Instance]:
    working = [
        Instance(region=inst.region.copy(), label=inst.label, score=inst.score)
        for inst in instances
    ]
    n = len(working)
    if n < 2:
        return working

    changed = True
    passes = 0
    while changed and passes < max_passes:
        changed = False
        passes += 1
        for i in range(n):
            for j in range(i + 1, n):
                if not working[i].region.any() or not working[j].region.any():
                    continue
                ki = canonical_class_key(working[i].label)
                kj = canonical_class_key(working[j].label)
                if ki == kj:
                    continue
                if ki in BACKGROUND_LABELS or kj in BACKGROUND_LABELS:
                    continue

                inter = np.logical_and(working[i].region, working[j].region)
                if not inter.any():
                    continue

                inter_s = int(inter.sum())
                ai = int(working[i].region.sum())
                aj = int(working[j].region.sum())
                r_on_j = inter_s / aj if aj > 0 else 0.0
                r_on_i = inter_s / ai if ai > 0 else 0.0

                si, sj = working[i].score, working[j].score
                remove_i = False
                remove_j = False
                if r_on_j > frac_thresh and r_on_i > frac_thresh:
                    if si > sj:
                        remove_j = True
                    elif sj > si:
                        remove_i = True
                    else:
                        remove_j = ai >= aj
                        remove_i = not remove_j
                elif r_on_j > frac_thresh:
                    remove_i = True
                elif r_on_i > frac_thresh:
                    remove_j = True

                if remove_i:
                    new_r = np.logical_and(working[i].region, np.logical_not(inter))
                    if not np.array_equal(new_r, working[i].region):
                        working[i].region = new_r
                        changed = True
                if remove_j:
                    new_r = np.logical_and(working[j].region, np.logical_not(inter))
                    if not np.array_equal(new_r, working[j].region):
                        working[j].region = new_r
                        changed = True

    return [w for w in working if w.region.any()]


def dedup_cross_class_bbox(
    instances: List[Instance],
    desk_table_mask_iou_thresh: float = DESK_TABLE_MASK_IOU_THRESH,
    frac_thresh: float = MASK_OVERLAP_FRAC_THRESH,
) -> List[Instance]:
    if not instances:
        return instances
    after_desk, _ = _dedup_desk_table_mask_iou(instances, desk_table_mask_iou_thresh)
    return _dedup_cross_class_mask_overlap_fraction(after_desk, frac_thresh=frac_thresh)
