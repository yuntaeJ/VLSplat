#!/usr/bin/env python3
"""Build a scene graph from a 3DGS+codebook checkpoint using SAM3 votes only.

Every eligible codebook group becomes a node. class_name is the SAM3 view-vote
winner among the 13 semantic classes, or "none". No VLM / text_prompt stage.
target=true iff class_name is in --target-classes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_graph_builder.build_scene_graph_nodes import (
    NONE_CLASS_NAME,
    _class_from_vote_row,
    _eligible_groups_by_high_sim_pct,
    _relabel_none_nodes,
    extra_normalized_core_groups,
)
from scene_graph_builder.build_scene_graph_nodes import compute_class_ids_np
from scene_graph_builder.finalize_scene_graph import (
    DEFAULT_TARGET_CLASSES,
    add_common_args,
    append_stuff_only_nodes,
    load_features_and_codebook,
    open_vote_session,
    parse_class_names,
    resolve_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a SAM3-only scene graph for semantic eval.")
    add_common_args(p)
    p.add_argument("--min-count", type=int, default=100)
    p.add_argument("--codebook-cosine-threshold", type=float, default=0.9)
    p.add_argument("--min-high-sim-pct", type=float, default=50.0)
    p.add_argument("--min-render-pixels", type=int, default=1000)
    p.add_argument(
        "--valid-mask-margin-pct",
        type=float,
        default=0.1,
        help="Margin for thing classes only (cabinet/bed/chair/desk/sofa/monitor). Stuff votes on the full image.",
    )
    p.add_argument(
        "--target-classes",
        nargs="+",
        default=list(DEFAULT_TARGET_CLASSES),
        help="Classes that get target=true after SAM3 assignment.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_path).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    ckpt = resolve_checkpoint(args.exp_dir, also_require=[source_path])

    target_classes = set(parse_class_names(args.target_classes))
    features, codebook_np = load_features_and_codebook(ckpt, args.num_objects)
    group_per_point = compute_class_ids_np(features, codebook_np, normalize=True).astype("int64")

    filters = dict(
        features=features,
        codebook_np=codebook_np,
        min_count=int(args.min_count),
        sim_thresh=float(args.codebook_cosine_threshold),
        min_high_sim_pct=float(args.min_high_sim_pct),
    )
    eligible_groups, _ = _eligible_groups_by_high_sim_pct(group_per_point=group_per_point, **filters)
    if not eligible_groups:
        raise RuntimeError("No eligible codebook groups after count / similarity filters")
    extra_groups = extra_normalized_core_groups(already=eligible_groups, **filters)
    if extra_groups:
        print(f"[sam3-only] extra groups for no-margin rematch: {extra_groups}")

    session = open_vote_session(args, ckpt, int(args.min_render_pixels))

    print(f"[sam3-only] exp-dir        = {ckpt.exp_dir}")
    print(f"[sam3-only] eligible groups = {len(eligible_groups)}")
    print(f"[sam3-only] label json     = {session.label_json_root}")
    print(f"[sam3-only] target classes = {sorted(target_classes)}")

    semantic_by_gid = session.vote(set(int(g) for g in eligible_groups), float(args.valid_mask_margin_pct))

    nodes: List[Dict[str, Any]] = []
    for node_id, group_id in enumerate(eligible_groups, start=1):
        class_name = _class_from_vote_row(semantic_by_gid.get(int(group_id)))
        nodes.append(
            {
                "id": int(node_id),
                "group_id": int(group_id),
                "group_ids": [int(group_id)],
                "target": bool(class_name in target_classes),
                "class_name": class_name,
            }
        )

    # None nodes (and the rescued extras) get one more pass on the full image.
    rematch_gids = sorted(
        {int(n["group_id"]) for n in nodes if str(n.get("class_name") or NONE_CLASS_NAME) == NONE_CLASS_NAME}
        | {int(g) for g in extra_groups}
    )
    if rematch_gids:
        none_by_gid = session.vote(set(rematch_gids), 0.0)
        n_none = _relabel_none_nodes(nodes, none_by_gid)
        if n_none:
            print(f"[sam3-only] none-node rematch without margin (stuff only): updated {n_none}")
        append_stuff_only_nodes(nodes, extra_groups, none_by_gid, _class_from_vote_row)

    doc = {"nodes": nodes}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    counts = Counter(str(n["class_name"]) for n in nodes)
    n_target = sum(bool(n.get("target")) for n in nodes)
    print(f"[sam3-only] nodes={len(nodes)} target={n_target}")
    print("[sam3-only] class_name counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Saved SAM3-only scene graph: {out_json}")


if __name__ == "__main__":
    main()
