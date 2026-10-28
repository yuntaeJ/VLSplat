"""Scene-graph file helpers shared by every stage: the node schema, group ids, boxes, edges, and reading / writing.

A graph is ``{"nodes": [...], "edges": [...]}``. A node keeps only what refinement and evaluation read:
``id``, ``group_id`` (an int, or a list for merged nodes), ``target``, ``class_name``, and for target nodes
``bbox_tight`` and ``text_prompt``. ``edges`` link target nodes whose margin-expanded boxes overlap; they are written
for inspection only (graph from step 2 and the final graph of step 3) and never read back.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

NODE_FIELDS = ("id", "group_id", "target", "class_name", "bbox_tight", "text_prompt")
EDGE_MARGIN_FRACTION = 0.1
EDGE_MIN_OVERLAP = 1e-4


def node_group_ids(node: Dict[str, Any]) -> List[int]:
    """Codebook group ids of a node as a sorted, de-duplicated list."""
    raw = node.get("group_id")
    items = raw if isinstance(raw, list) else ([] if raw is None else [raw])
    return sorted({int(g) for g in items})


def minimal_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copies of ``nodes`` reduced to NODE_FIELDS, with ``group_id`` an int for one group and a sorted list otherwise."""
    out = []
    for n in nodes:
        node = {k: n[k] for k in NODE_FIELDS if k in n}
        gids = node_group_ids(node)
        node["group_id"] = gids[0] if len(gids) == 1 else gids
        out.append(node)
    return out


def expand_bbox(tight: Dict[str, float], fraction: float) -> Dict[str, float]:
    """``tight`` grown by ``fraction`` of its own extent on every side."""
    out = {}
    for ax in "xyz":
        lo, hi = float(tight[ax + "min"]), float(tight[ax + "max"])
        out[ax + "min"], out[ax + "max"] = lo - (hi - lo) * fraction, hi + (hi - lo) * fraction
    return out


def build_target_edges(nodes: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    """Undirected edges between target nodes whose expanded ``bbox_tight`` boxes overlap in volume."""
    boxes = [
        (int(n["id"]), expand_bbox(n["bbox_tight"], EDGE_MARGIN_FRACTION))
        for n in nodes
        if n.get("target") and isinstance(n.get("bbox_tight"), dict)
    ]
    edges = []
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1 :]:
            overlap = 1.0
            for ax in "xyz":
                overlap *= max(0.0, min(ba[ax + "max"], bb[ax + "max"]) - max(ba[ax + "min"], bb[ax + "min"]))
            if overlap >= EDGE_MIN_OVERLAP:
                edges.append({"node_1": min(a, b), "node_2": max(a, b)})
    return edges


def load_scene_graph(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_scene_graph(path: str, graph: Dict[str, Any], with_edges: bool = False) -> None:
    """Write ``graph``; ``with_edges`` recomputes ``edges`` from the current boxes, otherwise none are written."""
    doc = {"nodes": graph["nodes"]}
    if with_edges:
        doc["edges"] = build_target_edges(graph["nodes"])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
