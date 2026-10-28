#!/usr/bin/env python3
"""The last step of training: finalize the scene graph after the final iteration (``finalize_scene_graph``).

  1. groups that no node owns are attached to the node with the most similar codebook entry;
  2. SAM3 view-voting re-assigns the class of every node that is not a VLM target (``assign_node_classes``);
  3. every target node gets a ``bbox_tight`` and a ``text_prompt``;
  4. the graph is saved with its edges.

Target nodes (target=true, labelled by the VLM stage) are never rewritten by the vote. Every other node gets one of the
13 semantic classes (or "none") from the SAM3 votes of its best member group; a vote for a target class promotes the node
(class_name + target=true) without calling the VLM. Nodes left at "none" and codebook groups that no node owns get one
more vote on the full image (no margin) and can only become stuff classes.

The SAM3 voting helpers (VoteSession, open_vote_session, ...) are also used by prepare_scene_graph_for_unified_lift.py.
The voting itself is `_compute_semantic_votes` from build_scene_graph_nodes.py, the same function the structure stage uses.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_graph_builder.build_scene_graph_nodes import (
    CLASS_NAME_ALIAS,
    NONE_CLASS_NAME,
    STUFF_SEMANTIC_LABEL_SET,
    _build_scene,
    _class_from_vote_row,
    _compute_semantic_votes,
    _discover_labels,
    _parse_class_list,
    _relabel_none_nodes,
    _resolve_label_json_root,
    best_vote_row,
    extra_normalized_core_groups,
    load_codebook,
    read_object_features_from_ply,
)
from scene_graph_builder.graph_io import load_scene_graph, minimal_nodes, node_group_ids, save_scene_graph
from scene_graph_builder.refine_scene_graph import refresh_target_node_bbox_tight
from utils.local_runtime_utils import quiet_scene_io

DEFAULT_TARGET_CLASSES = ("cabinet", "bed", "chair", "desk", "sofa", "monitor")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The flags the graph-building scripts take, with identical meaning."""
    parser.add_argument("--exp-dir", required=True, help="Checkpoint dir with point_cloud.ply and classifier.pth")
    parser.add_argument("--source-path", required=True, help="Colmap / dataset scene root")
    parser.add_argument("--object-path", default="", help="SAM mask dir. Default: <source-path>/object_mask")
    parser.add_argument(
        "--label-json-root", default="", help="SAM3 label JSON dir. Default: <source-path>/object_mask_json"
    )
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--num-objects", type=int, default=16)
    parser.add_argument("--out-json", required=True)
    return parser


def parse_class_names(values: Sequence[Any], *, strict: bool = True) -> List[str]:
    """Split/normalize class names.

    strict=True rejects anything outside the 13 semantic labels (used for --target-classes on the command line);
    strict=False passes unknown names through.
    """
    if strict:
        return _parse_class_list(values)
    out: List[str] = []
    seen: Set[str] = set()
    for v in values:
        for part in str(v).replace(",", " ").split():
            lab = str(part).strip().lower()
            lab = CLASS_NAME_ALIAS.get(lab, lab)
            if not lab or lab in seen:
                continue
            seen.add(lab)
            out.append(lab)
    return out


@dataclass
class Checkpoint:
    exp_dir: Path
    ply_path: Path
    classifier_path: Path
    model_root: Path
    iteration: int


def resolve_checkpoint(exp_dir: Any, *, also_require: Sequence[Any] = ()) -> Checkpoint:
    """Locate point_cloud.ply / classifier.pth and the iteration they belong to."""
    exp_dir = Path(exp_dir).expanduser().resolve()
    ply_path = exp_dir / "point_cloud.ply"
    classifier_path = exp_dir / "classifier.pth"
    for path in (exp_dir, ply_path, classifier_path, *(Path(p).expanduser().resolve() for p in also_require)):
        if not (path.is_file() or path.is_dir()):
            raise FileNotFoundError(f"missing: {path}")

    name = exp_dir.name
    is_iter_dir = name.startswith("iteration_")
    iteration = 0
    if is_iter_dir:
        try:
            iteration = int(name.split("_", 1)[1])
        except ValueError:
            iteration = 0
    return Checkpoint(
        exp_dir=exp_dir,
        ply_path=ply_path,
        classifier_path=classifier_path,
        model_root=exp_dir.parent.parent if is_iter_dir else exp_dir,
        iteration=iteration,
    )


def load_features_and_codebook(ckpt: Checkpoint, num_objects: int) -> Tuple[np.ndarray, np.ndarray]:
    features = read_object_features_from_ply(str(ckpt.ply_path), int(num_objects))
    codebook = load_codebook(str(ckpt.classifier_path), int(num_objects))
    if hasattr(codebook, "detach"):
        codebook = codebook.detach().cpu().numpy()
    return features, np.asarray(codebook)


class VoteSession:
    """A loaded scene that can be voted against SAM3 labels more than once."""

    def __init__(self, *, cameras, gaussians, codebook_t, pipeline, label_json_root: str, target_labels, min_render_pixels: int):
        self.cameras = cameras
        self.gaussians = gaussians
        self.codebook_t = codebook_t
        self.pipeline = pipeline
        self.label_json_root = label_json_root
        self.target_labels = target_labels
        self.min_render_pixels = int(min_render_pixels)

    def vote(self, gid_set: Set[int], margin_pct: float) -> Dict[int, Dict[str, Any]]:
        """Vote `gid_set` at this margin; returns rows keyed by group id."""
        if not gid_set:
            return {}
        rows, _ = _compute_semantic_votes(
            cameras=self.cameras,
            gaussians=self.gaussians,
            codebook_t=self.codebook_t,
            pipeline=self.pipeline,
            label_json_root=self.label_json_root,
            valid_gid_set=set(int(g) for g in gid_set),
            target_labels=self.target_labels,
            min_sam_pixels=self.min_render_pixels,
            valid_mask_margin_pct=float(margin_pct),
            min_render_pixels=self.min_render_pixels,
        )
        return {int(r["group_id"]): r for r in rows}


def open_vote_session(args: argparse.Namespace, ckpt: Checkpoint, min_render_pixels: int) -> VoteSession:
    """Resolve the mask/label dirs and load the scene once."""
    source_path = Path(args.source_path).expanduser().resolve()
    object_path = str(args.object_path).strip() or str(source_path / "object_mask")
    label_json_root = _resolve_label_json_root(
        str(source_path), object_path, label_json_root=str(args.label_json_root or "")
    )
    with quiet_scene_io():
        gaussians, codebook_t, cameras, pipeline = _build_scene(
            model_root=ckpt.model_root,
            source_path=str(source_path),
            object_path=object_path,
            render_point_cloud_path=ckpt.ply_path,
            render_classifier_path=ckpt.classifier_path,
            images=str(args.images),
            resolution=int(args.resolution),
            num_objects=int(args.num_objects),
        )
    return VoteSession(
        cameras=cameras,
        gaussians=gaussians,
        codebook_t=codebook_t,
        pipeline=pipeline,
        label_json_root=label_json_root,
        target_labels=_discover_labels(label_json_root),
        min_render_pixels=int(min_render_pixels),
    )


def append_stuff_only_nodes(
    nodes: List[Dict[str, Any]],
    gids: Sequence[int],
    rows_by_gid: Dict[int, Dict[str, Any]],
    class_of,
) -> int:
    """Add rescued groups as new non-target nodes, keeping stuff classes only."""
    have = {gid for n in nodes for gid in node_group_ids(n)}
    next_id = 1 + max((int(n.get("id", 0)) for n in nodes), default=0)
    n_added = 0
    for gid in gids:
        gid = int(gid)
        if gid in have:
            continue
        class_name = class_of(rows_by_gid.get(gid))
        if class_name not in STUFF_SEMANTIC_LABEL_SET:
            class_name = NONE_CLASS_NAME
        nodes.append({"id": int(next_id), "group_id": gid, "target": False, "class_name": class_name})
        have.add(gid)
        next_id += 1
        n_added += 1
    return n_added


def assign_node_classes(graph: Dict[str, Any], iteration_dir, *, source_path, object_path, label_json_root, images, resolution, cfg: Dict[str, Any]) -> str:
    """Update ``graph["nodes"]`` in place from the SAM3 votes on the checkpoint in ``iteration_dir``; returns a summary line.

    ``cfg`` is the GRAPH section of the dataset config (target_classes, structure_min_render_pixels, valid_mask_margin_pct,
    min_high_sim_count, codebook_cosine_threshold)."""
    ckpt = resolve_checkpoint(iteration_dir)
    target_classes = set(parse_class_names(cfg["target_classes"], strict=False))
    min_render = int(cfg["structure_min_render_pixels"])
    margin_pct = float(cfg["valid_mask_margin_pct"])

    nodes: List[Dict[str, Any]] = graph["nodes"]
    frozen = [n for n in nodes if n.get("target")]
    frozen_gids = {g for n in frozen for g in node_group_ids(n)}
    existing_gids = {g for n in nodes for g in node_group_ids(n)}
    voted = [n for n in nodes if not n.get("target")]
    vote_gids = {g for n in voted for g in node_group_ids(n) if g not in frozen_gids}

    features, codebook_np = load_features_and_codebook(ckpt, 16)
    rescued = extra_normalized_core_groups(
        features=features,
        codebook_np=codebook_np,
        already=existing_gids | frozen_gids,
        min_count=int(cfg["min_high_sim_count"]),
        sim_thresh=float(cfg["codebook_cosine_threshold"]),
        min_high_sim_pct=50.0,
    )

    args = SimpleNamespace(
        source_path=str(source_path), object_path=str(object_path), label_json_root=str(label_json_root),
        images=str(images), resolution=int(resolution), num_objects=16,
    )
    session = open_vote_session(args, ckpt, min_render)
    by_gid = session.vote(vote_gids, margin_pct)

    before = {id(n): str(n.get("class_name") or NONE_CLASS_NAME) for n in voted}
    promoted = 0
    for node in voted:
        new_name = _class_from_vote_row(best_vote_row(by_gid, node_group_ids(node)))
        node["class_name"] = new_name
        node["target"] = new_name in target_classes
        promoted += bool(node["target"])

    none_gids = [g for n in nodes if not n.get("target") and str(n.get("class_name") or NONE_CLASS_NAME) == NONE_CLASS_NAME for g in node_group_ids(n)]
    rematch_gids = sorted({int(g) for g in none_gids + rescued if int(g) >= 0})
    added = 0
    if rematch_gids:
        none_by_gid = session.vote(set(rematch_gids), 0.0)
        _relabel_none_nodes(nodes, none_by_gid)
        added = append_stuff_only_nodes(nodes, rescued, none_by_gid, _class_from_vote_row)

    relabeled = sum(str(n.get("class_name") or NONE_CLASS_NAME) != before[id(n)] for n in voted)
    graph["nodes"] = minimal_nodes(nodes)
    return f"[scene_graph] class assignment: target nodes kept={len(frozen)} relabeled={relabeled} promoted={promoted} extra nodes={added}"


def _attach_orphan_groups(graph: dict, gids: np.ndarray, cb: np.ndarray, min_count: int, min_cosine: float):
    """Attach every codebook group that no node owns to the node with the most similar codebook entry.

    Refinement can move Gaussians into such groups. A group joins only if it has more than ``min_count`` Gaussians, the
    cosine to the best node's entry is >= ``min_cosine``, and that node is not a `none` node. Returns
    ``(attached, grown_ids)``: (group, size, node id, class, cosine) tuples and the ids of the nodes that gained groups."""
    cn = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-12)
    owner = {g: n for n in graph["nodes"] for g in node_group_ids(n)}
    ids, counts = np.unique(gids, return_counts=True)
    attached, grown = [], set()
    for g, c in sorted(zip(ids.tolist(), counts.tolist()), key=lambda t: -t[1]):
        if g in owner or c <= min_count:
            continue
        best = max(owner, key=lambda k: float(cn[g] @ cn[k]))
        cos = float(cn[g] @ cn[best])
        node = owner[best]
        if cos < min_cosine or node["class_name"] == "none":
            continue
        node["group_id"] = node_group_ids(node) + [g]
        grown.add(node["id"])
        attached.append((g, c, node["id"], node["class_name"], cos))
    return attached, grown


def _complete_target_nodes(graph: dict, grown_ids: set, ply_path: str, cb: torch.Tensor, gids: np.ndarray, cfg: Dict[str, Any]) -> None:
    """Give every target node a ``bbox_tight`` and a ``text_prompt``.

    Boxes are refit with the rule of every refinement round for nodes that gained groups and for nodes without a box
    (promoted by the class vote). A promoted node whose ROI is empty cannot be refined and stays a non-target node. A node
    without a prompt (promoted, so never seen by the VLM) is prompted with its class name."""
    from scene import GaussianModel

    todo = [n for n in graph["nodes"] if n["target"] and (n["id"] in grown_ids or not n.get("bbox_tight"))]
    if todo:
        gaussians = GaussianModel(3)
        gaussians.load_ply(ply_path)
        refresh_target_node_bbox_tight({"nodes": todo}, gaussians, cb, torch.from_numpy(gids).to(cb.device), cfg)
    for n in graph["nodes"]:
        if not n["target"]:
            continue
        if not n.get("bbox_tight"):
            n["target"] = False
        elif not n.get("text_prompt"):
            n["text_prompt"] = n["class_name"]


def finalize_scene_graph(iteration_dir, cfg, assign=None, min_cosine=0.9):
    """Write the final ``scene_graph.json`` in ``iteration_dir`` after the last training iteration (called by train_vlsplat.py).

    1. node-less groups are attached to the most similar node (``_attach_orphan_groups``, group size > ``cfg["min_count"]``);
    2. if ``assign`` (the keyword arguments of ``assign_node_classes``) is given, the classes of the non-target nodes are
       re-voted with SAM3 (VLM-labelled target nodes stay frozen);
    3. every target node gets a box and a prompt (``_complete_target_nodes``);
    4. the graph is saved with its ``edges``."""
    from scene_graph_builder.build_scene_graph_nodes import compute_class_ids_np, load_codebook, read_object_features_from_ply

    d = Path(iteration_dir).resolve()
    ply_path = str(d / "point_cloud.ply")
    graph = load_scene_graph(str(d / "scene_graph.json"))
    cb = load_codebook(str(d / "classifier.pth"), device="cuda").detach()
    gids = compute_class_ids_np(read_object_features_from_ply(ply_path, 16), cb.cpu().numpy())

    attached, grown = _attach_orphan_groups(graph, gids, cb.cpu().numpy(), int(cfg["min_count"]), min_cosine)
    print(f"[scene_graph] attached {len(attached)} node-less groups ({sum(a[1] for a in attached)} Gaussians) to existing nodes", flush=True)
    for g, c, nid, cls, cos in attached:
        print(f"[scene_graph]   group {g} ({c} pts) -> node {nid} {cls} (cosine {cos:.2f})", flush=True)
    if assign:
        print(assign_node_classes(graph, d, **assign), flush=True)
    _complete_target_nodes(graph, grown, ply_path, cb, gids, cfg)
    graph["nodes"] = minimal_nodes(graph["nodes"])
    save_scene_graph(str(d / "scene_graph.json"), graph, with_edges=True)
    return attached
