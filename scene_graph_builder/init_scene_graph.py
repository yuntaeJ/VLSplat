#!/usr/bin/env python3
"""Build one complete scene_graph.json from a trained checkpoint, in three stages.

  structure     nodes (target + non-target), edges, per-node 3D bbox
                -> build_scene_graph_nodes.py
  vlm           VLM class review of every node, then one CLIP-friendly text_prompt per target node
                -> label_and_describe_nodes.py
  merge         link the fragments of each thing object into one node (a merged node keeps the best-evidence member's prompt)
                -> merge_object_fragments.py

Each stage reads the previous stage's JSON and adds to it. Intermediate files stay in
a temporary directory. This writes only the final scene_graph.json.

  bash 2_init_scene_graph.sh
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scene_graph_builder.graph_io import load_scene_graph, minimal_nodes, save_scene_graph  # noqa: E402

STAGE_STRUCTURE = "structure"
STAGE_VLM = "vlm_review_and_description"
STAGE_MERGE = "merge"

BUILD_SCRIPT = _HERE / "build_scene_graph_nodes.py"
REVIEW_SCRIPT = _HERE / "label_and_describe_nodes.py"


def _load_config() -> Dict[str, Any]:
    """GRAPH section of config/<dataset>.py (--dataset-config / $DATASET_CONFIG, default vlsplat_render_scannetpp)."""
    from config.load_config import load_dataset_config

    return load_dataset_config()["GRAPH"]


def _parse_args() -> argparse.Namespace:
    cfg = _load_config()
    p = argparse.ArgumentParser(
        description="Build a complete scene_graph.json (structure + class review + VLM text). "
        "Defaults come from the GRAPH section of config/<dataset>.py; every key there is also a flag here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("iteration_dir", type=str, help="Checkpoint dir, e.g. .../iteration_15000.")
    p.add_argument("--source-path", type=str, required=True, help="Dataset scene root.")
    p.add_argument("--object-path", type=str, default="", help="Default: <source-path>/object_mask.")
    p.add_argument(
        "--label-json-root", type=str, default="", help="Default: <source-path>/object_mask_json."
    )
    p.add_argument("--images", type=str, default="images")
    p.add_argument("--resolution", type=int, default=4)
    p.add_argument("--dataset-config", type=str, default=None, help="config/<name>.py (default: $DATASET_CONFIG or vlsplat_render_scannetpp).")
    p.add_argument("--out-name", type=str, default="scene_graph.json", help="Final JSON filename.")
    p.add_argument("--variant", choices=["sam3", "sam3_vlm"], default="sam3_vlm",
                   help="sam3: classes from SAM3 label votes only, no VLM call (text_prompt = class name). "
                        "sam3_vlm: SAM3 proposes the nodes, the VLM assigns every node's class.")
    p.add_argument("--save-request-images", action="store_true")

    # Every config key becomes a flag whose default is the config value.
    for key, value in cfg.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, list):
            p.add_argument(flag, nargs="+", default=value)
        else:
            p.add_argument(flag, type=type(value), default=value)

    return p.parse_args()


def _run_stage(label: str, cmd: Sequence[str], *, cwd: Path) -> None:
    print(f"[{label}] ...", flush=True)
    env = dict(os.environ)
    env["OPENROUTER_USAGE_STAGE"] = label
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    started = time.time()
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise SystemExit(f"[{label}] failed with exit code {result.returncode}")
    print(f"[{label}] done in {time.time() - started:.1f}s", flush=True)


def _validate(graph_path: Path, num_views: int) -> None:
    """Fail if the final graph is missing class names, views, or prompts."""
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes: List[Dict[str, Any]] = list(graph.get("nodes", []))
    targets = [n for n in nodes if bool(n.get("target", False))]
    missing_class = [int(n.get("id", -1)) for n in nodes if not str(n.get("class_name", "")).strip()]
    if missing_class:
        raise SystemExit(f"nodes missing class_name: {missing_class}")

    unreviewed = [n["id"] for n in nodes if not n.get("class_review")]
    if unreviewed:
        raise SystemExit(f"nodes missing class review: {unreviewed}")

    incomplete: List[str] = []
    for node in targets:
        refs = list(node.get("vlm_reference") or [])
        prompt = str(node.get("text_prompt", "") or "").strip()
        issues: List[str] = []
        if len(refs) < int(num_views):
            issues.append(f"{len(refs)}/{num_views} views")
        if not prompt:
            issues.append("no text_prompt")
        if issues:
            incomplete.append(
                f"node {int(node['id'])} {str(node.get('class_name', ''))}: " + ", ".join(issues)
            )
    if incomplete:
        print("  [warn] some target nodes are incomplete:", flush=True)
        for line in incomplete:
            print(f"    {line}", flush=True)


def _minimize(graph_path: Path) -> None:
    """Reduce the stage output to the final node schema (graph_io.NODE_FIELDS) and add the edges."""
    graph = load_scene_graph(str(graph_path))
    graph["nodes"] = minimal_nodes(graph["nodes"])
    save_scene_graph(str(graph_path), graph, with_edges=True)


def main() -> None:
    args = _parse_args()

    iter_dir = Path(args.iteration_dir).resolve()
    if not iter_dir.is_dir():
        raise FileNotFoundError(f"iteration dir not found: {iter_dir}")
    source_path = Path(args.source_path).resolve()
    object_path = Path(args.object_path).resolve() if args.object_path else source_path / "object_mask"
    label_json_root = (
        Path(args.label_json_root).resolve()
        if args.label_json_root
        else source_path / "object_mask_json"
    )

    final_json = iter_dir / str(args.out_name)

    # Fail before the expensive rendering stages rather than after them.
    if args.variant == "sam3_vlm" and not os.environ.get(str(args.api_key_env), "").strip():
        raise SystemExit(
            f"{args.api_key_env} is not set, the {STAGE_VLM} stage needs it:\n"
            f"  export {args.api_key_env}='sk-or-...'"
        )

    with tempfile.TemporaryDirectory(prefix="scene_graph_stages_") as tmp:
        structure_json = Path(tmp) / "1_structure.json"
        views_json = Path(tmp) / "2_views.json"

        structure_cmd = [
            sys.executable,
            BUILD_SCRIPT,
            iter_dir,
            "--source-path", source_path,
            "--object-path", object_path,
            "--label-json-root", label_json_root,
            "--images", args.images,
            "--resolution", args.resolution,
            "--codebook-cosine-threshold", args.codebook_cosine_threshold,
            "--min-high-sim-count", args.min_high_sim_count,
            "--roi-k", args.roi_k,
            "--roi-overlap-tolerance", args.roi_overlap_tolerance,
            "--roi-top-group-fraction", args.roi_top_group_fraction,
            "--roi-margin-fraction", args.roi_margin_fraction,
            "--min-render-pixels", args.structure_min_render_pixels,
            "--valid-mask-margin-pct", args.valid_mask_margin_pct,
            "--target-classes", *args.target_classes,
            "--out-json", structure_json,
            "--quiet",
        ]
        _run_stage(STAGE_STRUCTURE, structure_cmd, cwd=_ROOT)
        if not structure_json.is_file():
            raise SystemExit(f"[{STAGE_STRUCTURE}] produced no JSON at {structure_json}")

        if args.variant == "sam3":
            graph = json.loads(structure_json.read_text(encoding="utf-8"))
            for node in graph["nodes"]:
                if node.get("target"):
                    node["text_prompt"] = str(node.get("class_name", "")).strip()  # no VLM: CLIP prompt = class name
            final_json.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
            _minimize(final_json)
            n_t = sum(1 for n in graph["nodes"] if n.get("target"))
            print(f"[sam3] {len(graph['nodes'])} nodes, {n_t} targets -> {final_json}", flush=True)
            return

        cmd = [sys.executable, REVIEW_SCRIPT, iter_dir,
               "--source-path", source_path, "--object-path", object_path,
               "--scene-graph", structure_json, "--out-json", views_json,
               "--images", args.images, "--resolution", args.resolution,
               "--num-views", args.num_views, "--model", args.model,
               "--api-key-env", args.api_key_env,
               "--min-confidence", args.class_review_min_confidence,
               "--target-classes", *args.target_classes,
               "--sam3-min-views", args.sam3_min_views, "--sam3-min-share", args.sam3_min_share,
               "--cover", args.review_cover, "--max-frames", args.review_max_frames,
               "--max-escalations", args.review_max_escalations, "--workers", args.review_workers,
               "--box-color", args.box_color, "--max-image-side", args.max_image_side,
               "--max-words", args.max_words, "--temperature", args.temperature,
               "--max-repair", args.max_repair, "--sleep-between", args.sleep_between]
        if bool(args.save_request_images):
            cmd.append("--save-request-images")
        _run_stage(STAGE_VLM, cmd, cwd=_ROOT)
        if not views_json.is_file():
            raise SystemExit(f"[{STAGE_VLM}] produced no JSON at {views_json}")
        _run_stage(STAGE_MERGE, [
            sys.executable, _HERE / "merge_object_fragments.py", iter_dir,
            "--scene-graph", views_json, "--out-json", views_json,
            "--target-classes", *args.target_classes], cwd=_ROOT)
        final_json.write_text(views_json.read_text(encoding="utf-8"), encoding="utf-8")

    _validate(final_json, int(args.num_views))
    _minimize(final_json)
    print(f"Saved: {final_json}", flush=True)


if __name__ == "__main__":
    main()
