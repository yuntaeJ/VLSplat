#!/usr/bin/env python3
"""Make the graph one node per physical object (thing classes only).

Two thing nodes of one class merge when their Gaussians are linked by the anisotropic affinity graph
(or one is a small fragment right next to the other).
The merged node keeps the list of group ids, the union box, and the highest-evidence member's review/prompt. Stuff nodes are left untouched."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scene_graph_builder.graph_io import node_group_ids
from scene_graph_builder.label_and_describe_nodes import update_graph_categories

AX = ('x', 'y', 'z')


def _box(n):
    b = n['bbox_tight']
    return np.array([[b[a + 'min'] for a in AX], [b[a + 'max'] for a in AX]])


THING_CLASSES = ('cabinet', 'bed', 'chair', 'desk', 'sofa', 'monitor')


def _vol(b):
    return float(np.clip(b[1] - b[0], 0, None).prod())


def _aabb(points):
    lo, hi = points.min(0), points.max(0)
    return {a + s_: float((lo if s_ == 'min' else hi)[i]) for i, a in enumerate(AX) for s_ in ('min', 'max')}


def thing_cores(nodes, components_of, mu_all, min_frac, min_points):
    """ROI core of every thing node, used for the link test and to refit its box.

    Anisotropic components smaller than ``min_frac`` of the core (and < min_points) are dropped as strays.
    Returns the node list (thing boxes refit on the kept core) and {node index: core Gaussian indices}."""
    out, cores = [], {}
    for n in nodes:
        gl = n['group_id']
        if n['class_name'] not in THING_CLASSES or isinstance(gl, list):
            out.append(n)
            cores[len(out) - 1] = np.zeros(0, dtype=np.int64)
            continue
        comps = components_of(int(gl)); total = sum(len(c) for c in comps)
        comps = [c for c in comps if len(c) >= max(min_points, min_frac * total)] or [max(comps, key=len)]
        idx = np.concatenate(comps)
        node = dict(n); node['bbox_tight'] = _aabb(mu_all[idx])
        out.append(node); cores[len(out) - 1] = idx
    return out, cores


def merge_graph(graph, node_cores, link, skip_classes=('none',), max_growth=1.5, attach_ratio=0.15,
                attach_gap=0.3, min_link_edges=10, min_link_frac=0.05):
    """Greedy component merge of same-class thing nodes.

    Two components merge only if their Gaussians are actually connected by the anisotropic affinity graph
    (>= min_link_edges cross edges covering >= min_link_frac of the smaller side), or one is a small fragment
    (<= attach_ratio of the volume) whose Gaussians come within attach_gap metres of the other; and the union
    box may not exceed max_growth times the larger box. Far-apart groups therefore never chain together."""
    nodes = graph['nodes']
    comps = [dict(idx=[i], box=_box(n), groups={g for g in node_group_ids(n)}) for i, n in enumerate(nodes)]
    cache = {}
    def ok(c1, c2, cls):
        if cls not in THING_CLASSES:
            return False
        key = (tuple(sorted(c1['idx'])), tuple(sorted(c2['idx'])))
        if key not in cache:
            cache[key] = link(np.concatenate([node_cores[i] for i in key[0]]), np.concatenate([node_cores[i] for i in key[1]]))
        info = cache[key]
        v1, v2 = _vol(c1['box']), _vol(c2['box'])
        strong = info['cross_edges'] >= min_link_edges and info['linked_frac'] >= min_link_frac
        attach = min(v1, v2) <= attach_ratio * max(v1, v2) and info['gap'] <= attach_gap
        if not (strong or attach):
            return False
        union = np.array([np.minimum(c1['box'][0], c2['box'][0]), np.maximum(c1['box'][1], c2['box'][1])])
        return _vol(union) / max(max(v1, v2), 1e-12) <= max_growth + 1e-9
    changed = True
    while changed:
        changed = False
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                cls = nodes[comps[i]['idx'][0]]['class_name']
                if cls != nodes[comps[j]['idx'][0]]['class_name'] or cls in skip_classes:
                    continue
                if ok(comps[i], comps[j], cls):
                    comps[i]['idx'] += comps[j]['idx']; comps[i]['groups'] |= comps[j]['groups']
                    comps[i]['box'] = np.array([np.minimum(comps[i]['box'][0], comps[j]['box'][0]), np.maximum(comps[i]['box'][1], comps[j]['box'][1])])
                    del comps[j]; changed = True; break
            if changed:
                break
    merged, report = [], []
    for comp in comps:
        ms = [nodes[i] for i in comp['idx']]
        if len(ms) == 1:
            merged.append(dict(ms[0])); continue
        primary = max(ms, key=lambda n: (n['n_gaussians'], -n['id']))
        node = dict(primary)
        node['group_id'] = sorted({g for n in ms for g in node_group_ids(n)})
        node['bbox_tight'] = {a + s: float((min if s == 'min' else max)(n['bbox_tight'][a + s] for n in ms)) for a in AX for s in ('min', 'max')}
        node['n_gaussians'] = sum(n['n_gaussians'] for n in ms)
        node['target'] = any(n.get('target') for n in ms)
        merged.append(node)
        report.append(dict(class_name=node['class_name'], ids=[n['id'] for n in ms], groups=node['group_id']))
    merged.sort(key=lambda n: n['id'])
    for new_id, n in enumerate(merged, 1):
        n['id'] = new_id
    graph['nodes'] = merged
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('iteration_dir', type=Path)
    p.add_argument('--scene-graph', type=Path, required=True)
    p.add_argument('--out-json', type=Path, required=True)
    p.add_argument('--min-link-edges', type=int, default=10, help='thing classes: min anisotropic affinity edges between the two groups\' ROI cores')
    p.add_argument('--min-link-frac', type=float, default=0.05, help='...covering at least this fraction of the smaller core')
    p.add_argument('--attach-gap', type=float, default=0.3, help='small fragments attach when their Gaussians come within this distance (m)')
    p.add_argument('--core-cosine', type=float, default=0.9)
    p.add_argument('--stray-min-frac', type=float, default=0.05, help='components below this share of the core (and < --stray-min-points) are dropped as strays')
    p.add_argument('--stray-min-points', type=int, default=20)
    p.add_argument('--target-classes', nargs='+', default=['cabinet', 'bed', 'chair', 'desk', 'sofa', 'monitor'])
    a = p.parse_args()
    from scene_graph_builder.build_scene_graph_nodes import read_object_features_from_ply, load_codebook, compute_class_ids_np
    r = a.iteration_dir.resolve()
    gids = compute_class_ids_np(read_object_features_from_ply(str(r / 'point_cloud.ply'), 16),
                                load_codebook(str(r / 'classifier.pth'), 16).detach().cpu().numpy())
    f = read_object_features_from_ply(str(r / 'point_cloud.ply'), 16)
    cb = load_codebook(str(r / 'classifier.pth'), 16).detach().cpu().numpy()
    fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8); cn = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-8)
    graph = json.loads(a.scene_graph.read_text())
    from scene_graph_builder.build_scene_graph_nodes import build_gaussian_graph, group_gaussians, load_gaussians_from_ply
    from scipy.spatial import cKDTree
    mu_all, sigma_all = load_gaussians_from_ply(r / 'point_cloud.ply', scales_are_log=True, eps=1e-6)
    def core_components(gid):
        """Connected components of the anisotropic affinity graph for a group's cosine-filtered core: list of index arrays."""
        idx = np.flatnonzero(gids == gid); cos = fn[idx] @ cn[gid]; sel = idx[cos >= a.core_cosine]
        if len(sel) < 30 and len(idx):
            sel = idx[cos >= np.quantile(cos, 0.5)]
        if len(sel) < 2:
            return [sel]
        _, _, lab, _ = group_gaussians(mu_all[sel], sigma_all[sel], k=4, overlap_tolerance=0.5)
        return [sel[lab == u] for u in np.unique(lab)]
    def link(ia, ib):
        if len(ia) == 0 or len(ib) == 0:
            return dict(cross_edges=0, linked_frac=0., gap=1e9)
        idx = np.concatenate([ia, ib]); na = len(ia)
        edges, _ = build_gaussian_graph(mu_all[idx], sigma_all[idx], k=8, overlap_tolerance=0.5)
        cross = [(i, j) for i, j, _w in edges if (i < na) != (j < na)]
        n_side_a = len({i if i < na else j for i, j in cross}); n_side_b = len({j if i < na else i for i, j in cross})
        return dict(cross_edges=len(cross), linked_frac=min(n_side_a, n_side_b) / max(1, min(na, len(ib))), gap=float(cKDTree(mu_all[ib]).query(mu_all[ia])[0].min()))
    before = len(graph['nodes'])
    graph['nodes'], node_cores = thing_cores(graph['nodes'], core_components, mu_all, a.stray_min_frac, a.stray_min_points)
    report = merge_graph(graph, node_cores, link, min_link_edges=a.min_link_edges, min_link_frac=a.min_link_frac, attach_gap=a.attach_gap)
    update_graph_categories(graph, a.target_classes)
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(graph, indent=2) + '\n')
    print(f'[merge] {before} -> {len(graph["nodes"])} nodes ({len(report)} merged groups of nodes)', flush=True)
    for x in report:
        print(f"[merge]   {x['class_name']}: old node ids {x['ids']} -> groups {x['groups']}", flush=True)


if __name__ == '__main__':
    main()
