"""ScanNet++ (render-level): ADC with movable Gaussians, train/test split.

One file for steps 1-3 (1_train_15k.sh, 2_init_scene_graph.sh, 3_train_15k_30k.sh) on this dataset.
Select it with   DATASET_CONFIG=vlsplat_render_scannetpp bash 1_train_15k.sh   (default: vlsplat_render_scannetpp).
Environment variables of the shell scripts (SCENE, OUTPUT_DIR, PORT, ...) still override the paths below, and any
GRAPH / REFINE key is a command-line flag of init_scene_graph.py / train_vlsplat.py, e.g.
    bash 2_init_scene_graph.sh --roi-k 8        bash 3_train_15k_30k.sh --sg-gate-thr 0.6
"""

# ---- dataset, run names, and training mode (steps 1 and 3) -------------------------------------------------------
DATASET = {
    "name": "vlsplat_render_scannetpp",
    "dataset_root_dir": "./data/scannetpp",
    "output_dir": "./output/scannetpp",
    "images_dir": "images",
    "object_mask_dir": "object_mask",
    "default_scene": "8b5caf3398",
    "seed": 0,
    "train_test_split": True,          # train on the train split only, hold out the test views (--train_split --eval)
    "fixed_gaussian": False,           # False = ADC (densify/prune) with movable xyz
    "densify_until_iter": 10000,         # ADC (densify / prune / opacity reset) runs until this iteration
    "freeze_gaussian_xyz": False,      # positions are optimized
    "num_classes": 256,                # codebook size (output channels of the object classifier)
    "base_iters": 15000,               # step 1 length = start of refinement
    "final_iters": 30000,
    "base_run": "base_15k",            # step 1 output  -> <output_dir>/<scene>/<base_run>
    "refine_run": "vlsplat_30k",       # step 3 output  (+ _<PIPELINE_VARIANT> when set)
    "unified_lift_run": "unified_lift_30k",   # baseline output when UNIFIED_LIFT=1
    "unified_lift": False,             # True trains the Unified-Lift baseline instead (see unified_lift+ours_scannetpp.py)
    "graph_resolution": 4,             # image downscale used by step 2 when reading masks / views
}

# ---- step 2: scene-graph construction (init_scene_graph.py) ------------------------------------------------------
GRAPH = {
    # ---- stage 1: structure (codebook groups -> nodes, boxes, edges) -----------------------------------------
    "target_classes": ["cabinet", "bed", "chair", "desk", "sofa", "monitor"],  # thing classes refined later
    "codebook_cosine_threshold": 0.9,   # tau_cb: cosine >= this counts a Gaussian as belonging to the group
    "min_high_sim_count": 100,          # a group becomes a node only with >= this many such Gaussians
    "roi_k": 4,                         # KNN k of the spatial ROI crop
    "roi_overlap_tolerance": 0.5,       # anisotropic grouping: 0 = strict, 1 = loose overlap test between neighbouring Gaussians
    "roi_top_group_fraction": 0.90,     # the largest spatial components covering this share of the Gaussians form the ROI
    "roi_margin_fraction": 0.0,         # ROI crop margin (the tight box carries no margin)
    "structure_min_render_pixels": 1000,  # a view votes for a group only if it renders >= this many pixels of it
    "valid_mask_margin_pct": 0.1,       # image-edge margin for thing-class SAM3 votes (stuff uses the full image)

    # ---- stage 2: VLM class review (variant sam3_vlm) ---------------------------------------------------------
    "num_views": 3,                     # reference views kept per target node
    "class_review_min_confidence": 0.75,  # VLM answers below this confidence keep the SAM3 class
    "sam3_min_views": 5,                # SAM3 label is kept without a VLM call at >= this many agreeing views ...
    "sam3_min_share": 0.8,              # ... and >= this share of the votes
    "review_cover": 2,                  # every reviewed node is visible in >= this many review frames
    "review_max_frames": 20,            # cap on the number of review frames
    "review_max_escalations": 20,       # per-node calls for split / low-confidence votes
    "review_workers": 4,                # parallel VLM calls

    # ---- stage 2: VLM text description of target nodes --------------------------------------------------------
    "model": "google/gemini-3-flash-preview",  # OpenRouter model id
    "api_key_env": "OPENROUTER_API_KEY",  # environment variable that holds the API key
    "box_color": "lime",                # colour of the node box drawn on the views
    "max_image_side": 1024,             # longest image side sent to the VLM
    "max_words": 10,                    # description length limit
    "temperature": 0.0,
    "max_repair": 2,                    # re-ask attempts when a description breaks the rules
    "sleep_between": 0.0,               # seconds between VLM calls
}

# ---- step 3: intra-/inter-node refinement (train_vlsplat.py --sg-*; key gate_thr <-> --sg-gate-thr) ---------------
REFINE = {
    # ---- schedule ---------------------------------------------------------------------------------------------
    "refine_end": 29500,                # last refinement round (rounds: start, start+interval, ... <= end)
    "refine_interval": 500,             # iterations between rounds (start = START_ITER of the shell script)

    # ---- CLIP gate (per node, per round) ----------------------------------------------------------------------
    "clip_iters": 40,                   # Adam steps on the opacity gates
    "views_per_iter": 4,                # 2 random views + their 2 azimuth-opposite views
    "render_width": 224,
    "render_height": 224,
    "clip_model": "ViT-L-14",
    "clip_pretrained": "openai",
    "clip_input_size": 224,
    "radius_min": 0.8,                  # orbit radius sampled per view, in a frame where the node is 0.5 wide
    "radius_max": 1.2,
    "gate_thr": 0.7,                    # tau_g: gate >= this keeps a Gaussian
    "gate_warm_start": 0.5,             # gate initialization

    # ---- codebook-feature update (Eq. 6) ----------------------------------------------------------------------
    "feat_sim_threshold": 0.95,         # tau_f
    "alpha_attract": 0.55,
    "cos_cap": 0.95,                    # attract target cosine
    "repel_base": 0.02,
    "repel_scale": 0.15,
    "cos_floor": 0.05,

    # ---- ROI / boxes (same as the graph construction in config/scene_graph_config.py) -------------------------
    "bbox_margin_fraction": 0.1,        # per-side expansion of bbox_tight for inter-node candidates
    "roi_k": 4,
    "roi_overlap_tolerance": 0.5,
    "roi_top_group_fraction": 0.90,
    "roi_margin_fraction": 0.0,
    "roi_eps": 1e-6,
    "codebook_sim_threshold": 0.9,      # tau_cb
    "min_count": 100,
    "cross_group_union_min_count": 100,

    # ---- logging ----------------------------------------------------------------------------------------------
    "quiet_node_refine": True,          # hide per-node CLIP-gate logs
}
