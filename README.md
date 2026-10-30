# VLSplat

### Vision-Language Guided Object-Centric 3D Gaussian Splatting via Scene Graph

[Project Page](https://vlsplat.github.io/)

VLSplat lifts 2D semantic masks into object-centric 3D Gaussian groups and refines them with object-level language priors organized as a scene graph. A codebook associates Gaussians into object groups; each group becomes a graph node carrying a class, a VLM-written description, and a 3D box. Two refinement stages then run under CLIP guidance on synthetic object views — *intra-node* suppresses noisy Gaussians inside a node, *inter-node* acquires missing parts from neighbouring or background nodes — updating features only, never geometry.

## Code Structure

```
VLSplat/
├── 1_train_15k.sh                 stage 1  codebook lifting
├── 2_init_scene_graph.sh          stage 2  scene graph + VLM prompts
├── 3_train_15k_30k.sh             stage 3  intra/inter refinement
├── test_vlsplat_2d_render.sh      eval     render-level (VLSplat)
├── test_unified-lift_2d_render.sh eval     render-level (baseline)
├── test_vlsplat_3d_point.sh       eval     point-level
├── train_vlsplat.py               training loop
├── config/                        one config per setup (paths, training mode, graph and refinement settings)
├── preprocess/                    stage 0  SAM3 + SAM2 masks (run_masks.sh)
├── scene_graph_builder/           scene graph: construction (step 2), per-cycle refinement and last-step finalize (step 3)
├── clip_refine/                   synthetic object views + CLIP gating
├── script/                        evaluation (and the Unified-Lift eval graph)
├── scene/ gaussian_renderer/ utils/ arguments/    3DGS core
└── submodules/                    diff-gaussian-rasterization, simple-knn
```

## Installation

Python 3.8, PyTorch 2.4.1, CUDA 12.1, gcc/g++ 12, one GPU with ≥ 24 GB.

```bash
conda create -n vlsplat python=3.8 -y
conda activate vlsplat

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install plyfile==0.8.1 tqdm scipy wandb opencv-python scikit-learn lpips open_clip_torch

CC=gcc-12 CXX=g++-12 pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation

export OPENROUTER_API_KEY='sk-or-...'   # VLM of step 2
```

Mask preprocessing uses a separate `sam3` environment (Python 3.12, PyTorch 2.10, CUDA 12.8):

```bash
conda create -n sam3 python=3.12 pip -y
conda activate sam3
pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/facebookresearch/sam3.git preprocess/sam3/sam3
git clone https://github.com/facebookresearch/sam2.git preprocess/sam2/sam2

SAM2_BUILD_CUDA=0 pip install --no-build-isolation \
    -e preprocess/sam3/sam3 -e preprocess/sam2/sam2 \
    -r preprocess/requirements.txt \
    'setuptools<81' 'opencv-python<4.12' scipy matplotlib

mkdir -p preprocess/sam3/weights preprocess/sam2/weights
hf auth login    # access to facebook/sam3 required
hf download facebook/sam3 sam3.pt --local-dir preprocess/sam3/weights
curl -fL --retry 3 \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
    -o preprocess/sam2/weights/sam2.1_hiera_large.pt
```

## Data

Place each scene under `data/<dataset>/<scene>/` (`<dataset>` = `scannetpp`, `scannetv2`, ...); a symlink works: `ln -s /path/to/datasets data`.

**Input (before preprocessing)**

```
data/<dataset>/<scene>/
├── images/                RGB frames
├── sparse/0/              COLMAP model (cameras, images, points3D)
├── train_test_lists.json  render protocol: train/test split
└── gt/
    ├── render/            render protocol: semantic index PNGs, one per test view
    └── points/            point protocol: *_vh_clean_2.labels.ply, *.aggregation.json, *.segs.json
```

### Preprocessing

```bash
conda activate sam3
DATASET=scannetpp SCENE=8b5caf3398 bash preprocess/run_masks.sh
DATASET=scannetv2 SCENE=scene0590_00 bash preprocess/run_masks.sh
```

**Output (new folders next to `images/`)**

```
data/<dataset>/<scene>/
├── object_mask/           per-image object-ID masks, used for training
├── object_mask_json/      per-image label of each mask ID (SAM3 class, or SAM2 fill)
└── object_mask_vis/       colour visualizations for inspection
```

Training outputs go to `output/<dataset>/<scene>/<run>/`.

## Training and Inference

### VLSplat (render-level)

```bash
conda activate vlsplat
export DATASET_CONFIG=vlsplat_render_scannetpp SCENE=8b5caf3398   # config/vlsplat_render_scannetpp.py

bash 1_train_15k.sh          # → base_15k
bash 2_init_scene_graph.sh   # → base_15k/point_cloud/iteration_15000/scene_graph_sam3_vlm.json
bash 3_train_15k_30k.sh      # → vlsplat_30k (final graph: point_cloud/iteration_30000/scene_graph.json)
```

Without an OpenRouter key, export `PIPELINE_VARIANT=sam3` before steps 2 and 3 (SAM3 votes only, no VLM).

Unified-Lift baseline (reuses the same 15k checkpoint):

```bash
DATASET_CONFIG=unified_lift+ours_scannetpp bash 3_train_15k_30k.sh   # → unified_lift_30k
```

### VLSplat (point-level)

```bash
export DATASET_CONFIG=vlsplat_point_scannetv2 SCENE=scene0590_00   # config/vlsplat_point_scannetv2.py

bash 1_train_15k.sh          # → base_15k_fixed
bash 2_init_scene_graph.sh
bash 3_train_15k_30k.sh      # → vlsplat_30k_fixed
```

## Evaluation

```bash
SCENE=8b5caf3398 bash test_vlsplat_2d_render.sh        # render-level
SCENE=8b5caf3398 bash test_unified-lift_2d_render.sh   # render-level baseline
SCENE=scene0590_00 bash test_vlsplat_3d_point.sh       # point-level
```

## Acknowledgements

Built on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and
[Unified-Lift](https://github.com/Runsong123/Unified-Lift), with masks from
[SAM2](https://github.com/facebookresearch/sam2) /
[SAM3](https://github.com/facebookresearch/sam3) and language features from
[OpenCLIP](https://github.com/mlfoundations/open_clip). Please follow the original 3D Gaussian
Splatting license terms, which restrict use to non-commercial research.

## Citation

```bibtex
@inproceedings{jeon2026vlsplat,
  title     = {VLSplat: Vision-Language Guided Object-Centric 3D Gaussian Splatting via Scene Graph},
  author    = {Jeon, Yuntae and Jeon, Younho and Jin, Sujin and Jo, Sungho and Park, Seunghee},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
