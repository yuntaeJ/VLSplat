"""
Per image: SAM3 inference → postprocess → save once
  → sam3_masks/
  → sam3_json/

Final overlay is written only by merge → object_mask_vis/
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Put the local checkout before this script's directory, which also contains
# a directory named sam3 and can shadow an editable installation.
_SAM3_SRC = Path(__file__).resolve().parent / "sam3"
if (_SAM3_SRC / "sam3" / "__init__.py").is_file():
    sys.path.insert(0, str(_SAM3_SRC))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from importlib.resources import files

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from mask_utils import (
    extract_instances,
    merge_background_per_type,
)
from sam3_postprocess import (
    JSON_OUT_SUBDIR,
    MASK_OUT_SUBDIR,
    MIN_AREA,
    dedup_cross_class_bbox,
    paint_instance_mask_uint16,
)


_SCRIPT_DIR = Path(__file__).resolve().parent
# Tokenizer vocab, located next to the SAM3 package.
BPE_PATH = str(files("sam3") / "assets" / "bpe_simple_vocab_16e6.txt.gz")
SAM3_CKPT_DEFAULT = str(_SCRIPT_DIR / "weights" / "sam3.pt")

TEXT_PROMPTS: List[str] = [
    "Wall",
    "floor",
    "cabinet",
    "ceiling",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "picture",
    "desk",
    "curtain",
]


def build_processor(bpe_path: str, checkpoint_path: Optional[str] = None) -> Sam3Processor:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Architecture is defined in sam3.model_builder (no YAML).
    kwargs = {"bpe_path": bpe_path, "load_from_HF": False}
    if checkpoint_path:
        kwargs["checkpoint_path"] = checkpoint_path
    else:
        kwargs["load_from_HF"] = True
    model = build_sam3_image_model(**kwargs)
    return Sam3Processor(model, confidence_threshold=0.3)


def get_image_paths(root: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG"}
    return sorted(p for p in Path(root).rglob("*") if p.suffix in exts and p.is_file())


def _clone_text_outputs(text_outputs: Dict) -> Dict:
    """Detach/clone language tensors so each image/prompt run is independent."""
    out = {}
    for k, v in text_outputs.items():
        if torch.is_tensor(v):
            out[k] = v.detach().clone()
        else:
            out[k] = v
    return out


@torch.inference_mode()
def build_text_cache(
    processor: Sam3Processor, text_prompts: List[str]
) -> Dict[str, Dict]:
    """Encode each prompt once; reuse across all images."""
    cache: Dict[str, Dict] = {}
    print(f"=== text encode cache ({len(text_prompts)} prompts) ===")
    for prompt in text_prompts:
        cache[prompt] = processor.model.backbone.forward_text(
            [prompt], device=processor.device
        )
        print(f"  cached: {prompt}")
    return cache


def run_prompt_cached(
    processor: Sam3Processor,
    state: Dict,
    prompt: str,
    text_cache: Dict[str, Dict],
) -> Dict:
    """Like Sam3Processor.set_text_prompt, but uses a precomputed text encoding."""
    if "backbone_out" not in state:
        raise ValueError("You must call set_image before run_prompt_cached")
    state["backbone_out"].update(_clone_text_outputs(text_cache[prompt]))
    state["geometric_prompt"] = processor.model._get_dummy_prompt()
    return processor._forward_grounding(state)


def postprocess_in_memory(
    raw_mask: np.ndarray,
    confidence_map: Dict[int, Dict],
) -> Tuple[np.ndarray, Dict[str, Dict], list]:
    """Apply v2 postprocess without writing raw intermediates."""
    label_map = {str(k): v for k, v in confidence_map.items()}
    instances = extract_instances(raw_mask.astype(np.uint32), label_map)
    instances = merge_background_per_type(instances)
    instances = dedup_cross_class_bbox(instances)
    out_mask, instances = paint_instance_mask_uint16(instances, raw_mask.shape[:2])
    n_after = len(instances)
    instances = [inst for inst in instances if int(inst.region.sum()) >= MIN_AREA]
    if len(instances) < n_after:
        out_mask, instances = paint_instance_mask_uint16(instances, raw_mask.shape[:2])

    out_label_map = {
        str(inst.new_id): {"label": inst.label, "score": inst.score}
        for inst in instances if inst.new_id > 0
    }
    return out_mask, out_label_map, instances


def process_image(
    image_path: Path,
    processor: Sam3Processor,
    text_prompts: List[str],
    text_cache: Dict[str, Dict],
    mask_dir: Path,
    json_dir: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    raw_mask = np.zeros((height, width), dtype=np.uint16)
    instance_id = 0
    confidence_map: Dict[int, Dict] = {}

    state = processor.set_image(image)

    for prompt in text_prompts:
        state = run_prompt_cached(processor, state, prompt, text_cache)

        masks = state.get("masks")
        scores = state.get("scores")

        if masks is None or masks.shape[0] == 0:
            continue

        masks_np = masks.squeeze(1).cpu().numpy().astype(bool)
        scores_np = scores.float().cpu().numpy() if scores is not None else np.ones(masks_np.shape[0])

        for k in range(masks_np.shape[0]):
            m = masks_np[k]
            if not m.any():
                continue
            instance_id += 1
            score = float(np.clip(scores_np[k], 0.0, 1.0))
            raw_mask[m] = instance_id
            confidence_map[instance_id] = {"label": prompt, "score": round(score, 6)}

    torch.cuda.empty_cache()

    out_mask, out_label_map, _ = postprocess_in_memory(raw_mask, confidence_map)

    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_mask.astype(np.uint16)).save(mask_dir / f"{image_path.stem}.png")

    json_dir.mkdir(parents=True, exist_ok=True)
    with open(json_dir / f"{image_path.stem}.json", "w") as f:
        json.dump(out_label_map, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SAM3 masks and run postprocess v2."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="RGB image directory (.../<scene>/images)",
    )
    parser.add_argument(
        "--text-prompts",
        nargs="+",
        default=None,
        help="Text prompts for SAM3 (space-separated). Default: built-in indoor list.",
    )
    parser.add_argument(
        "--sam3-ckpt",
        default=SAM3_CKPT_DEFAULT if Path(SAM3_CKPT_DEFAULT).exists() else None,
        help="Local SAM3 checkpoint .pt (default: sam3/weights/sam3.pt). "
             "If missing, download from HuggingFace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).resolve()
    scene_dir = image_dir.parent
    text_prompts = args.text_prompts if args.text_prompts else list(TEXT_PROMPTS)

    print(f"images:  {image_dir}")
    print(f"scene:   {scene_dir}")
    print(f"prompts: {text_prompts}")

    image_paths = get_image_paths(str(image_dir))
    if not image_paths:
        print(f"no images under {image_dir}")
        return
    print(f"{len(image_paths)} images\n")

    bpe_path = Path(BPE_PATH)
    if not bpe_path.exists():
        raise FileNotFoundError(f"BPE missing: {bpe_path}")
    if args.sam3_ckpt is not None and not Path(args.sam3_ckpt).exists():
        raise FileNotFoundError(f"SAM3 checkpoint missing: {args.sam3_ckpt}")
    processor = build_processor(str(bpe_path), checkpoint_path=args.sam3_ckpt)
    text_cache = build_text_cache(processor, text_prompts)

    mask_dir = scene_dir / MASK_OUT_SUBDIR
    json_dir = scene_dir / JSON_OUT_SUBDIR

    print("=== SAM3 inference + postprocess ===")
    for img_path in tqdm(image_paths, desc="SAM3", unit="img"):
        process_image(
            image_path=img_path,
            processor=processor,
            text_prompts=text_prompts,
            text_cache=text_cache,
            mask_dir=mask_dir,
            json_dir=json_dir,
        )

    print(f"\ndone -> {mask_dir}")


if __name__ == "__main__":
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        main()
