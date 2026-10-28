#!/usr/bin/env python3
"""Stage 2 of the scene graph: one VLM stage that labels every node and describes the targets.

1. class review   SAM3-confident nodes keep their class; the rest are classified from a few multi-node photos
                  (confidence-weighted vote), and only weak or split votes get one per-node call
2. description   every node that ends up in a target class gets a <= --max-words text_prompt from boxed views

Needs OPENROUTER_API_KEY. Run through 2_init_scene_graph.sh (init_scene_graph.py).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData
from threadpoolctl import threadpool_limits

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gaussian_renderer import render
from scene_graph_builder.build_scene_graph_nodes import (
    _build_scene,
    compute_class_ids_np,
    load_codebook,
    read_object_features_from_ply,
)

# ---------------------------------------------------------------------------
# Text prompts: one CLIP-friendly phrase per target node, via OpenRouter.
#
# For every target node: load its reference views, draw the node's bbox on each
# view as a bright rectangle (no text label, so nothing occludes the object),
# send one request with all views to a vision model, and sanitize the answer
# into a short noun phrase stored as "text_prompt".
#
# The bbox is only given visually: drawn marks ground the model far better than
# numeric coordinates in the prompt (Set-of-Mark, arXiv:2310.11441).
#
# Needs OPENROUTER_API_KEY in the environment.
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
IMAGE_EXTS = (".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png")
BOX_COLORS = {
    "lime": (0, 255, 0),
    "green": (0, 200, 0),
    "red": (255, 0, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "yellow": (255, 255, 0),
    "blue": (0, 80, 255),
}
STYLE_EXAMPLES = (
    "a white curved plastic chair with four slim metal legs",
    "a red mesh-back swivel office chair on a five-star castor base",
    "a white rectangular desk with four square grey metal legs",
    "a round white table on a single chrome pedestal base",
    "a tall beige wooden cabinet with two hinged front panels",
)
# Classes whose support structure (leg count, base type) is worth spelling out.
DEFAULT_LEG_CLASSES = ("chair", "desk", "table", "stool", "bench", "sofa", "couch")
# Mutually exclusive categories. A phrase may name its own category and nothing else from
# this list, so an axis-aligned bbox that unavoidably contains neighbours cannot leak them
# into the caption, and neither can relational wording like "wall-mounted".
DEFAULT_SEPARATE_CLASSES = (
    "wall",
    "floor",
    "ceiling",
    "bed",
    "cabinet",
    "chair",
    "curtain",
    "desk",
    "door",
    "monitor",
    "picture",
    "sofa",
    "window",
)
UNCERTAIN_WORDS = ("maybe", "possibly", "probably", "appears", "seems", "likely", "perhaps")
# CLIP cannot encode negation, so "a desk with no legs" pulls towards legged desks.
NEGATION_RE = re.compile(r"\b(?:no|not|without|lacking|absent|missing|free of|devoid)\b")
NEGATED_CLAUSE_RE = re.compile(
    r"\s*(?:[,;]\s*)?\b(?:with\s+no|with\s+out|without|and\s+no|lacking|missing)\b.*$", re.I
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate CLIP-friendly text_prompt per target node using OpenRouter."
    )
    p.add_argument(
        "--scene-graph",
        type=str,
        required=True,
        help="Scene graph JSON with vlm_reference on target nodes.",
    )
    p.add_argument("--source-path", type=str, required=True, help="Dataset scene root.")
    p.add_argument("--images", type=str, default="images", help="Image subdir under --source-path.")
    p.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Output JSON (default: <scene-graph>_text.json next to the input).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="google/gemini-3-flash-preview",
        help="OpenRouter model slug (must be vision capable).",
    )
    p.add_argument(
        "--api-key-env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable holding the OpenRouter API key.",
    )
    p.add_argument("--node-ids", nargs="+", default=[], help="Restrict to these target node ids.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Leave nodes that already have a non-empty text_prompt untouched.",
    )
    p.add_argument(
        "--box-color",
        choices=sorted(BOX_COLORS.keys()),
        default="lime",
        help="Rectangle colour drawn around the node.",
    )
    p.add_argument(
        "--box-width",
        type=int,
        default=0,
        help="Rectangle line width in px (0 = scale with image size).",
    )
    p.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Downscale each view to this longest side before upload (token cost).",
    )
    p.add_argument("--jpeg-quality", type=int, default=88)
    p.add_argument(
        "--max-words",
        type=int,
        default=18,
        help="Hard cap on words in the stored phrase (CLIP has a 77 token limit).",
    )
    p.add_argument(
        "--leg-classes",
        nargs="*",
        default=list(DEFAULT_LEG_CLASSES),
        help="Classes for which leg count / base type is requested. Pass with no values to disable.",
    )
    p.add_argument(
        "--separate-classes",
        nargs="*",
        default=list(DEFAULT_SEPARATE_CLASSES),
        help=(
            "Mutually exclusive categories: never described as part of one another, and the "
            "model may correct class_name to one of them. Pass with no values to disable."
        ),
    )
    # Greedy decoding, so visually identical objects converge to identical wording
    # and reruns stay reproducible.
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument(
        "--max-repair",
        type=int,
        default=2,
        help="Rewrite attempts when the phrase names another category (e.g. 'wall-mounted').",
    )
    p.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        help="Seconds to wait between nodes (rate limiting).",
    )
    p.add_argument(
        "--save-request-images",
        action="store_true",
        help="Save the exact marked images that are uploaded, for auditing.",
    )
    p.add_argument(
        "--request-image-dir",
        type=str,
        default="",
        help="Where to save marked images (default: <out-json dir>/vlm_request_images).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build everything and print the prompt, but make no API call.",
    )
    p.add_argument("--quiet", action="store_true", help="Only print warnings and errors.")
    return p.parse_args(argv)


def _find_image(images_dir: Path, image_name: str) -> Optional[Path]:
    direct = images_dir / image_name
    if direct.is_file():
        return direct
    stem = Path(image_name).stem
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    stem_low = stem.lower()
    if images_dir.is_dir():
        for p in sorted(images_dir.iterdir()):
            if p.is_file() and p.stem.lower() == stem_low:
                return p
    return None


def _mark_view(
    image_path: Path,
    bbox: Sequence[int],
    *,
    color: Tuple[int, int, int],
    box_width: int,
    max_side: int,
) -> Image.Image:
    """Full image with only a rectangle drawn around the node (no text, no fill)."""
    image = Image.open(image_path).convert("RGB")
    if max_side > 0 and max(image.width, image.height) > max_side:
        scale = max_side / float(max(image.width, image.height))
        new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
        bbox = [int(round(v * scale)) for v in bbox]
        image = image.resize(new_size, Image.LANCZOS)

    width = box_width if box_width > 0 else max(3, int(round(min(image.size) / 200.0)))
    if len(bbox) == 4:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        ImageDraw.Draw(image).rectangle([x0, y0, x1, y1], outline=color, width=width)
    return image


def _to_data_url(image: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality))
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _build_instruction(
    class_name: str,
    n_views: int,
    color_name: str,
    max_words: int,
    *,
    describe_legs: bool,
    separate_classes: Sequence[str],
    allow_class_correction: bool = True,
) -> str:
    examples = "\n".join(f"  {e}" for e in STYLE_EXAMPLES)

    correction_rule = (
        'The given category can also be wrong. If the outlined object clearly is one of '
        'the categories listed above but not the given category, use the correct one.\n'
        if allow_class_correction else
        f'The category "{class_name}" has already been verified or explicitly set. '
        'Keep this category exactly; describe it without reclassifying it.\n'
    )
    separation = ""
    if separate_classes:
        listed = ", ".join(separate_classes)
        others = ", ".join(c for c in separate_classes if c.lower() != class_name.lower())
        separation = (
            f"\nThese are separate object categories: {listed}.\n"
            f"Because the rectangle is axis-aligned it usually also contains neighbours: things "
            f"standing on, under, behind or pushed into the object, and the surfaces around it. "
            f"Describe the {class_name} on its own and leave every neighbour out.\n"
            f"HARD CONSTRAINT: \"{class_name}\" is the only word from that list allowed in your "
            f"phrase. None of these may appear at all, not even inside a compound word or a "
            f"hyphenated adjective: {others}. So \"wall-mounted\", \"floor-standing\", "
            f"\"next to the window\" and \"under the monitor\" are all forbidden.\n"
            f"Synonyms do not get around this. Dropping the banned word but still describing the "
            f"neighbour is just as wrong, so no \"integrated drawer pedestal\" for a separate "
            f"storage unit parked under a desk, and no \"glazed opening\" for a window. Drawers, "
            f"doors and storage compartments belong to a cabinet; a desk is only its top surface "
            f"with its legs or frame.\n"
            f"Describe the object itself, never its relationship to anything else. Say nothing "
            f"about what holds it up unless the support belongs to the object, such as its own "
            f"legs or its own base. If it has none of its own, simply end the phrase; never "
            f"write that it has no legs or no base, because a negation cannot be encoded.\n"
            f"{correction_rule}"
        )

    rules = [
        "start with \"a <category>\" or \"a <adjective> <category>\"",
        f"{max_words} words maximum, a noun phrase and not a sentence, no final period",
        "only visible physical attributes: colour, material, shape, distinctive parts",
        "prefer attributes that agree across the images where the object is clearly visible",
    ]
    if describe_legs:
        rules.append(
            "give the number of legs when the object stands on its own legs "
            '("four slim metal legs"), or name its own base when it has one '
            '("a single central pedestal base", "a five-star castor base"). Count across all '
            "the images. If the object has no legs or base of its own, say nothing about "
            "support at all rather than explaining what else holds it up"
        )
        rules.append(
            "add where the legs sit only when the arrangement is unusual "
            '("two front legs and a solid rear panel")'
        )
    rules += [
        "never mention the rectangle, the images, the viewpoints, the room, or any neighbouring "
        "object or surface",
        "no negations, no uncertainty words, no brand names",
    ]
    rules_text = "\n".join(f"- {r}" for r in rules)

    return (
        f"You are writing a caption that will be encoded by the CLIP text encoder.\n\n"
        f"The {n_views} images that follow show the SAME single object, photographed from "
        f"{n_views} different viewpoints of one room. In every image that object is outlined "
        f"by a bright {color_name} rectangle. Only describe what is inside that rectangle.\n"
        f"The object category is \"{class_name}\".\n"
        f"{separation}\n"
        f"Some of these images may be unreliable: a rectangle can cover only a small, heavily "
        f"occluded sliver of the object, or enclose the wrong thing. Judge each image for "
        f"yourself, discard the unreliable ones, and describe the object from the images where "
        f"it is clearly visible. A single good image is worth more than two bad ones.\n\n"
        f"Write ONE short English noun phrase describing that object.\n\n"
        f"Rules:\n{rules_text}\n\n"
        f"Style to imitate:\n{examples}\n\n"
        f"Reply with one JSON object and nothing else:\n"
        f'{{"class": "<the correct category>", "phrase": "<the noun phrase>"}}'
    )


def _parse_response(raw: str, fallback_class: str) -> Tuple[str, str]:
    """Pull (class, phrase) out of the model reply, tolerating fences and stray prose."""
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            phrase = str(data.get("phrase") or data.get("caption") or "").strip()
            cls = str(data.get("class") or data.get("category") or "").strip()
            if phrase:
                return (cls or fallback_class), phrase
        except (json.JSONDecodeError, AttributeError):
            pass

    # Model ignored the JSON request and answered with the bare phrase.
    return fallback_class, text


def _foreign_class_words(phrase: str, own_class: str, all_classes: Sequence[str]) -> List[str]:
    """Category words in the phrase that are not the object's own category.

    Word boundaries make hyphenated compounds count too, so "wall-mounted" reports "wall".
    """
    hits: List[str] = []
    low = phrase.lower()
    for cls in all_classes:
        cls = str(cls).lower()
        if cls == str(own_class).lower():
            continue
        if re.search(rf"\b{re.escape(cls)}s?\b", low):
            hits.append(cls)
    return hits


def _violations(phrase: str, own_class: str, all_classes: Sequence[str]) -> List[str]:
    """Human-readable reasons the phrase is unusable, for the repair request."""
    issues: List[str] = []
    foreign = _foreign_class_words(phrase, own_class, all_classes)
    if foreign:
        named = ", ".join(f'"{w}"' for w in foreign)
        issues.append(
            f"it names {named}, which are other object categories and are forbidden"
        )
    if NEGATION_RE.search(phrase.lower()):
        issues.append(
            "it uses a negation, and CLIP cannot encode negation, so simply leave the missing "
            "part unmentioned instead of stating its absence"
        )
    return issues


def _resolve_answer(
    raw: str,
    given_class: str,
    separate_classes: Sequence[str],
    max_words: int,
) -> Tuple[str, str, List[str]]:
    """(effective_class, sanitized phrase, warnings) from one model reply."""
    answered_class, phrase_raw = _parse_response(raw, given_class)
    effective_class = given_class
    allowed = {str(c).lower() for c in separate_classes}
    if (
        answered_class
        and answered_class.lower() != given_class.lower()
        and answered_class.lower() in allowed
    ):
        effective_class = answered_class.lower()
    phrase, warnings = _sanitize_phrase(phrase_raw, effective_class, max_words)
    return effective_class, phrase, warnings


def _build_messages(instruction: str, data_urls: Sequence[str]) -> List[Dict[str, Any]]:
    # OpenRouter parses better with the text part first, then the images.
    content: List[Dict[str, Any]] = [{"type": "text", "text": instruction}]
    for url in data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "user", "content": content}]


def _call_openrouter(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> str:
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    n_images = sum(
        1 for m in messages if isinstance(m.get("content"), list)
        for part in m["content"] if isinstance(part, dict) and part.get("type") == "image_url"
    )
    usage_log = os.environ.get("OPENROUTER_USAGE_LOG", "").strip()

    last_error = ""
    for attempt in range(1, int(max_retries) + 1):
        if usage_log:  # one line per HTTP request actually sent (retries included)
            try:
                with open(usage_log, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps({"stage": os.environ.get("OPENROUTER_USAGE_STAGE", ""), "model": model,
                                         "images": n_images, "attempt": attempt}) + "\n")
            except OSError:
                pass
        try:
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=float(timeout)
            )
        except Exception as exc:  # network hiccup
            last_error = f"request failed: {exc}"
            resp = None

        if resp is not None:
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    last_error = f"no choices in response: {json.dumps(data)[:300]}"
                else:
                    message = choices[0].get("message") or {}
                    text = str(message.get("content") or "").strip()
                    if text:
                        return text
                    last_error = "empty content in response"
            elif resp.status_code in (408, 409, 429, 500, 502, 503, 504):
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            else:
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:500]}")

        if attempt < int(max_retries):
            delay = min(30.0, 2.0 ** attempt)
            print(f"  [retry {attempt}/{max_retries - 1} in {delay:.0f}s] {last_error}", file=sys.stderr)
            time.sleep(delay)

    raise RuntimeError(f"OpenRouter call failed after {max_retries} attempts: {last_error}")


def _sanitize_phrase(raw: str, class_name: str, max_words: int) -> Tuple[str, List[str]]:
    """Reduce a model answer to one lowercase noun phrase; returns (phrase, warnings)."""
    warnings: List[str] = []
    text = str(raw).strip()

    # Keep the first meaningful line and drop common wrappers.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        text = lines[0]
        if len(lines) > 1:
            warnings.append("multi-line answer, kept first line")
    text = re.sub(r"^\s*(?:phrase|caption|answer|output)\s*[:\-]\s*", "", text, flags=re.I)
    text = text.strip().strip("`").strip()
    text = text.strip('"').strip("'").strip()
    text = re.sub(r"^[-*\u2022]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(".!,;:").strip()
    text = text.lower()

    if not text:
        return "", ["empty answer"]

    for word in UNCERTAIN_WORDS:
        if re.search(rf"\b{word}\w*\b", text):
            text = re.sub(rf"\b{word}\w*\b", "", text)
            warnings.append(f"removed uncertainty word '{word}'")
    text = re.sub(r"\s+", " ", text).strip()

    if class_name and class_name not in text:
        text = f"a {class_name}, {text}" if text else f"a {class_name}"
        warnings.append(f"class name '{class_name}' was missing, prepended")

    words = text.split()
    if len(words) > int(max_words):
        text = " ".join(words[: int(max_words)])
        warnings.append(f"truncated from {len(words)} to {max_words} words")

    return text, warnings


def _clip_token_count(text: str) -> Optional[int]:
    """Exact CLIP token count when open_clip is installed, else None."""
    try:
        import open_clip

        tokens = open_clip.tokenize([text])
        return int((tokens[0] != 0).sum().item())
    except Exception:
        return None


def describe_nodes(nodes: List[Dict[str, Any]], images_dir: Path, api_key: str, args: argparse.Namespace,
                   request_dir: Optional[Path] = None) -> Tuple[int, int]:
    """Write a text_prompt into every node in ``nodes`` (one VLM call each, plus repairs). Returns (done, failed)."""
    color = BOX_COLORS[str(args.box_color)]
    leg_classes = {str(c).lower() for c in (args.leg_classes or [])}
    separate_classes = [str(c).lower() for c in (args.separate_classes or [])]
    quiet = bool(getattr(args, "quiet", False))
    n_done = 0
    n_failed = 0
    for node in nodes:
        node_id = int(node["id"])
        class_name = str(node.get("class_name", "") or "object")
        class_locked = bool(node.get("class_name_locked", False))
        correction_classes = [] if class_locked else separate_classes
        references = list(node.get("vlm_reference") or [])

        if bool(args.skip_existing) and str(node.get("text_prompt", "")).strip():
            if not quiet:
                print(f"[node {node_id:>3}] {class_name}: text_prompt already set, skipped")
            continue
        if not references:
            print(f"[node {node_id:>3}] {class_name}: no vlm_reference, skipped", file=sys.stderr)
            continue

        marked: List[Image.Image] = []
        used_names: List[str] = []
        for view_i, ref in enumerate(references, start=1):
            image_name = str(ref.get("image_name", ""))
            bbox = [int(v) for v in (ref.get("bbox_xyxy") or [])]
            image_path = _find_image(images_dir, image_name)
            if image_path is None:
                print(f"[warn] node {node_id}: image not found: {image_name}", file=sys.stderr)
                continue
            view = _mark_view(
                image_path,
                bbox,
                color=color,
                box_width=int(args.box_width),
                max_side=int(args.max_image_side),
            )
            marked.append(view)
            used_names.append(image_name)
            if request_dir is not None:
                node_dir = request_dir / f"node_{node_id:03d}_{class_name}"
                node_dir.mkdir(parents=True, exist_ok=True)
                view.save(node_dir / f"view{view_i}_{Path(image_name).stem}.jpg", quality=int(args.jpeg_quality))

        if not marked:
            print(f"[node {node_id:>3}] {class_name}: no usable views, skipped", file=sys.stderr)
            n_failed += 1
            continue

        instruction = _build_instruction(
            class_name,
            len(marked),
            str(args.box_color),
            int(args.max_words),
            describe_legs=class_name.lower() in leg_classes,
            separate_classes=separate_classes,
            allow_class_correction=not class_locked,
        )
        data_urls = [_to_data_url(im, int(args.jpeg_quality)) for im in marked]
        messages = _build_messages(instruction, data_urls)

        payload_kb = sum(len(u) for u in data_urls) / 1024.0
        if not quiet:
            print(
                f"[node {node_id:>3}] {class_name:<8} views={len(marked)} "
                f"({', '.join(used_names)})  upload={payload_kb:.0f}KB"
            )

        if bool(args.dry_run):
            if node is nodes[0]:
                print("---- instruction ----")
                print(instruction)
                print("---------------------")
            continue

        try:
            raw = _call_openrouter(
                api_key=api_key,
                model=str(args.model),
                messages=messages,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                timeout=float(args.timeout),
                max_retries=int(args.max_retries),
            )
        except Exception as exc:
            print(f"[fail] node {node_id}: {exc}", file=sys.stderr)
            n_failed += 1
            continue

        effective_class, phrase, warnings = _resolve_answer(
            raw, class_name, correction_classes, int(args.max_words)
        )

        # The prompt already forbids foreign categories and negation, but models still slip in
        # "wall-mounted" or "with no legs"; rewrite while the images are still in the conversation.
        for _ in range(int(args.max_repair)):
            issues = _violations(phrase, effective_class, separate_classes) if phrase else []
            if not issues:
                break
            if not quiet:
                print(f"           [repair] {phrase!r}: {'; '.join(issues)}")
            listed = "\n".join(f"- {i}" for i in issues)
            convo = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Your phrase was \"{phrase}\". It has to be rewritten because:\n"
                        f"{listed}\n"
                        f"Rewrite it so that \"{effective_class}\" is the only category word, "
                        f"describing the {effective_class} on its own through its colour, "
                        f"material, shape, its own parts and its own legs or base. Do not say "
                        f"what holds it up, what is around it, or what it lacks. If it has no "
                        f"support of its own, just end the phrase after the shape. Reply with "
                        f"the same JSON object and nothing else."
                    ),
                },
            ]
            try:
                raw = _call_openrouter(
                    api_key=api_key,
                    model=str(args.model),
                    messages=convo,
                    temperature=float(args.temperature),
                    max_tokens=int(args.max_tokens),
                    timeout=float(args.timeout),
                    max_retries=int(args.max_retries),
                )
            except Exception as exc:
                print(f"[warn] node {node_id}: repair call failed: {exc}", file=sys.stderr)
                break
            effective_class, phrase, extra = _resolve_answer(
                raw, class_name, correction_classes, int(args.max_words)
            )
            warnings += extra

        if not phrase:
            print(f"[fail] node {node_id}: empty phrase from raw={raw!r}", file=sys.stderr)
            n_failed += 1
            continue

        if NEGATION_RE.search(phrase.lower()):
            stripped = NEGATED_CLAUSE_RE.sub("", phrase).strip().rstrip(",")
            if stripped and not NEGATION_RE.search(stripped.lower()):
                warnings.append(f"dropped negated clause, was {phrase!r}")
                phrase = stripped

        leftover = _violations(phrase, effective_class, separate_classes)
        if leftover:
            warnings.append(f"unresolved after {args.max_repair} repairs: {'; '.join(leftover)}")

        if effective_class != class_name:
            node["class_name_original"] = class_name
            node["class_name"] = effective_class
            if not quiet:
                print(f'           [class] "{class_name}" -> "{effective_class}" (per model)')

        node["text_prompt"] = phrase
        n_done += 1
        n_tokens = _clip_token_count(phrase)
        token_note = f", clip_tokens={n_tokens}" if n_tokens is not None else ""
        if not quiet:
            print(f'           -> "{phrase}" ({len(phrase.split())} words{token_note})')
            if raw.strip().lower() != phrase:
                print(f"           raw: {raw.strip()!r}")
            for w in warnings:
                print(f"           [note] {w}")

        if float(args.sleep_between) > 0:
            time.sleep(float(args.sleep_between))

    return n_done, n_failed


# ---------------------------------------------------------------------------
# Class review: one class per node, with few multi-node frames instead of
# 3 views x N nodes.
#
# 1. Nodes whose SAM3 label is already confident (many agreeing views) keep that
#    class; no VLM call.
# 2. The remaining nodes (none / low-agreement SAM3) are covered by a small set
#    of frames chosen greedily so every such node is visible >= --cover times.
#    One VLM call per frame: all of that frame's nodes are drawn as numbered
#    colored regions and classified together.
# 3. Per-node classes are the confidence-weighted vote over frames. Only nodes
#    whose vote is split or low-confidence are escalated to one per-node call
#    (isolated crop).
# 4. Every node that ends up in a target class gets its <= --max-words
#    text_prompt (boxed views, one call each).
#
# No GT labels are read; geometry and group IDs stay fixed.
# ---------------------------------------------------------------------------

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]
CLASSES = "wall, floor, ceiling, bed, cabinet, chair, curtain, desk, door, monitor, picture, sofa, window, none"

TAXONOMY = '''Allowed classes: ''' + CLASSES + '''.
Taxonomy: cabinet fronts, cabinet doors, glass cupboard doors, drawers, file cabinets and shelving/storage units are CABINET, not room DOOR. DOOR means an architectural passage door. BED is a sleeping surface (mattress, sheets, duvet, headboard, bed frame); SOFA has an upright backrest and armrests for sitting. A floor patch is FLOOR even when fragmented, shadowed or partly occluded. Framed art and photographs are PICTURE; the computer display is MONITOR. Books, vases, pillows, clocks and unrelated clutter are NONE if they are the highlighted object. A furniture fragment can take its owning furniture class if its identity is clear. Use NONE when it is not any allowed category, mixed unrelated objects, or too uncertain. Do not use scene ground-truth labels or assume a benchmark label.'''


def frame_instruction(ids):
    return ('Classify several highlighted 3D regions seen in ONE photo. You get two images: (1) the clean photo, (2) the same photo with each region '
            'filled with a distinct color and a white number tag (' + ', '.join(str(i) for i in ids) + '). Classify every numbered region by the physical '
            'object or surface under its colored pixels, not by a neighbouring object; a region covering only part of an object is classified as that object. '
            'Judge each region independently.\n' + TAXONOMY + '\nReturn JSON only: {"regions":[{"id":1,"class":"allowed class","confidence":0.0,"reason":"short visible evidence"}]} '
            'with one entry per number. Confidence is between 0 and 1.')


def node_instruction():
    return ('Classify the SAME highlighted 3D region across these RGB views. Each view has TWO images: (1) context with a green outline and faint green highlighted pixels, '
            '(2) ONLY the region pixels in original RGB, with every other pixel replaced by neutral gray. The second image is the exact region to classify. '
            'Identify the physical object or surface represented by the retained RGB pixels, not the most salient neighbour.\n' + TAXONOMY +
            '\nReturn JSON only: {"class":"allowed class", "confidence":0.0, "reason":"short visible evidence and any ambiguity"}.')


def _open_img(path):
    """Open an image, tolerating an extension-case mismatch (.jpg vs .JPG)."""
    from pathlib import Path as _P
    path = _P(path)
    if not path.exists():
        for ext in ('.JPG', '.jpg', '.JPEG', '.jpeg', '.PNG', '.png'):
            if path.with_suffix(ext).exists():
                path = path.with_suffix(ext); break
    return Image.open(path)

def update_graph_categories(graph, target_classes):
    """Set every node's `target` flag from its class."""
    targets = set(target_classes)
    for node in graph['nodes']:
        node['target'] = node.get('class_name') in targets


def choose_views(candidates, count):
    selected=[];pool=sorted(candidates,key=lambda c:-c['score'])[:80]
    while pool and len(selected)<count:
        def score(c):
            diversity=min((np.linalg.norm(np.array(c['position'])-x['position'])for x in selected),default=0)
            return c['score']*(1+.2*min(diversity,2))
        best=max(pool,key=score);selected.append(best)
        others=[c for c in pool if abs(c['frame']-best['frame'])>=20]
        if len(others)<count-len(selected):others=[c for c in pool if c is not best]
        pool=others
    return selected


def is_confident(node, a):
    v = node.get('sam3_vote') or {}
    return (v.get('label') not in (None, '', 'none') and int(v.get('n_vote_views', 0)) >= a.sam3_min_views
            and float(v.get('view_share', 0)) >= a.sam3_min_share and node.get('class_name') == v.get('label'))


def get_font(size):
    for f in ('DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()


def select_frames(pools, review_ids, cover, max_frames, max_nodes_per_frame, min_pixels):
    """Greedy set cover: every review node visible >= cover times, <= max_nodes_per_frame per frame."""
    frames = {}
    for nid in review_ids:
        for c in pools.get(nid, []):
            if c['pixels'] >= min_pixels:
                frames.setdefault(c['image'], {})[nid] = c
    need = {nid: min(cover, sum(1 for f in frames.values() if nid in f)) for nid in review_ids}
    chosen = []
    while len(chosen) < max_frames and any(v > 0 for v in need.values()):
        best, best_gain, best_nodes = None, 0., None
        for img, nodes in frames.items():
            if img in [c[0] for c in chosen]:
                continue
            ranked = sorted(nodes.items(), key=lambda kv: (need[kv[0]] <= 0, -kv[1]['score']))[:max_nodes_per_frame]
            gain = sum(min(1.0, c['score'] / 3000.) for nid, c in ranked if need[nid] > 0)
            if gain > best_gain:
                best, best_gain, best_nodes = img, gain, dict(ranked)
        if best is None:
            break
        chosen.append((best, best_nodes))
        for nid in best_nodes:
            need[nid] = max(0, need[nid] - 1)
    return chosen


def draw_frame(original, region_nodes, order):
    """Overlay of numbered colored regions (image, ids). `order` is the list of node ids -> number 1..m."""
    arr = np.array(original).astype(np.float32)
    h, w = arr.shape[:2]
    tags = []
    for k, nid in enumerate(order, 1):
        c = region_nodes[nid]
        m = np.unpackbits(np.frombuffer(bytes.fromhex(c['mask']), dtype=np.uint8))[:c['width'] * c['height']].reshape(c['height'], c['width']).astype(bool)
        m = np.array(Image.fromarray(m).resize((w, h), Image.Resampling.NEAREST))
        col = np.array(PALETTE[(k - 1) % len(PALETTE)], np.float32)
        arr[m] = arr[m] * 0.45 + col * 0.55
        ys, xs = np.where(m)
        if len(xs):
            tags.append((k, int(np.median(xs)), int(np.median(ys))))
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img)
    font = get_font(max(18, h // 28))
    for k, x, y in tags:
        t = str(k)
        bb = d.textbbox((x, y), t, font=font, anchor='mm')
        d.rectangle((bb[0] - 4, bb[1] - 2, bb[2] + 4, bb[3] + 2), fill=(0, 0, 0))
        d.text((x, y), t, fill=(255, 255, 255), font=font, anchor='mm')
    return img


def parse_json(raw):
    m = re.search(r'\{.*\}', raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except ValueError:
        return None


def call_parsed(tries=3, **kw):
    """Call the VLM and re-ask when the reply is not valid JSON."""
    doc = None
    for _ in range(tries):
        doc = parse_json(_call_openrouter(**kw))
        if doc is not None:
            break
        kw['temperature'] = 0.3
    return doc


def main():
    threadpool_limits(limits=1, user_api='blas')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('iteration_dir', type=Path); p.add_argument('--source-path', type=Path, required=True)
    p.add_argument('--scene-graph', type=Path, required=True); p.add_argument('--out-json', type=Path, required=True)
    p.add_argument('--object-path', type=Path, required=True); p.add_argument('--images', default='images')
    p.add_argument('--resolution', type=int, default=4); p.add_argument('--num-views', type=int, default=3)
    p.add_argument('--model', default='google/gemini-3-flash-preview'); p.add_argument('--api-key-env', default='OPENROUTER_API_KEY')
    p.add_argument('--min-confidence', type=float, default=.75)
    p.add_argument('--target-classes', nargs='+', default=['cabinet', 'bed', 'chair', 'desk', 'sofa', 'monitor'])
    p.add_argument('--audit-dir', type=Path)
    p.add_argument('--sam3-min-views', type=int, default=5, help='SAM3 is confident only with >= this many valid views')
    p.add_argument('--sam3-min-share', type=float, default=0.8, help='...and >= this share of valid views agreeing on the label')
    p.add_argument('--cover', type=int, default=2, help='each reviewed node should appear in >= this many frames')
    p.add_argument('--max-frames', type=int, default=20); p.add_argument('--max-nodes-per-frame', type=int, default=8)
    p.add_argument('--min-region-pixels', type=int, default=120)
    p.add_argument('--vote-min-share', type=float, default=0.6, help='weighted vote share needed to accept without escalation')
    p.add_argument('--max-escalations', type=int, default=20); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--box-color', default='lime'); p.add_argument('--max-image-side', type=int, default=1024)
    p.add_argument('--max-words', type=int, default=10); p.add_argument('--temperature', type=float, default=0.)
    p.add_argument('--max-repair', type=int, default=2); p.add_argument('--sleep-between', type=float, default=0.)
    p.add_argument('--save-request-images', action='store_true')
    args = p.parse_args()
    key = os.environ.get(args.api_key_env, '').strip()
    if not key:
        raise SystemExit('Missing API key environment variable')
    graph = json.loads(args.scene_graph.read_text())
    r = args.iteration_dir.resolve(); model_root = r.parent.parent; src = args.source_path.resolve()
    audit = args.audit_dir or r / 'class_review'; audit.mkdir(parents=True, exist_ok=True); imdir = audit / 'images'; imdir.mkdir(exist_ok=True)
    nodes = graph['nodes']
    confident = {n['id'] for n in nodes if is_confident(n, args)}
    review_ids = [n['id'] for n in nodes if n['id'] not in confident]
    target_ids = {n['id'] for n in nodes if n.get('target')}
    print(f"[batched] {len(nodes)} nodes: {len(confident)} SAM3-confident (no VLM), {len(review_ids)} to review", flush=True)
    need_pool = set(review_ids) | target_ids | {n['id'] for n in nodes if n['id'] in confident and n['class_name'] in args.target_classes}

    xyzv = PlyData.read(str(r / 'point_cloud.ply'))['vertex'].data; xyz = np.column_stack([xyzv[a] for a in 'xyz'])
    f = read_object_features_from_ply(str(r / 'point_cloud.ply'), 16); cb = load_codebook(str(r / 'classifier.pth'), 16).detach().cpu().numpy(); gids = compute_class_ids_np(f, cb)
    indices = {}
    for n in nodes:
        if n['id'] not in need_pool:
            continue
        b = n.get('bbox_tight'); mask = gids == int(n['group_id'])
        if b:
            for j, a in enumerate('xyz'):
                mask &= (xyz[:, j] >= b[a + 'min']) & (xyz[:, j] <= b[a + 'max'])
        indices[n['id']] = np.flatnonzero(mask)
    gauss, ct, cameras, pipeline = _build_scene(model_root=model_root, source_path=str(src), object_path=str(args.object_path), render_point_cloud_path=r / 'point_cloud.ply',
                                                render_classifier_path=r / 'classifier.pth', images=args.images, resolution=args.resolution, num_objects=16)
    meta = {c['img_name']: c for c in json.loads((model_root / 'cameras.json').read_text())}
    pools = {nid: [] for nid in need_pool}; bg = torch.zeros(3, device='cuda')
    for vi, cam in enumerate(cameras):
        stem = Path(cam.image_name).stem; m = meta[stem]
        with torch.no_grad():
            obj = render(cam, gauss, pipeline, bg)['render_object']; _, h, w = obj.shape; pred = (ct @ obj.reshape(16, -1)).argmax(0).reshape(h, w).cpu().numpy()
        cp = (xyz - np.array(m['position'])) @ np.array(m['rotation']); z = cp[:, 2]
        uv = cp[:, :2] / np.maximum(z[:, None], .001) * [m['fx'] * w / m['width'], m['fy'] * h / m['height']] + [w / 2, h / 2]
        for n in nodes:
            nid = n['id']
            if nid not in need_pool:
                continue
            ids = indices[nid]; visible = ids[(z[ids] > .1) & (uv[ids, 0] >= 0) & (uv[ids, 0] < w) & (uv[ids, 1] >= 0) & (uv[ids, 1] < h)]
            if len(visible) < 8:
                continue
            pts = uv[visible]; lo = np.floor(np.quantile(pts, .01, axis=0) - 2).astype(int); hi = np.ceil(np.quantile(pts, .99, axis=0) + 2).astype(int)
            x0, y0 = np.maximum(lo, [0, 0]); x1, y1 = np.minimum(hi, [w - 1, h - 1])
            mask = pred == int(n['group_id']); mask[:y0] = False; mask[y1 + 1:] = False; mask[:, :x0] = False; mask[:, x1 + 1:] = False
            yy, xx = np.where(mask); area = len(xx)
            if area < 24:
                continue
            x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max()); edge = min(x0, y0, w - 1 - x1, h - 1 - y1); score = area * (.7 + .3 * min(edge / 15, 1)); pool = pools[nid]
            if len(pool) >= 100 and score <= min(c['score'] for c in pool):
                continue
            pool.append(dict(image=stem + '.jpg', bbox=[x0, y0, x1, y1], width=w, height=h, score=score, pixels=area, frame=int(stem) if stem.isdigit() else vi,
                             position=m['position'], mask=np.packbits(mask).tobytes().hex()))
            if len(pool) > 100:
                pool.sort(key=lambda c: -c['score']); pool.pop()
        if vi % 200 == 0:
            print(f'[batched views] {vi + 1}/{len(cameras)}', flush=True)

    calls = dict(frame=0, escalation=0)
    # ---- batched frame calls ----
    chosen = select_frames(pools, review_ids, args.cover, args.max_frames, args.max_nodes_per_frame, args.min_region_pixels)
    covered = {nid for _, d in chosen for nid in d}
    print(f"[batched] {len(chosen)} frames cover {len(covered)}/{len(review_ids)} review nodes", flush=True)

    def call_frame(item):
        idx, (img_name, region_nodes) = item
        order = list(region_nodes)
        original = _open_img(src / args.images / img_name).convert('RGB')
        overlay = draw_frame(original, region_nodes, order)
        clean = original.copy(); clean.thumbnail((1024, 1024)); overlay.thumbnail((1024, 1024)); overlay.save(imdir / f'frame_{idx:02d}_{img_name}')
        content = [{'type': 'text', 'text': frame_instruction(list(range(1, len(order) + 1)))}, {'type': 'text', 'text': 'Image 1: clean photo.'},
                   {'type': 'image_url', 'image_url': {'url': _to_data_url(clean, 85)}}, {'type': 'text', 'text': 'Image 2: regions highlighted and numbered.'},
                   {'type': 'image_url', 'image_url': {'url': _to_data_url(overlay, 85)}}]
        doc = call_parsed(api_key=key, model=args.model, messages=[{'role': 'user', 'content': content}], temperature=0., max_tokens=140 * len(order) + 100, timeout=180., max_retries=4) or {}
        out = []
        for reg in doc.get('regions', []):
            try:
                k = int(reg['id']); nid = order[k - 1]
            except Exception:
                continue
            out.append((nid, str(reg.get('class', '')).strip().lower(), float(reg.get('confidence', 0)), str(reg.get('reason', '')), img_name, region_nodes[nid]['pixels']))
        return out
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(call_frame, list(enumerate(chosen))))
    calls['frame'] = len(chosen)
    votes = {nid: [] for nid in review_ids}
    for res in results:
        for nid, cls, conf, reason, img, px in res:
            if cls in (*DEFAULT_SEPARATE_CLASSES, 'none'):
                votes[nid].append(dict(cls=cls, conf=conf, reason=reason, image=img, weight=conf * np.sqrt(px)))

    def tally(vs):
        score = {}
        for v in vs:
            score[v['cls']] = score.get(v['cls'], 0.) + v['weight']
        if not score:
            return None, 0., 0.
        top = max(score, key=score.get); share = score[top] / sum(score.values())
        mconf = float(np.average([v['conf'] for v in vs if v['cls'] == top], weights=[v['weight'] for v in vs if v['cls'] == top]))
        return top, share, mconf

    # ---- escalate split / weak / unseen nodes with one per-node call ----
    decided, escalate = {}, []
    for nid in review_ids:
        top, share, mconf = tally(votes[nid])
        if top is not None and share >= args.vote_min_share and mconf >= args.min_confidence:
            decided[nid] = dict(cls=top, conf=mconf, share=share, source='frame_vote', reason=next(v['reason'] for v in votes[nid] if v['cls'] == top))
        else:
            escalate.append((nid, top, share, mconf))
    escalate.sort(key=lambda t: t[2])
    skipped_esc = escalate[args.max_escalations:]; escalate = escalate[:args.max_escalations]
    print(f"[batched] {len(decided)} nodes decided by frame vote, {len(escalate)} escalated, {len(skipped_esc)} left provisional (cap)", flush=True)

    def call_node(item):
        nid, top, share, mconf = item
        sel = choose_views(pools.get(nid, []), args.num_views)
        if not sel:
            return nid, None
        content = [{'type': 'text', 'text': node_instruction()}]
        for rank, c in enumerate(sel):
            original = _open_img(src / args.images / c['image']).convert('RGB'); w, h = original.size
            scale = np.array([w / c['width'], h / c['height'], w / c['width'], h / c['height']]); bbox = np.rint(np.array(c['bbox']) * scale).astype(int).tolist()
            region = np.unpackbits(np.frombuffer(bytes.fromhex(c['mask']), dtype=np.uint8))[:c['width'] * c['height']].reshape(c['height'], c['width']).astype(bool)
            region = np.array(Image.fromarray(region).resize(original.size, Image.Resampling.NEAREST)); px = np.array(original); px[region] = (px[region] * .85 + np.array([0, 255, 0]) * .15).astype(np.uint8)
            marked = Image.fromarray(px); ImageDraw.Draw(marked).rectangle(bbox, outline='lime', width=4); marked.thumbnail((1024, 1024))
            pad = 12; iso = np.full((h, w, 3), 160, dtype=np.uint8); iso[region] = np.asarray(original)[region]
            crop = Image.fromarray(iso).crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(w, bbox[2] + pad), min(h, bbox[3] + pad))); crop.thumbnail((768, 768))
            for label, image in [('Context: green pixels identify this region.', marked), ('Classify ONLY these retained RGB pixels; gray is excluded context.', crop)]:
                content.append({'type': 'text', 'text': label}); content.append({'type': 'image_url', 'image_url': {'url': _to_data_url(image, 85)}})
        return nid, call_parsed(api_key=key, model=args.model, messages=[{'role': 'user', 'content': content}], temperature=0., max_tokens=220, timeout=120., max_retries=4)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        esc_results = list(ex.map(call_node, escalate))
    calls['escalation'] = len(escalate)
    esc_prev = {nid: (top, share, mconf) for nid, top, share, mconf in escalate}
    for nid, ans in esc_results:
        if ans and str(ans.get('class', '')).strip().lower() in (*DEFAULT_SEPARATE_CLASSES, 'none'):
            decided[nid] = dict(cls=str(ans['class']).strip().lower(), conf=float(ans.get('confidence', 0)), share=1., source='escalation', reason=str(ans.get('reason', '')),
                                frame_vote=esc_prev[nid][0])

    # ---- write back ----
    results_log = []
    for n in nodes:
        nid = n['id']; old = n['class_name']
        vote_rec = n.get('sam3_vote')
        if nid in confident:
            rec = dict(initial_class=old, vlm_class=None, confidence=float(vote_rec.get('view_share', 0)), accepted=True, source='sam3_confident', final_class=old,
                       reason=f"SAM3: {vote_rec.get('view_votes')}/{vote_rec.get('n_vote_views')} views agree on {old}")
        else:
            d = decided.get(nid)
            accepted = bool(d) and d['conf'] >= args.min_confidence
            final = d['cls'] if accepted else old
            rec = dict(initial_class=old, vlm_class=d['cls'] if d else None, confidence=d['conf'] if d else 0., accepted=accepted, source=d['source'] if d else 'unreviewed',
                       vote_share=d['share'] if d else 0., frames_seen=len(votes.get(nid, [])), reason=d['reason'] if d else 'no usable frame / escalation result; SAM3 class kept',
                       final_class=final, model=args.model)
        if rec['final_class'] != old:
            n['class_name_original'] = old; n['class_name'] = rec['final_class']
        n['class_review'] = rec; n['class_name_locked'] = True; n.pop('text_prompt', None)
        results_log.append(dict(node_id=nid, group_id=n['group_id'], **rec))
        print(f"[review] node={nid} group={n['group_id']} {old} -> {rec['final_class']} ({rec['source']}, conf={rec['confidence']:.2f})", flush=True)
    update_graph_categories(graph, args.target_classes)
    # views for the text-prompt stage (target nodes only; no extra VLM call here)
    for n in nodes:
        if n.get('target'):
            sel = choose_views(pools.get(n['id'], []), args.num_views); refs = []
            for c in sel:
                original = _open_img(src / args.images / c['image']); w, h = original.size
                scale = np.array([w / c['width'], h / c['height'], w / c['width'], h / c['height']]); refs.append(dict(image_name=c['image'], bbox_xyxy=np.rint(np.array(c['bbox']) * scale).astype(int).tolist()))
            n['vlm_reference'] = refs
    # ---- one text_prompt per target node, written in the same stage (classes are final and locked here) ----
    desc_args = _parse_args(['--scene-graph', str(args.scene_graph), '--source-path', str(src), '--images', args.images, '--model', args.model,
                                   '--api-key-env', args.api_key_env, '--box-color', args.box_color, '--max-image-side', str(args.max_image_side),
                                   '--max-words', str(args.max_words), '--temperature', str(args.temperature), '--max-repair', str(args.max_repair),
                                   '--sleep-between', str(args.sleep_between), '--quiet'])
    request_dir = audit / 'vlm_request_images' if args.save_request_images else None
    if request_dir:
        request_dir.mkdir(exist_ok=True)
    described, failed = describe_nodes([n for n in nodes if n.get('target')], src / args.images, key, desc_args, request_dir)
    print(f"[describe] {described} text_prompts written, {failed} failed", flush=True)
    cost = dict(nodes=len(nodes), sam3_confident=len(confident), reviewed=len(review_ids), frame_calls=calls['frame'], escalation_calls=calls['escalation'],
                per_node_baseline_calls=len(nodes), images_sent=2 * calls['frame'] + 6 * calls['escalation'])
    (audit / 'review.json').write_text(json.dumps(results_log, indent=2) + '\n'); args.out_json.parent.mkdir(parents=True, exist_ok=True); args.out_json.write_text(json.dumps(graph, indent=2) + '\n')
    print(f"[batched] VLM calls: {calls['frame']} frame + {calls['escalation']} escalation = {calls['frame'] + calls['escalation']} "
          f"(per-node baseline would be {len(nodes)}); images sent ~{cost['images_sent']} (baseline ~{6 * len(nodes)}); {sum(n['target'] for n in nodes)} targets", flush=True)


if __name__ == '__main__':
    main()
