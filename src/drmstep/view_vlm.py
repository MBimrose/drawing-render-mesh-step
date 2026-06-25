"""Combined VLM + LocateAnything view extraction.

Two grounding models cooperate to find the isometric pictorial:

  1. LocateAnything-3B (in-process) detects all "3D view of the part" regions.
     Bboxes tend to be tight on actual ink content (the model is trained on
     bbox grounding, not language understanding).
  2. Qwen3-VL-235B (HTTP) describes the drawing's views and emits a coarse
     bbox for the isometric, distinguishing it from orthographic views.

Combination strategy:
  - If LocateAnything returns ≥1 plausible boxes, pick the one with maximum
    overlap with Qwen's bbox. Take the UNION of the two to ensure no clipping.
  - If LA returns nothing useful, fall back to Qwen's bbox alone.
  - If both fail, fall back to classical CV.

Final crop is padded 15% to absorb residual tightness.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Optional

import litellm
from PIL import Image

from .config import Config

logger = logging.getLogger(__name__)

# LocateAnything-3B is in-process; serialize calls to avoid:
#   1. Concurrent AutoModel.from_pretrained collisions during first load.
#   2. CUDA contention on a single GPU when multiple threads call generate().
_LA_LOCK = threading.Lock()

_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_ARRAY_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\]}]")
_MAX_EDGE_PX = 1536  # cap for the VLM call


@dataclass(frozen=True)
class VLMExtractResult:
    bbox_xyxy: Optional[tuple[int, int, int, int]]
    raw_response: str


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _downscale(image: Image.Image, max_edge: int) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_edge:
        return image
    scale = max_edge / max(w, h)
    return image.resize((int(w * scale), int(h * scale)))


def _parse_bbox(text: str) -> Optional[tuple[int, int, int, int]]:
    """Try several formats VLMs use for bboxes."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # JSON object with 'bbox' key
    for m in [text] + [x.group(0) for x in _JSON_OBJ_RE.finditer(text)]:
        try:
            data = json.loads(m)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            arr = data.get("bbox") or data.get("box") or data.get("isometric_bbox")
            if isinstance(arr, (list, tuple)) and len(arr) == 4:
                try:
                    x1, y1, x2, y2 = (int(round(float(v))) for v in arr)
                except (TypeError, ValueError):
                    continue
                if x2 > x1 and y2 > y1:
                    return (x1, y1, x2, y2)
            try:
                x1 = int(round(float(data["x1"])))
                y1 = int(round(float(data["y1"])))
                x2 = int(round(float(data["x2"])))
                y2 = int(round(float(data["y2"])))
            except (KeyError, TypeError, ValueError):
                continue
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)

    # Bare array
    m = _ARRAY_RE.search(text)
    if m:
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
    return None


_SYSTEM_DESCRIBE = (
    "You are an expert mechanical engineer who reads multi-view technical drawings. "
    "Be concise and factual. Do not guess about features you cannot see."
)

_USER_DESCRIBE = (
    "This is a mechanical engineering drawing of a single part. "
    "List the distinct views shown (e.g. top, front, side, section A-A, isometric pictorial, "
    "detail B, etc.). For the ISOMETRIC pictorial view (the shaded/lined 3D rendering of the "
    "whole part — NOT any orthographic projection, section, or detail), describe roughly "
    "where it is on the page (upper-left, upper-right, center, etc.) and its approximate size. "
    "If multiple 3D pictorial views exist, identify the largest single one of the whole part. "
    "Reply in 2-4 short sentences."
)

_SYSTEM_LOCATE = (
    "You are a precise visual-grounding model. Output ONE JSON object only, no prose. "
    "Schema: {\"bbox\": [x1, y1, x2, y2]} where coordinates are integer absolute pixel "
    "positions in the image. The bbox must tightly enclose ONLY the requested view, with "
    "no dimension callouts, no orthographic views, and no title block."
)


def _build_locate_prompt(w: int, h: int, description: str) -> str:
    return (
        f"The image is {w} pixels wide by {h} pixels tall.\n\n"
        f"Earlier description of the drawing's views:\n{description}\n\n"
        "Now return the bounding box of the isometric pictorial view of the whole part. "
        "If multiple 3D pictorial views exist, pick the LARGEST single one. "
        "IMPORTANT: be GENEROUS — the bbox MUST include EVERY pixel of the 3D pictorial "
        "(every fillet, every hole, every feature, every protrusion) plus a small margin "
        "of whitespace around it. It is MUCH worse to cut off part of the geometry than "
        "to include some extra blank space. Do not include other views or the title block. "
        "Reply with ONLY the JSON object: {\"bbox\": [x1, y1, x2, y2]}"
    )


def _build_fallback_prompt(w: int, h: int, description: str) -> str:
    """Used when no isometric exists — locate the most informative orthographic view."""
    return (
        f"The image is {w} pixels wide by {h} pixels tall.\n\n"
        f"Earlier description of the drawing's views:\n{description}\n\n"
        "No isometric pictorial view of the whole part is available. Instead, return the "
        "bounding box of the SINGLE most informative orthographic view — the one that shows "
        "the part's overall shape most clearly. Prefer the front or side view over top views, "
        "and prefer a full view over a section/detail view. "
        "IMPORTANT: be GENEROUS — the bbox MUST include every pixel of the geometry plus "
        "a small whitespace margin. Cutting off part of the part is MUCH worse than including "
        "extra blank space. Do not include other views or the title block. "
        "Reply with ONLY the JSON object: {\"bbox\": [x1, y1, x2, y2]}"
    )


def _locate_anything_boxes(
    image: Image.Image, config: Config
) -> list[tuple[int, int, int, int]]:
    """Run NVIDIA LocateAnything-3B on ``image`` and return all bboxes it finds
    that plausibly enclose a 3D view of a mechanical part. Coords are in the
    pixel space of ``image``.

    Returns an empty list on any failure (model load fail, no parse, no boxes).
    """
    try:
        from .view_extract import _call_locate_anything, _LA_BOX_BLOCK_RE, _LA_NUM_RE
    except Exception as exc:
        logger.warning("LocateAnything backend unavailable: %s", exc)
        return []

    try:
        with _LA_LOCK:
            _, raw = _call_locate_anything(image, config)
    except Exception as exc:
        logger.warning("LocateAnything call failed: %s", exc)
        return []
    if not raw:
        return []

    w, h = image.size
    boxes: list[tuple[int, int, int, int]] = []
    for block in _LA_BOX_BLOCK_RE.findall(raw):
        nums = _LA_NUM_RE.findall(block)
        if len(nums) < 4:
            continue
        try:
            x1n, y1n, x2n, y2n = (float(n) for n in nums[:4])
        except ValueError:
            continue
        if max(x1n, y1n, x2n, y2n) > 1000:
            continue
        x1 = int(x1n / 1000.0 * w)
        y1 = int(y1n / 1000.0 * h)
        x2 = int(x2n / 1000.0 * w)
        y2 = int(y2n / 1000.0 * h)
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1) / float(w * h)
        if area < 0.005 or area > 0.85:  # skip the "I dunno" full-image box
            continue
        boxes.append((x1, y1, x2, y2))
    # dedupe while preserving order
    seen = set()
    unique: list[tuple[int, int, int, int]] = []
    for b in boxes:
        if b not in seen:
            seen.add(b)
            unique.append(b)
    return unique


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / float(a_area + b_area - inter)


def _union(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def extract_isometric_bbox(image: Image.Image, config: Config) -> VLMExtractResult:
    """Combined LocateAnything + Qwen3-VL extraction.

    Returns bbox in **original-image pixel coordinates**, or None on failure.
    """
    full = image.convert("RGB")
    w_full, h_full = full.size
    small = _downscale(full, _MAX_EDGE_PX)
    w_small, h_small = small.size

    img_url = _image_to_data_url(small)

    # LocateAnything-3B detects all "3D view of the part" regions. Boxes are in
    # full-image pixel coords; LA tends to crop tightly to ink content.
    la_boxes_full = _locate_anything_boxes(full, config)
    # Pre-rank by area so the largest comes first.
    la_boxes_full.sort(
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True
    )
    logger.info("LocateAnything boxes (largest first): %s", la_boxes_full[:5])

    # Pass 1: describe
    description = ""
    try:
        resp = litellm.completion(
            model=config.vlm_model,
            api_base=config.vlm_url,
            api_key=config.vlm_api_key,
            temperature=0.0,
            max_tokens=350,
            messages=[
                {"role": "system", "content": _SYSTEM_DESCRIBE},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": _USER_DESCRIBE},
                    ],
                },
            ],
        )
        description = (resp.choices[0].message.content or "").strip()
        logger.info("VLM describe: %s", description[:200])
    except Exception as exc:
        logger.warning("VLM describe failed: %s", exc)

    def _locate(prompt_builder, label: str) -> tuple[Optional[tuple[int, int, int, int]], str]:
        try:
            resp = litellm.completion(
                model=config.vlm_model,
                api_base=config.vlm_url,
                api_key=config.vlm_api_key,
                temperature=0.0,
                max_tokens=150,
                messages=[
                    {"role": "system", "content": _SYSTEM_LOCATE},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_url}},
                            {"type": "text", "text": prompt_builder(w_small, h_small, description)},
                        ],
                    },
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            logger.info("VLM %s response: %s", label, text[:200])
            return _parse_bbox(text), text
        except Exception as exc:
            logger.warning("VLM %s failed: %s", label, exc)
            return None, ""

    # Pass 2a: locate the isometric pictorial.
    bbox_small, raw = _locate(_build_locate_prompt, "locate-iso")

    # Pass 2b: if the model returned no bbox (or a degenerate one), the drawing
    # may not contain an isometric at all. Ask for the best orthographic view
    # so we still have geometry to feed Hunyuan3D.
    def _is_degenerate(b: Optional[tuple[int, int, int, int]]) -> bool:
        if b is None:
            return True
        x1, y1, x2, y2 = b
        return x2 <= x1 or y2 <= y1

    if _is_degenerate(bbox_small):
        logger.info("VLM found no isometric; falling back to best orthographic view")
        bbox_small_fb, raw_fb = _locate(_build_fallback_prompt, "locate-ortho-fallback")
        raw = raw + "\n--- fallback ---\n" + raw_fb
        if not _is_degenerate(bbox_small_fb):
            bbox_small = bbox_small_fb

    qwen_bbox_full: Optional[tuple[int, int, int, int]] = None
    if bbox_small is not None:
        # Qwen3-VL natively returns bboxes in 0–1000 normalized coords regardless
        # of prompt wording. Detect and rescale to small-image pixels.
        x1_raw, y1_raw, x2_raw, y2_raw = bbox_small
        if (max(x1_raw, y1_raw, x2_raw, y2_raw) <= 1000
                and (w_small > 1000 or h_small > 1000)):
            logger.info("Qwen bbox looks normalized 0-1000; converting to pixels")
            x1 = int(round(x1_raw / 1000.0 * w_small))
            y1 = int(round(y1_raw / 1000.0 * h_small))
            x2 = int(round(x2_raw / 1000.0 * w_small))
            y2 = int(round(y2_raw / 1000.0 * h_small))
        else:
            x1, y1, x2, y2 = x1_raw, y1_raw, x2_raw, y2_raw
        x1 = max(0, min(w_small, x1))
        y1 = max(0, min(h_small, y1))
        x2 = max(0, min(w_small, x2))
        y2 = max(0, min(h_small, y2))
        if x2 > x1 and y2 > y1:
            area_frac = ((x2 - x1) * (y2 - y1)) / float(w_small * h_small)
            if 0.005 <= area_frac <= 0.85:
                sx, sy = w_full / w_small, h_full / h_small
                qwen_bbox_full = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

    # Combination strategy:
    #   - If both LA and Qwen produced boxes, pick the LA box with highest IoU
    #     vs Qwen's bbox, then take the UNION so neither model's tightness wins.
    #   - If only LA produced boxes, pick its largest.
    #   - If only Qwen produced a bbox, use it.
    #   - Otherwise fail.
    chosen: Optional[tuple[int, int, int, int]] = None
    if la_boxes_full and qwen_bbox_full is not None:
        best_la = max(la_boxes_full, key=lambda b: _iou(b, qwen_bbox_full))
        if _iou(best_la, qwen_bbox_full) > 0.1:
            chosen = _union(best_la, qwen_bbox_full)
            logger.info("Combined: LA=%s ∪ Qwen=%s → %s", best_la, qwen_bbox_full, chosen)
        else:
            # No overlap → trust LA's largest box (it found the part; Qwen may be off-target)
            chosen = la_boxes_full[0]
            logger.info("LA and Qwen disagree; using largest LA box: %s", chosen)
    elif la_boxes_full:
        chosen = la_boxes_full[0]
        logger.info("Only LA found boxes; using largest: %s", chosen)
    elif qwen_bbox_full is not None:
        chosen = qwen_bbox_full
        logger.info("Only Qwen produced a bbox; using it: %s", chosen)

    if chosen is None:
        return VLMExtractResult(bbox_xyxy=None,
                                raw_response=description + "\n---\n" + raw)

    # Apply 15% padding so we don't clip features from either model's bbox.
    pad = 0.15
    fx1, fy1, fx2, fy2 = chosen
    pad_x = int((fx2 - fx1) * pad)
    pad_y = int((fy2 - fy1) * pad)
    padded = (
        max(0, fx1 - pad_x),
        max(0, fy1 - pad_y),
        min(w_full, fx2 + pad_x),
        min(h_full, fy2 + pad_y),
    )

    return VLMExtractResult(bbox_xyxy=padded, raw_response=description + "\n---\n" + raw)
