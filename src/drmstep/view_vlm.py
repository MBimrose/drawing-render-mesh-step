"""VLM-driven view extraction.

Asks the configured VLM (default: Qwen3-VL-235B via litellm) to identify the
isometric / 3D pictorial view of the part and return its bounding box in pixel
coordinates. Uses a two-pass strategy:

  Pass 1 (describe): "List all the views in this drawing." Forces the model to
    enumerate views and disambiguate isometric from orthographic.
  Pass 2 (locate): "Return the bbox of the isometric view as [x1,y1,x2,y2]."
    Conditioned on the pass-1 description so the model commits to one view.

If the model returns a degenerate or out-of-range bbox, we fall back to the
classical CV extractor as a safety net.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import litellm
from PIL import Image

from .config import Config

logger = logging.getLogger(__name__)

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


def extract_isometric_bbox(image: Image.Image, config: Config) -> VLMExtractResult:
    """Two-pass VLM extraction: describe views, then locate isometric bbox.

    Returns bbox in **original-image pixel coordinates**, or None on failure.
    """
    full = image.convert("RGB")
    w_full, h_full = full.size
    small = _downscale(full, _MAX_EDGE_PX)
    w_small, h_small = small.size

    img_url = _image_to_data_url(small)

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

    if bbox_small is None:
        return VLMExtractResult(bbox_xyxy=None, raw_response=description + "\n---\n" + raw)

    # Qwen3-VL natively returns bboxes in 0–1000 normalized coordinates regardless
    # of what we asked for in the prompt. Detect this by checking whether all
    # coords ≤ 1000 (and the image is bigger than 1000 in either dim — otherwise
    # they could legitimately be pixel coords).
    x1_raw, y1_raw, x2_raw, y2_raw = bbox_small
    if (max(x1_raw, y1_raw, x2_raw, y2_raw) <= 1000
            and (w_small > 1000 or h_small > 1000)):
        logger.info("VLM bbox looks normalized 0-1000; converting to pixels")
        x1 = int(round(x1_raw / 1000.0 * w_small))
        y1 = int(round(y1_raw / 1000.0 * h_small))
        x2 = int(round(x2_raw / 1000.0 * w_small))
        y2 = int(round(y2_raw / 1000.0 * h_small))
    else:
        x1, y1, x2, y2 = x1_raw, y1_raw, x2_raw, y2_raw

    # Validate: bbox must be inside image bounds, non-degenerate, and not the entire image.
    x1 = max(0, min(w_small, x1))
    y1 = max(0, min(h_small, y1))
    x2 = max(0, min(w_small, x2))
    y2 = max(0, min(h_small, y2))
    if x2 <= x1 or y2 <= y1:
        return VLMExtractResult(bbox_xyxy=None, raw_response=description + "\n---\n" + raw)
    area_frac = ((x2 - x1) * (y2 - y1)) / float(w_small * h_small)
    if area_frac < 0.005:
        logger.info("VLM bbox too small (%.3f); rejecting", area_frac)
        return VLMExtractResult(bbox_xyxy=None, raw_response=description + "\n---\n" + raw)
    if area_frac > 0.85:
        logger.info("VLM bbox covers whole page (%.3f); rejecting", area_frac)
        return VLMExtractResult(bbox_xyxy=None, raw_response=description + "\n---\n" + raw)

    # Map small-image bbox back to full-res coords.
    sx, sy = w_full / w_small, h_full / h_small
    bbox_full = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

    # Apply generous 20% padding — Qwen3-VL's bboxes are consistently too tight
    # on isometric views, often clipping the part's outermost features
    # (fillets, edges, protrusions). 20% recovers them at the cost of some
    # extra whitespace, which Hunyuan3D handles fine.
    pad = 0.20
    fx1, fy1, fx2, fy2 = bbox_full
    pad_x = int((fx2 - fx1) * pad)
    pad_y = int((fy2 - fy1) * pad)
    padded = (
        max(0, fx1 - pad_x),
        max(0, fy1 - pad_y),
        min(w_full, fx2 + pad_x),
        min(h_full, fy2 + pad_y),
    )

    return VLMExtractResult(bbox_xyxy=padded, raw_response=description + "\n---\n" + raw)
