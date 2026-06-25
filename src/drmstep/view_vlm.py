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

import cv2
import litellm
import numpy as np
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
    cleaned_crop: Optional[Image.Image] = None


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
    "whole part — NOT any orthographic projection, section, detail view, OR DIMENSION TABLE "
    "containing rows of numerical values), describe roughly where it is on the page "
    "(upper-left, upper-right, center, etc.) and its approximate size. "
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
        "to include some extra blank space. "
        "DO NOT return the bbox of: orthographic views, section views, detail views, the "
        "title block, OR any dimension table (a rectangular grid of numerical values). "
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


def _clean_outside_hull(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Crop ``image`` to ``bbox`` and white out ink OUTSIDE the convex hull of the
    largest part-shaped blob found inside. Bbox dimensions are preserved (no
    tightening); we only erase the ink that doesn't belong to the part.
    """
    w_full, h_full = image.size
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_full, x2), min(h_full, y2)
    if x2 <= x1 or y2 <= y1:
        return image.crop(bbox)
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    cw, ch = crop.size

    gray = np.array(crop.convert("L"))
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    close_k = max(5, int(min(cw, ch) * 0.012))
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k)),
    )
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num <= 1:
        return crop
    label = _pick_part_blob(stats, cw, ch)
    if label == 0:
        return crop

    component_mask = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    interior_mask = np.zeros_like(component_mask)
    if contours:
        hull = cv2.convexHull(np.concatenate(contours))
        cv2.drawContours(interior_mask, [hull], -1, 255, thickness=-1)
        # Dilate the hull ~2% of the crop's smaller dim — keeps the part's
        # outermost outline and a small whitespace margin, so the iso never
        # looks aggressively clipped at the edge.
        dilate_k = max(7, int(min(cw, ch) * 0.02))
        interior_mask = cv2.dilate(
            interior_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)),
        )
    else:
        interior_mask = component_mask

    arr = np.array(crop)
    outside_hull = (interior_mask == 0) & (binary > 0)
    arr[outside_hull] = (255, 255, 255)
    return Image.fromarray(arr)


def _pick_part_blob(stats: np.ndarray, cw: int, ch: int) -> int:
    """Pick the connected-component label that is most likely the part.

    Largest by area, but with hard filters against:
      - ruler-letter columns (tall+thin hugging an edge)
      - dimension tables (huge rectangular grids → most of the bbox is ink
        with a very regular shape; not characteristic of a 3D pictorial)
    Returns the chosen label (1-indexed) or 0 if no plausible blob.
    """
    page_area = float(cw * ch)
    best_label = 0
    best_area = -1
    for lbl in range(1, stats.shape[0]):
        bx = int(stats[lbl, cv2.CC_STAT_LEFT])
        by = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area / page_area < 0.005:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        # Tall, thin column hugging an edge → page ruler letters.
        if aspect > 4.0 and bw / cw < 0.20 and (bx + bw) / cw > 0.85:
            continue
        if aspect > 4.0 and bw / cw < 0.20 and bx / cw < 0.15:
            continue
        # Dimension tables: very high ink-density (ratio of component pixels
        # to bbox area). A 3D pictorial has many thin lines → low density.
        # A grid table has cells densely filled with text → high density.
        bbox_area = bw * bh
        if bbox_area > 0:
            density = area / float(bbox_area)
            if density > 0.35 and bbox_area / page_area > 0.15:
                continue
        if area > best_area:
            best_area = area
            best_label = lbl
    return best_label


def _tighten_to_largest_blob(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    pad_frac: float = 0.08,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop ``image`` to ``bbox``, then tighten to the part-shaped connected
    component (after morphological closing to bond hidden-line gaps).

    Whites out ONLY ink that falls OUTSIDE the convex hull of the chosen
    component. Internal dimension lines, hole indicators, and feature
    annotations that live inside the part outline are preserved — Hunyuan3D
    sees the part's full contour with all of its detail intact.

    Returns: (cleaned_crop, bbox_in_full_image_coords)
    """
    w_full, h_full = image.size
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_full, x2), min(h_full, y2)
    if x2 <= x1 or y2 <= y1:
        return image.crop(bbox), bbox

    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    cw, ch = crop.size

    # Binarize: ink = 255, paper = 0.
    gray = np.array(crop.convert("L"))
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Closing: bond hidden-line gaps + thin disconnections WITHIN the part.
    # Kernel size ~1% of crop's smaller dim — large enough to bridge the part's
    # internal gaps, small enough to NOT bond it to adjacent ruler labels.
    close_k = max(5, int(min(cw, ch) * 0.012))
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k)),
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num <= 1:
        return crop, (x1, y1, x2, y2)

    largest_label = _pick_part_blob(stats, cw, ch)
    if largest_label == 0:
        return crop, (x1, y1, x2, y2)

    bx = int(stats[largest_label, cv2.CC_STAT_LEFT])
    by = int(stats[largest_label, cv2.CC_STAT_TOP])
    bw = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    bh = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
    bx2, by2 = bx + bw, by + bh

    # Build a "part interior" mask: the convex hull of the chosen component,
    # filled. Anything inside the hull is preserved as-is; only ink OUTSIDE
    # the hull gets erased. This keeps internal dimension lines, leader
    # arrows, hole indicators, and annotations that live within the part
    # contour — the user can still see all the part detail.
    component_mask = (labels == largest_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    interior_mask = np.zeros_like(component_mask)
    if contours:
        # Hull over ALL the contour points so we cover the part's full envelope
        # even if internal gaps split it into nested pieces.
        all_pts = np.concatenate(contours)
        hull = cv2.convexHull(all_pts)
        cv2.drawContours(interior_mask, [hull], -1, 255, thickness=-1)
        # Dilate a few px so the hull boundary itself stays whole.
        dilate_k = max(3, int(min(cw, ch) * 0.005))
        interior_mask = cv2.dilate(
            interior_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k)),
        )
    else:
        interior_mask = component_mask

    arr = np.array(crop)
    outside_hull = (interior_mask == 0) & (binary > 0)
    arr[outside_hull] = (255, 255, 255)
    cleaned = Image.fromarray(arr)

    # Tighten to the component's bbox + small padding.
    pad_x = int(bw * pad_frac)
    pad_y = int(bh * pad_frac)
    tight = (
        max(0, bx - pad_x),
        max(0, by - pad_y),
        min(cw, bx2 + pad_x),
        min(ch, by2 + pad_y),
    )
    tight_crop = cleaned.crop(tight)

    # Map tight crop's bbox back to full-image coordinates.
    full_bbox = (
        x1 + tight[0],
        y1 + tight[1],
        x1 + tight[2],
        y1 + tight[3],
    )
    return tight_crop, full_bbox


def _trim_page_frame(image: Image.Image, margin_frac: float = 0.05) -> tuple[Image.Image, tuple[int, int]]:
    """Crop off the outermost margin of the drawing (page frame + ruler markers).

    Returns (trimmed_image, (offset_x, offset_y)) so callers can map bbox coords
    back to the original full image.
    """
    w, h = image.size
    mx = int(w * margin_frac)
    my = int(h * margin_frac)
    return image.crop((mx, my, w - mx, h - my)), (mx, my)


def extract_isometric_bbox(image: Image.Image, config: Config) -> VLMExtractResult:
    """Combined LocateAnything + Qwen3-VL extraction.

    Returns bbox in **original-image pixel coordinates**, or None on failure.
    """
    full = image.convert("RGB")
    w_full, h_full = full.size

    # Trim the page frame + ruler markers (8% margin) before showing the drawing
    # to the VLM. Stops the bbox/tightening step from picking up the column-label
    # rulers (A/B/C/D on the right, numbers 1-12 on top) — those used to win the
    # largest-blob race on samples like 121.
    trimmed, (off_x, off_y) = _trim_page_frame(full, margin_frac=0.08)

    small = _downscale(trimmed, _MAX_EDGE_PX)
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
                # Map small-image coords → trimmed-image coords → full-image coords.
                w_trim, h_trim = trimmed.size
                sx, sy = w_trim / w_small, h_trim / h_small
                qwen_bbox_full = (
                    off_x + int(x1 * sx),
                    off_y + int(y1 * sy),
                    off_x + int(x2 * sx),
                    off_y + int(y2 * sy),
                )

    # Combination strategy:
    #   - If both models produced boxes:
    #       * If LA's LARGEST box overlaps Qwen's bbox (IoU > 0.1): take their
    #         union — this is the "they agree, just slightly different shape"
    #         case.
    #       * Else if LA's largest box's CENTER is reasonably close to Qwen's
    #         center (within ~35% of image diagonal): pages with multiple iso
    #         views where Qwen mis-identified which is the largest. Trust LA's
    #         largest, since LA reliably tracks bbox size while Qwen often
    #         picks the wrong "largest" view (fixture 127's two isos).
    #       * Else: TRUST QWEN. LA likely hallucinated a section/detail view
    #         (fixture 110's section circle). Qwen reads the view labels.
    #   - If only LA produced boxes, pick its largest.
    #   - If only Qwen produced a bbox, use it.
    #   - Otherwise fail.
    def _center(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def _distinct_la_boxes(boxes, iou_thresh=0.7):
        """Cluster near-duplicate LA boxes; return one representative per cluster."""
        out: list[tuple[int, int, int, int]] = []
        for b in boxes:
            if not any(_iou(b, o) > iou_thresh for o in out):
                out.append(b)
        return out

    chosen: Optional[tuple[int, int, int, int]] = None
    if la_boxes_full and qwen_bbox_full is not None:
        distinct_la = _distinct_la_boxes(la_boxes_full)
        # Match LA boxes against Qwen by IoU. Qwen knows the view-type (iso vs
        # section/detail); LA gives tighter content envelopes. The IOU-matching
        # LA box is the one Qwen's "this is an iso" judgement endorses.
        matching_la = [b for b in distinct_la if _iou(b, qwen_bbox_full) > 0.1]
        if matching_la:
            # Among LA boxes that Qwen endorses as overlapping the iso area,
            # pick the largest. UNION with Qwen so we don't shrink either bbox.
            best_la = max(matching_la,
                          key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            chosen = _union(best_la, qwen_bbox_full)
            logger.info(
                "Matched LA=%s with Qwen=%s → %s "
                "(picked from %d Qwen-overlapping LA candidate%s)",
                best_la, qwen_bbox_full, chosen,
                len(matching_la), "s" if len(matching_la) != 1 else "",
            )
        else:
            # No LA box overlaps Qwen → LA likely picked a section/detail view.
            # Trust Qwen.
            chosen = qwen_bbox_full
            logger.info("No LA box overlaps Qwen; trusting Qwen: %s "
                        "(largest LA was %s)", chosen, la_boxes_full[0])
    elif la_boxes_full:
        chosen = la_boxes_full[0]
        logger.info("Only LA found boxes; using largest: %s", chosen)
    elif qwen_bbox_full is not None:
        chosen = qwen_bbox_full
        logger.info("Only Qwen produced a bbox; using it: %s", chosen)

    if chosen is None:
        return VLMExtractResult(bbox_xyxy=None,
                                raw_response=description + "\n---\n" + raw)

    # Expand the seed bbox (35%) before tightening — VLMs often clip the bbox
    # to the densest interior of the view, missing the part's outer features.
    # The tightening step then clips back to the part-shaped blob, so a generous
    # over-expansion is safer than under-expanding. The union-with-seed defence
    # at the end ensures we never end up smaller than the model's original pick.
    pad = 0.35
    fx1, fy1, fx2, fy2 = chosen
    pad_x = int((fx2 - fx1) * pad)
    pad_y = int((fy2 - fy1) * pad)
    # Clamp expansion to the trimmed region so we never re-introduce the page
    # frame/rulers we trimmed off.
    trim_x1, trim_y1 = off_x, off_y
    trim_x2, trim_y2 = off_x + trimmed.size[0], off_y + trimmed.size[1]
    padded = (
        max(trim_x1, fx1 - pad_x),
        max(trim_y1, fy1 - pad_y),
        min(trim_x2, fx2 + pad_x),
        min(trim_y2, fy2 + pad_y),
    )

    # Tighten to the part-shaped blob inside the padded region.
    _, tight_bbox = _tighten_to_largest_blob(full, padded)

    # Defensive: never end up smaller than the seed — tightening should only
    # remove adjacent junk, not chop into the actual view. Union the tightened
    # bbox with the (un-padded) seed.
    final_bbox = _union(tight_bbox, chosen)
    # Clamp to the trimmed region so we still avoid the page frame/rulers.
    final_bbox = (
        max(off_x, final_bbox[0]),
        max(off_y, final_bbox[1]),
        min(off_x + trimmed.size[0], final_bbox[2]),
        min(off_y + trimmed.size[1], final_bbox[3]),
    )

    # Apply ink-cleanup (not bbox-tightening) on the final bbox. This whites
    # out adjacent content (ruler letters, neighbouring views) without altering
    # the bbox dimensions.
    cleaned = _clean_outside_hull(full, final_bbox)

    return VLMExtractResult(
        bbox_xyxy=final_bbox,
        raw_response=description + "\n---\n" + raw,
        cleaned_crop=cleaned,
    )
