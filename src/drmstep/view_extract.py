"""View extraction for engineering drawings.

Backends:
  - ``classical`` (default): OpenCV whitespace-split + Hough diagonal-line scoring.
    Deterministic, ~100ms, no GPU. See :mod:`drmstep.view_classical`.
  - ``locate_anything``: NVIDIA LocateAnything-3B in-process, with a top-right
    quadrant prefilter (drafting convention puts the isometric there).
  - ``qwen``: Qwen3-VL via litellm, with the same TR-quadrant prefilter.

Set ``DRMSTEP_VIEW_BACKEND`` to choose. All backends fall back to the full image
when grounding fails.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import litellm
from PIL import Image

from .config import Config

logger = logging.getLogger(__name__)

_JSON_BBOX_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_ARRAY_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\]}]")
# LocateAnything emits boxes as <box><x1><y1><x2><y2></box>. Also accept the
# whitespace-separated variant just in case.
_LA_BOX_BLOCK_RE = re.compile(r"<box>(.*?)</box>", re.DOTALL)
_LA_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
_MIN_BBOX_AREA_FRAC = 0.02
_MAX_BBOX_AREA_FRAC = 0.95  # boxes spanning the whole image are LocateAnything "I dunno"
_PAD_FRAC = 0.05
_MAX_EDGE_PX = 1280  # cap for the VLM call


@dataclass(frozen=True)
class ExtractResult:
    image: Image.Image
    bbox_xyxy: Optional[tuple[int, int, int, int]]
    raw_response: str


def _pad_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int, frac: float = _PAD_FRAC
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * frac)
    pad_y = int((y2 - y1) * frac)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _downscale(image: Image.Image, max_edge: int) -> Image.Image:
    w, h = image.size
    scale = min(1.0, max_edge / max(w, h))
    if scale >= 1.0:
        return image
    return image.resize((int(w * scale), int(h * scale)))


_QWEN_SYSTEM = (
    "You read engineering drawings and locate views by returning a single bounding box "
    "as a 4-element array of absolute pixel coordinates [x1, y1, x2, y2]. "
    "Reply with ONE JSON object only — no prose, no markdown — in the schema "
    "{\"bbox\": [x1, y1, x2, y2]}. "
    "Use the image's actual pixel coordinate system, not a normalized one. "
    "If the requested view is not present, return {\"bbox\": [0, 0, 0, 0]}."
)

_QWEN_USER = (
    "Mechanical engineering drawing. Locate the ISOMETRIC view — the single 3D shaded "
    "pictorial view of the whole part. It is typically drawn smaller in a corner of the "
    "sheet, with shading or hidden lines, and shows depth (no flat orthographic projection). "
    "Ignore the orthographic (top/front/side), section, and detail views. "
    "The image is {w} pixels wide by {h} pixels tall. "
    "Reply with {{\"bbox\": [x1, y1, x2, y2]}} only."
)


def _parse_json_bbox(text: str) -> Optional[tuple[int, int, int, int]]:
    """Pull a 4-int bbox out of Qwen's reply.

    Accepted forms (lenient because VLMs sometimes mangle JSON):
      - {"bbox": [x1, y1, x2, y2]}
      - {"x1": .., "y1": .., "x2": .., "y2": ..}
      - [x1, y1, x2, y2] anywhere in the text (fallback)
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    json_candidates = [text] + [m.group(0) for m in _JSON_BBOX_RE.finditer(text)]
    for cand in json_candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            arr = data.get("bbox")
            if isinstance(arr, (list, tuple)) and len(arr) == 4:
                try:
                    x1, y1, x2, y2 = (int(v) for v in arr)
                except (TypeError, ValueError):
                    pass
                else:
                    if x2 > x1 and y2 > y1:
                        return (x1, y1, x2, y2)
            try:
                x1 = int(data["x1"]); y1 = int(data["y1"])
                x2 = int(data["x2"]); y2 = int(data["y2"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if x2 > x1 and y2 > y1:
                    return (x1, y1, x2, y2)

    m = _ARRAY_RE.search(text)
    if m:
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
    return None


def _call_qwen_for_bbox(crop: Image.Image, config: Config) -> tuple[Optional[tuple[int, int, int, int]], str]:
    """Ask Qwen to locate the isometric view inside ``crop``. Returns bbox in crop pixel coords."""
    small = _downscale(crop, _MAX_EDGE_PX)
    w_small, h_small = small.size
    try:
        resp = litellm.completion(
            model=config.vlm_model,
            api_base=config.vlm_url,
            api_key=config.vlm_api_key,
            temperature=0.0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": _QWEN_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(small)}},
                        {"type": "text", "text": _QWEN_USER.format(w=w_small, h=h_small)},
                    ],
                },
            ],
        )
    except Exception as exc:
        logger.warning("view extraction VLM call failed: %s", exc)
        return None, str(exc)

    raw = resp.choices[0].message.content or ""
    bbox_small = _parse_json_bbox(raw)
    if bbox_small is None:
        return None, raw

    # Map small-image bbox back to ``crop`` coords.
    w_crop, h_crop = crop.size
    sx, sy = w_crop / w_small, h_crop / h_small
    x1, y1, x2, y2 = bbox_small
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)), raw


def _extract_with_qwen(image: Image.Image, config: Config) -> ExtractResult:
    """Two-stage view extraction.

    Stage 1: pre-crop to the top-right quadrant of the drawing. Engineering
    drawings put the isometric pictorial there by convention; this makes the
    isometric a much larger fraction of the image the VLM sees, which helps
    Qwen3-VL pin it down precisely instead of grabbing nearby dimension callouts.

    Stage 2: ask Qwen to locate the isometric inside that quadrant.

    If Qwen fails to find one in the quadrant, retry once on the full image
    before giving up to the full-image fallback.
    """
    image_full = image.convert("RGB")
    w_full, h_full = image_full.size

    # Stage 1: top-right quadrant. Use the right half + top half-and-bit so we
    # don't accidentally clip the isometric if the layout sits a bit lower.
    quad_box = (w_full // 2, 0, w_full, int(h_full * 0.6))
    quad = image_full.crop(quad_box)

    bbox_in_quad, raw = _call_qwen_for_bbox(quad, config)
    if bbox_in_quad is not None:
        qx1, qy1, _, _ = quad_box
        x1, y1, x2, y2 = bbox_in_quad
        bbox_full = (qx1 + x1, qy1 + y1, qx1 + x2, qy1 + y2)
    else:
        # Stage 2 fallback: try the full image. Maybe the layout isn't TR-quadrant.
        logger.info("view extraction: TR quadrant bbox missing; retrying on full image")
        bbox_in_full, raw_full = _call_qwen_for_bbox(image_full, config)
        raw = raw + "\n--- full image retry ---\n" + raw_full
        if bbox_in_full is None:
            logger.info("view extraction: no parseable bbox; using full image")
            return ExtractResult(image=image_full, bbox_xyxy=None, raw_response=raw)
        bbox_full = bbox_in_full

    x1, y1, x2, y2 = bbox_full
    area_frac = ((x2 - x1) * (y2 - y1)) / float(w_full * h_full)
    if area_frac < _MIN_BBOX_AREA_FRAC:
        logger.info("view extraction: bbox too small (%.3f frac); using full image", area_frac)
        return ExtractResult(image=image_full, bbox_xyxy=None, raw_response=raw)

    padded = _pad_bbox(bbox_full, w_full, h_full)
    crop = image_full.crop(padded)
    return ExtractResult(image=crop, bbox_xyxy=padded, raw_response=raw)


# ---------------------------------------------------------------------------
# LocateAnything-3B (in-process)
# ---------------------------------------------------------------------------

_LA_PROMPT = "Locate the 3D view of the part."


@lru_cache(maxsize=1)
def _load_locate_anything(model_id: str):
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    logger.info("loading %s on cuda", model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    return model, processor, tokenizer


def _parse_la_bbox(response: str, w: int, h: int) -> Optional[tuple[int, int, int, int]]:
    """Pick the largest non-degenerate <box> from a LocateAnything response.

    LocateAnything emits each box as ``<box><x1><y1><x2><y2></box>`` (numbers wrapped
    in their own angle brackets) with coords 0–999 normalized. Two filters:
      - drop boxes that span ~the entire image (the model's "I don't know" output,
        often repeated dozens of times after a real answer),
      - drop sub-area-threshold boxes.
    Among the survivors, return the largest by area.
    """
    boxes = []
    for raw in _LA_BOX_BLOCK_RE.findall(response):
        nums = _LA_NUM_RE.findall(raw)
        if len(nums) < 4:
            continue
        x1, y1, x2, y2 = (float(n) for n in nums[:4])
        bx1 = int(x1 / 1000.0 * w); by1 = int(y1 / 1000.0 * h)
        bx2 = int(x2 / 1000.0 * w); by2 = int(y2 / 1000.0 * h)
        if bx2 <= bx1 or by2 <= by1:
            continue
        area_frac = ((bx2 - bx1) * (by2 - by1)) / float(w * h)
        if area_frac >= _MAX_BBOX_AREA_FRAC:
            continue
        boxes.append((area_frac, (bx1, by1, bx2, by2)))
    if not boxes:
        return None
    return max(boxes)[1]


def _call_locate_anything(crop: Image.Image, config: Config) -> tuple[Optional[tuple[int, int, int, int]], str]:
    """Run LocateAnything on ``crop`` and return its largest bbox in crop pixel coords."""
    import torch

    model, processor, tokenizer = _load_locate_anything(config.locate_anything_model)
    small = _downscale(crop, 1024)

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": small}, {"type": "text", "text": _LA_PROMPT}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[small], return_tensors="pt").to("cuda")

    with torch.inference_mode():
        response = model.generate(
            **inputs, tokenizer=tokenizer, max_new_tokens=256,
            use_cache=True, generation_mode="hybrid",
        )

    # bbox coords are 0-1000 normalized, so map back to the original ``crop`` size.
    w_crop, h_crop = crop.size
    return _parse_la_bbox(response, w_crop, h_crop), response


def _extract_with_locate_anything(image: Image.Image, config: Config) -> ExtractResult:
    """Two-stage view extraction with LocateAnything-3B.

    Stage 1: pre-crop the top-right quadrant of the drawing. CAD drafting
    convention puts the isometric pictorial in the upper-right corner of the
    sheet, so this dramatically reduces the noise LocateAnything has to look at.

    Stage 2: ask LocateAnything for "the 3D view of the part" inside that
    quadrant; keep the largest returned box (helps when it returns multiple
    candidate views).

    If the quadrant call finds nothing, retry once on the full image before
    giving up to the full-image fallback.
    """
    image_full = image.convert("RGB")
    w_full, h_full = image_full.size

    # Stage 1: top-right quadrant. A little taller than half-height to keep the
    # whole pictorial even if it dips below center.
    quad_box = (w_full // 2, 0, w_full, int(h_full * 0.6))
    quad = image_full.crop(quad_box)
    bbox_in_quad, raw = _call_locate_anything(quad, config)

    if bbox_in_quad is not None:
        qx1, qy1, _, _ = quad_box
        x1, y1, x2, y2 = bbox_in_quad
        bbox_full = (qx1 + x1, qy1 + y1, qx1 + x2, qy1 + y2)
    else:
        logger.info("view extraction: TR quadrant bbox missing; retrying on full image")
        bbox_in_full, raw_full = _call_locate_anything(image_full, config)
        raw = (raw or "") + "\n--- full image retry ---\n" + (raw_full or "")
        if bbox_in_full is None:
            logger.info("view extraction: no parseable bbox; using full image")
            return ExtractResult(image=image_full, bbox_xyxy=None, raw_response=raw)
        bbox_full = bbox_in_full

    x1, y1, x2, y2 = bbox_full
    area_frac = ((x2 - x1) * (y2 - y1)) / float(w_full * h_full)
    if area_frac < _MIN_BBOX_AREA_FRAC:
        logger.info("view extraction: bbox too small (%.3f frac); using full image", area_frac)
        return ExtractResult(image=image_full, bbox_xyxy=None, raw_response=raw)

    padded = _pad_bbox(bbox_full, w_full, h_full)
    return ExtractResult(image=image_full.crop(padded), bbox_xyxy=padded, raw_response=raw)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _extract_with_classical(image: Image.Image, config: Config) -> ExtractResult:
    """OpenCV-based view splitting; pick the region with the highest diagonal-line score."""
    from . import view_classical

    image_full = image.convert("RGB")
    w_full, h_full = image_full.size
    bbox = view_classical.find_isometric_bbox(image_full)
    if bbox is None:
        logger.info("view extraction (classical): no candidate region; using full image")
        return ExtractResult(image=image_full, bbox_xyxy=None,
                             raw_response="classical: no candidate")

    x1, y1, x2, y2 = bbox
    area_frac = ((x2 - x1) * (y2 - y1)) / float(w_full * h_full)
    if area_frac < _MIN_BBOX_AREA_FRAC:
        logger.info("view extraction (classical): bbox too small; using full image")
        return ExtractResult(image=image_full, bbox_xyxy=None,
                             raw_response=f"classical: bbox={bbox} area_frac={area_frac:.3f}")

    padded = _pad_bbox(bbox, w_full, h_full)
    crop = image_full.crop(padded)
    return ExtractResult(image=crop, bbox_xyxy=padded,
                         raw_response=f"classical: bbox={bbox} area_frac={area_frac:.3f}")


def _extract_with_vlm(image: Image.Image, config: Config) -> ExtractResult:
    """Multi-stage Qwen3-VL extraction with classical CV fallback.

    Two-pass: VLM describes views, then locates the isometric bbox. On VLM
    failure or implausible bbox, falls back to the classical CV scorer.
    """
    from . import view_vlm

    image_full = image.convert("RGB")
    w_full, h_full = image_full.size
    res = view_vlm.extract_isometric_bbox(image_full, config)
    if res.bbox_xyxy is None:
        logger.info("view extraction (vlm): no bbox; falling back to classical CV")
        return _extract_with_classical(image, config)
    crop = image_full.crop(res.bbox_xyxy)
    return ExtractResult(image=crop, bbox_xyxy=res.bbox_xyxy, raw_response=res.raw_response)


def extract_isometric_view(image: Image.Image, config: Config) -> ExtractResult:
    """Crop the isometric view from a multi-view engineering drawing.

    Backend is chosen by ``config.view_backend``. Falls back to the full image
    when grounding fails.
    """
    backend = getattr(config, "view_backend", "vlm")
    if backend == "vlm":
        return _extract_with_vlm(image, config)
    if backend == "qwen":  # legacy alias
        return _extract_with_qwen(image, config)
    if backend == "locate_anything":
        return _extract_with_locate_anything(image, config)
    return _extract_with_classical(image, config)
