"""View extraction with NVIDIA LocateAnything-3B.

Given a multi-view engineering drawing, return a crop of the isometric/3D view.
If grounding fails (no bbox, or bbox too small), return the original image unchanged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from PIL import Image

from .config import Config

logger = logging.getLogger(__name__)

_BBOX_RE = re.compile(r"<box>\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*</box>")
_MIN_BBOX_AREA_FRAC = 0.05
_PAD_FRAC = 0.05


@dataclass(frozen=True)
class ExtractResult:
    image: Image.Image
    bbox_xyxy: Optional[tuple[int, int, int, int]]
    raw_response: str


@lru_cache(maxsize=1)
def _load_model(model_id: str):
    """Load LocateAnything-3B once. Returns (model, processor) on cuda."""
    import torch
    from transformers import AutoModel, AutoProcessor

    logger.info("loading %s on cuda", model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    return model, processor


def _parse_bbox(response: str, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    """Parse <box>x1 y1 x2 y2</box> with 0-1000 normalized coords. Pick the largest match."""
    matches = _BBOX_RE.findall(response)
    if not matches:
        return None
    boxes_px = []
    for x1, y1, x2, y2 in matches:
        bx1 = int(float(x1) / 1000.0 * width)
        by1 = int(float(y1) / 1000.0 * height)
        bx2 = int(float(x2) / 1000.0 * width)
        by2 = int(float(y2) / 1000.0 * height)
        if bx2 <= bx1 or by2 <= by1:
            continue
        area = (bx2 - bx1) * (by2 - by1)
        boxes_px.append((area, (bx1, by1, bx2, by2)))
    if not boxes_px:
        return None
    boxes_px.sort(reverse=True)
    return boxes_px[0][1]


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


def extract_isometric_view(image: Image.Image, config: Config) -> ExtractResult:
    """Locate and crop the isometric view from a multi-view drawing.

    Args:
        image: source PIL image (an engineering drawing).
        config: pipeline config.

    Returns:
        ExtractResult with the cropped (or original) image, the bbox if used, and raw VLM text.
    """
    model, processor = _load_model(config.locate_anything_model)
    image = image.convert("RGB")
    w, h = image.size

    prompt = (
        "Locate the isometric or 3D perspective view of the part. "
        "If multiple 3D views exist, return only the largest one."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

    import torch

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    response = processor.batch_decode(out, skip_special_tokens=True)[0]

    bbox = _parse_bbox(response, w, h)
    if bbox is None:
        logger.info("no bbox parsed; using full image")
        return ExtractResult(image=image, bbox_xyxy=None, raw_response=response)

    x1, y1, x2, y2 = bbox
    area_frac = ((x2 - x1) * (y2 - y1)) / float(w * h)
    if area_frac < _MIN_BBOX_AREA_FRAC:
        logger.info("bbox too small (%.3f frac); using full image", area_frac)
        return ExtractResult(image=image, bbox_xyxy=None, raw_response=response)

    padded = _pad_bbox(bbox, w, h)
    crop = image.crop(padded)
    return ExtractResult(image=crop, bbox_xyxy=padded, raw_response=response)
