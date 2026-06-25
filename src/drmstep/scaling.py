"""Read drawing dimensions via Claude (litellm proxy), emit per-axis scale factors."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import litellm
import trimesh
from PIL import Image

from .config import Config

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class ScaleResult:
    sx: float
    sy: float
    sz: float
    unit: str
    rationale: str
    raw_response: str


SYSTEM_PROMPT = """You read mechanical engineering drawings and emit scale factors.

You will be given (a) an annotated engineering drawing with orthographic and isometric views,
(b) the task description text, and (c) the current bounding-box dimensions of a candidate mesh
(in arbitrary units, normalized roughly to unit cube). Your job: determine the target real-world
dimensions implied by the drawing, and emit per-axis multiplicative scale factors so that

    candidate_bbox * (sx, sy, sz) ~= target real-world bbox

Notes:
- Use the dimensions printed in the drawing if available. Otherwise infer from the task description.
- If you cannot reliably tell which mesh axis aligns to which drawing axis, return uniform scale
  (sx = sy = sz = uniform_factor) where uniform_factor maps max(candidate_dims) -> max(target_dims).
- Return ONLY a single JSON object, no prose, in this exact schema:
  {"sx": <float>, "sy": <float>, "sz": <float>, "unit": "mm", "rationale": "<one short line>"}
- Default unit is "mm" unless the drawing clearly uses another (e.g. inches).
"""


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _recon_dims(recon_stl: Path) -> tuple[float, float, float]:
    mesh = trimesh.load(recon_stl, force="mesh")
    extents = mesh.bounding_box.extents
    return float(extents[0]), float(extents[1]), float(extents[2])


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    # Strip optional ``` fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _fallback_uniform(target_max: float, candidate_dims: tuple[float, float, float]) -> ScaleResult:
    s = target_max / max(candidate_dims) if max(candidate_dims) > 0 else 1.0
    return ScaleResult(
        sx=s, sy=s, sz=s, unit="mm",
        rationale=f"fallback uniform scale {s:.4f}", raw_response=""
    )


def compute_scale(
    drawing: Image.Image,
    task_description: str,
    recon_stl: Path,
    config: Config,
    *,
    target_max_fallback: float = 50.0,
) -> ScaleResult:
    """Ask Claude for per-axis scale factors mapping CADFit recon to drawing dims."""
    cx, cy, cz = _recon_dims(recon_stl)
    user_text = (
        f"Task description:\n{task_description}\n\n"
        f"Current candidate bounding-box (unscaled, CADFit recon):\n"
        f"  X = {cx:.4f}\n  Y = {cy:.4f}\n  Z = {cz:.4f}\n\n"
        "Emit the JSON object now."
    )

    completion_kwargs = dict(
        model=config.vlm_model,
        api_base=config.litellm_url,
        api_key=config.litellm_api_key,
        temperature=0.0,
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(drawing)}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
    )

    raw = ""
    parsed: dict | None = None
    for attempt in (1, 2):
        try:
            resp = litellm.completion(**completion_kwargs)
        except Exception as exc:
            logger.warning("litellm completion failed (attempt %d): %s", attempt, exc)
            break
        raw = resp.choices[0].message.content or ""
        parsed = _parse_json(raw)
        if parsed and all(k in parsed for k in ("sx", "sy", "sz")):
            break
        completion_kwargs["messages"].append({"role": "assistant", "content": raw})
        completion_kwargs["messages"].append({
            "role": "user",
            "content": "Your previous response did not parse as the required JSON schema. "
                       "Reply with ONLY the JSON object: "
                       '{"sx": <float>, "sy": <float>, "sz": <float>, "unit": "mm", "rationale": "..."}'
        })

    if not parsed or not all(k in parsed for k in ("sx", "sy", "sz")):
        logger.warning("scaling: could not parse JSON from Claude; falling back uniform")
        return _fallback_uniform(target_max_fallback, (cx, cy, cz))

    try:
        return ScaleResult(
            sx=float(parsed["sx"]),
            sy=float(parsed["sy"]),
            sz=float(parsed["sz"]),
            unit=str(parsed.get("unit", "mm")),
            rationale=str(parsed.get("rationale", ""))[:200],
            raw_response=raw,
        )
    except (TypeError, ValueError) as exc:
        logger.warning("scaling: bad floats in parsed JSON: %s", exc)
        return _fallback_uniform(target_max_fallback, (cx, cy, cz))
