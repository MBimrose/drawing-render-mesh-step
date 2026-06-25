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


SYSTEM_PROMPT = """You read mechanical engineering drawings and emit a single uniform scale factor.

You will be given (a) an annotated engineering drawing with orthographic and isometric views,
(b) the task description text, and (c) the current LARGEST bounding-box edge length of a
candidate mesh (in arbitrary units, normalized roughly to a unit cube). Your job: identify
the largest real-world dimension implied by the drawing, then emit a single multiplicative
scale factor ``s`` such that

    s * max(candidate_dims) ~= max(target_real_world_dims_in_mm)

Notes:
- Read the largest printed dimension on the drawing (typically a Ø value or an overall
  length annotation). Express it in mm.
- Return ONLY a single JSON object, no prose, no markdown, in this exact schema:
    {"scale": <float>, "unit": "mm", "rationale": "<one short line>"}
- Default unit is "mm" unless the drawing clearly uses another.
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


def _recon_max_dim(recon_stl: Path) -> float:
    return max(_recon_dims(recon_stl))


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


def _fallback_uniform(target_max: float, candidate_max: float) -> ScaleResult:
    s = target_max / candidate_max if candidate_max > 0 else 1.0
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
    """Ask the VLM for a single uniform scale factor mapping CADFit recon to drawing dims.

    Uniform scaling preserves the recon's aspect ratio (which Hunyuan3D-2 + CADFit
    already got mostly right) and produces clean BREP geometry — non-uniform
    ``gp_GTrsf`` warps shapes in ways that defeat the evaluator's tessellator.
    """
    cmax = _recon_max_dim(recon_stl)
    user_text = (
        f"Task description:\n{task_description}\n\n"
        f"Current candidate's largest bounding-box edge (unscaled):\n  max_dim = {cmax:.4f}\n\n"
        "Reply with the JSON object now."
    )

    completion_kwargs = dict(
        model=config.vlm_model,
        api_base=config.vlm_url,
        api_key=config.vlm_api_key,
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
        if parsed and "scale" in parsed:
            break
        completion_kwargs["messages"].append({"role": "assistant", "content": raw})
        completion_kwargs["messages"].append({
            "role": "user",
            "content": "Your previous response did not parse as the required JSON schema. "
                       'Reply with ONLY: {"scale": <float>, "unit": "mm", "rationale": "..."}'
        })

    if not parsed or "scale" not in parsed:
        logger.warning("scaling: could not parse JSON; using uniform fallback")
        return _fallback_uniform(target_max_fallback, cmax)

    try:
        s = float(parsed["scale"])
    except (TypeError, ValueError) as exc:
        logger.warning("scaling: bad float in parsed JSON: %s", exc)
        return _fallback_uniform(target_max_fallback, cmax)

    # The VLM emits the *target real-world max dim*. Convert to a multiplicative factor
    # by dividing by the candidate's current max dim. (Both VLMs and Claude tend to put
    # the millimeter value here, not the multiplicative factor.) Heuristic: if the value
    # is between 0.1 and 20 and the candidate_max is also ~1, treat it as a pure ratio.
    if cmax > 0 and (s < 0.2 or s > 50):
        # value looks like a target dim in mm — convert
        factor = s / cmax
    else:
        # value already looks like a unit-scale ratio
        factor = s

    return ScaleResult(
        sx=factor, sy=factor, sz=factor,
        unit=str(parsed.get("unit", "mm")),
        rationale=f"requested target_max={s:.2f}, candidate_max={cmax:.4f} → s={factor:.4f}; "
                  + str(parsed.get("rationale", ""))[:120],
        raw_response=raw,
    )
