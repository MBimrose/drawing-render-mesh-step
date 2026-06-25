"""Read drawing dimensions via Qwen3-VL and compute a uniform scale factor.

Two-stage VLM check so the scale is grounded in MULTIPLE dimensions, not just
the largest. The pipeline:

  1. Show the full drawing + the candidate's three bounding-box edges to Qwen.
  2. Qwen reads several explicit dimension annotations from the drawing and
     returns the real-world value of each candidate edge AND the overall
     largest-dim target.
  3. We compute three candidate scale factors (target_i / candidate_i) and
     take the MEDIAN — a uniform scale that's robust to one outlier annotation.

This catches the common failure where Qwen confidently reads ONE dimension that
turns out to be a hole diameter or a partial measurement; the median across
three reads is far more reliable.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import statistics
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


SYSTEM_PROMPT = """You are a mechanical engineer who reads multi-view engineering drawings.

You will be given:
  (a) An annotated engineering drawing of a single part with dimension callouts.
  (b) The CURRENT candidate part's bounding-box edge lengths (X, Y, Z) in
      arbitrary CAD units — these are NOT in millimeters, they are normalized
      values from a CAD reconstruction.

Your job: read the drawing's dimension annotations and return the
**real-world dimensions (in millimeters)** that correspond to each candidate
edge AND the part's overall largest extent.

Rules:
  - Identify the LARGEST OVERALL dimension of the part on the drawing — the
    longest length, the largest diameter (Ø), the tallest height — whichever
    is bigger. This is ``target_max_mm``.
  - Independently identify a SECOND largest dimension (e.g. width perpendicular
    to the length) — ``target_second_mm``.
  - Independently identify a THIRD dimension (e.g. thickness, height) —
    ``target_third_mm``.
  - These should be three DIFFERENT real measurements actually printed on the
    drawing — overall extents, not detail-feature sizes (hole diameters are OK
    only if they are the largest extent of the part).
  - If a dimension cannot be reliably read, set it to null.
  - Return ONLY a JSON object, no prose, in this schema:
      {
        "target_max_mm": <float>,
        "target_second_mm": <float|null>,
        "target_third_mm": <float|null>,
        "rationale": "<one short line: which three dims you read>"
      }
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
    """Ask Qwen3-VL for MULTIPLE drawing dimensions, return a uniform scale.

    The scale factor is the MEDIAN of (target_i / candidate_sorted_i) across
    the three largest part edges. The median is robust to one bad VLM read.
    """
    cx, cy, cz = _recon_dims(recon_stl)
    cand_sorted = sorted([cx, cy, cz], reverse=True)
    cmax, cmid, cmin = cand_sorted

    user_text = (
        f"Task description:\n{task_description}\n\n"
        f"Current candidate part bounding-box edges (CAD units, "
        f"NOT mm — these will be scaled to match the drawing):\n"
        f"  Largest edge  = {cmax:.4f}\n"
        f"  Second edge   = {cmid:.4f}\n"
        f"  Smallest edge = {cmin:.4f}\n\n"
        "Read three dimensions from the drawing now and return the JSON object."
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
        if parsed and "target_max_mm" in parsed:
            break
        completion_kwargs["messages"].append({"role": "assistant", "content": raw})
        completion_kwargs["messages"].append({
            "role": "user",
            "content": "Your previous response did not parse as the required JSON schema. "
                       'Reply with ONLY: {"target_max_mm": <float>, '
                       '"target_second_mm": <float|null>, "target_third_mm": <float|null>, '
                       '"rationale": "..."}'
        })

    if not parsed or "target_max_mm" not in parsed:
        logger.warning("scaling: could not parse JSON; using uniform fallback")
        return _fallback_uniform(target_max_fallback, cmax)

    def _to_float(v):
        if v is None:
            return None
        try:
            f = float(v)
            return f if f > 0 else None
        except (TypeError, ValueError):
            return None

    t_max = _to_float(parsed.get("target_max_mm"))
    t_mid = _to_float(parsed.get("target_second_mm"))
    t_min = _to_float(parsed.get("target_third_mm"))

    if t_max is None:
        return _fallback_uniform(target_max_fallback, cmax)

    # Compute per-edge scale candidates.
    scale_candidates: list[float] = []
    if cmax > 0:
        scale_candidates.append(t_max / cmax)
    if t_mid is not None and cmid > 0:
        scale_candidates.append(t_mid / cmid)
    if t_min is not None and cmin > 0:
        scale_candidates.append(t_min / cmin)

    if not scale_candidates:
        return _fallback_uniform(target_max_fallback, cmax)

    # Robust: median (or mean when only 1 reading). Single reading is OK because
    # target_max is the most reliably-readable dim on most drawings.
    if len(scale_candidates) >= 3:
        factor = float(statistics.median(scale_candidates))
    elif len(scale_candidates) == 2:
        factor = float(statistics.mean(scale_candidates))
    else:
        factor = float(scale_candidates[0])

    rationale = (
        f"candidates={[round(s, 4) for s in scale_candidates]} → s={factor:.4f}; "
        f"t_max={t_max} t_second={t_mid} t_third={t_min}; "
        + str(parsed.get("rationale", ""))[:120]
    )

    return ScaleResult(
        sx=factor, sy=factor, sz=factor,
        unit="mm", rationale=rationale, raw_response=raw,
    )
