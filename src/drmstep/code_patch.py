"""Claude-driven patcher for the edit task.

Takes the CADFit-recovered CadQuery program plus an edit instruction and returns a
new CadQuery program implementing the requested modification.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass
from typing import Optional

import litellm
from PIL import Image

from .config import Config

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.+?)\s*```", re.DOTALL)


@dataclass(frozen=True)
class PatchResult:
    cadquery_code: str
    rationale: str
    raw_response: str


SYSTEM_PROMPT = """You modify CadQuery programs to implement small editing requests.

You will be given:
- The original CadQuery program that approximately reconstructs an input STEP part.
- The text instruction for the edit (e.g. "remove the 5mm boss on the top face").
- The annotated engineering drawing as image context.

Rules:
- Return ONE Python code block (```python ... ```) and nothing else.
- The code must define a CadQuery result and assign it to a variable named ``result``.
- Keep the geometry, sizes, and orientation of the un-edited parts unchanged.
- Do NOT call ``cq.exporters.export`` or print anything; the runner handles export.
- Do NOT introduce new dependencies; stick to ``cadquery``.
- Express dimensions in millimeters.
"""


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _extract_code(text: str) -> Optional[str]:
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: assume the whole reply is code if it parses as having `import cadquery`.
    if "cadquery" in text.lower() and "import" in text.lower():
        return text.strip()
    return None


def patch_cadquery(
    original_code: str,
    edit_instruction: str,
    drawing: Image.Image,
    config: Config,
) -> PatchResult:
    user_text = (
        f"Edit instruction:\n{edit_instruction}\n\n"
        f"Original CadQuery program (recovered by CADFit, may include parameter blocks "
        f"and helpers):\n\n```python\n{original_code}\n```\n\n"
        "Return one ```python``` code block whose final statement assigns the modified "
        "result to a variable named `result`."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_to_data_url(drawing)}},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    raw = ""
    code: Optional[str] = None
    for attempt in (1, 2):
        try:
            resp = litellm.completion(
                model=config.vlm_model,
                api_base=config.vlm_url,
                api_key=config.vlm_api_key,
                temperature=0.0,
                max_tokens=4000,
                messages=messages,
            )
        except Exception as exc:
            logger.warning("litellm completion failed (attempt %d): %s", attempt, exc)
            break
        raw = resp.choices[0].message.content or ""
        code = _extract_code(raw)
        if code and "result" in code:
            break
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": "Your previous reply did not contain a parseable ```python``` block "
                       "with a `result` variable. Try again — one code block only.",
        })

    if not code:
        raise RuntimeError("could not extract CadQuery code from Claude response")

    return PatchResult(cadquery_code=code, rationale="", raw_response=raw)
