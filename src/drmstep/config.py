"""Centralized config for the pipeline. Overridable via env vars."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(BaseModel):
    # Service endpoints. Defaults assume a local OpenAI-compatible vLLM (Qwen3-VL).
    # Override via DRMSTEP_VLM_URL / DRMSTEP_VLM_KEY / DRMSTEP_VLM_MODEL to point at
    # Anthropic (https://api.anthropic.com), an Anthropic-compatible litellm proxy,
    # or another OpenAI-compatible endpoint.
    hunyuan_url: str = Field(
        default_factory=lambda: os.environ.get("DRMSTEP_HUNYUAN_URL", "http://localhost:8081")
    )
    vlm_url: str = Field(
        default_factory=lambda: os.environ.get(
            "DRMSTEP_VLM_URL", "http://wpk-serv-07.mechse.illinois.edu:8002/v1"
        )
    )
    vlm_api_key: str = Field(
        default_factory=lambda: os.environ.get("DRMSTEP_VLM_KEY", "not-needed")
    )

    # Models. Default is the locally-available Qwen3-VL-235B. Set DRMSTEP_VLM_MODEL
    # to e.g. "anthropic/claude-opus-4-7" + ANTHROPIC_API_KEY to use real Claude.
    vlm_model: str = Field(
        default_factory=lambda: os.environ.get("DRMSTEP_VLM_MODEL", "openai/qwen3-vl-235b")
    )
    locate_anything_model: str = "nvidia/LocateAnything-3B"

    # View-extraction backend:
    #   "vlm"            — Multi-stage Qwen3-VL (default; falls back to classical)
    #   "classical"      — OpenCV whitespace-split + diagonal-line scoring
    #   "locate_anything" — NVIDIA LocateAnything-3B in-process
    #   "qwen"            — Legacy single-call Qwen3-VL
    view_backend: str = Field(
        default_factory=lambda: os.environ.get("DRMSTEP_VIEW_BACKEND", "vlm")
    )

    # Hunyuan3D request defaults
    hunyuan_num_inference_steps: int = 50
    hunyuan_octree_resolution: int = 384
    hunyuan_guidance_scale: float = 5.0
    hunyuan_timeout_s: float = 300.0

    # CADFit
    cadfit_dir: Path = Field(default_factory=lambda: REPO_ROOT / "third_party" / "CADFit")
    cadfit_python: Path = Field(
        default_factory=lambda: REPO_ROOT / "third_party" / "CADFit" / ".venv" / "bin" / "python"
    )
    cadfit_max_iterations: int = 15
    cadfit_fillet_chamfer: bool = True
    cadfit_timeout_s: float = 5400.0

    # CadQuery runner
    cadquery_python: Path = Field(
        default_factory=lambda: _resolve_cq_python()
    )
    cadquery_timeout_s: float = 120.0

    # I/O
    work_root: Path = Field(
        default_factory=lambda: Path(os.environ.get("DRMSTEP_WORK_ROOT", str(REPO_ROOT / "runs" / "_work")))
    )


def _default_cq_python() -> Path:
    """Prefer CADFit's venv (which already has cadquery), else current interpreter."""
    cf = REPO_ROOT / "third_party" / "CADFit" / ".venv" / "bin" / "python"
    if cf.exists():
        return cf
    import sys
    return Path(sys.executable)


def _resolve_cq_python() -> Path:
    env = os.environ.get("DRMSTEP_CQ_PYTHON")
    if env:
        return Path(env)
    return _default_cq_python()


def load_config() -> Config:
    return Config()
