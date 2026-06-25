"""Centralized config for the pipeline. Overridable via env vars."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(BaseModel):
    # Service endpoints
    hunyuan_url: str = Field(
        default_factory=lambda: os.environ.get("DRMSTEP_HUNYUAN_URL", "http://localhost:8081")
    )
    litellm_url: str = Field(
        default_factory=lambda: os.environ.get("DRMSTEP_LITELLM_URL", "http://localhost:4000")
    )
    litellm_api_key: str = Field(
        default_factory=lambda: os.environ.get("DRMSTEP_LITELLM_KEY", "sk-anything")
    )

    # Models
    vlm_model: str = Field(
        default_factory=lambda: os.environ.get("DRMSTEP_VLM_MODEL", "claude-opus-4-7")
    )
    locate_anything_model: str = "nvidia/LocateAnything-3B"

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
    cadfit_max_iterations: int = 3
    cadfit_fillet_chamfer: bool = True
    cadfit_timeout_s: float = 1200.0

    # CadQuery runner
    cadquery_python: Path = Field(
        default_factory=lambda: Path(os.environ.get("DRMSTEP_CQ_PYTHON", "")) or _default_cq_python()
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


def load_config() -> Config:
    return Config()
