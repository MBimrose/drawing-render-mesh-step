"""Edit pipeline: VLM-patches-CadQuery.

input.step + edit instruction (description.yaml.task_description + input.png)
  -> tessellate input.step to STL
  -> CADFit recovers a CadQuery program
  -> Claude patches it per the instruction
  -> execute -> output.step (fallback: copy input.step verbatim)
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

from . import cad_fit, code_patch, runners
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class EditResult:
    sample: str
    status: str          # "ok", "noop_fallback", "failed"
    output_path: Path
    notes: str = ""


def _step_to_stl(step_path: Path, stl_path: Path, config: Config) -> None:
    """Tessellate a STEP to STL via cadquery in a subprocess (CADFit's venv has cq)."""
    import subprocess
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""
import cadquery as cq
shape = cq.importers.importStep({str(step_path)!r})
cq.exporters.export(shape, {str(stl_path)!r}, tolerance=0.05, angularTolerance=0.2)
"""
    proc = subprocess.run(
        [str(config.cadquery_python), "-c", script],
        capture_output=True, text=True, timeout=120.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"step->stl failed: {proc.stderr[-1000:]}")
    if not stl_path.exists():
        raise RuntimeError(f"step->stl did not produce {stl_path}")


def run_edit(
    sample: str,
    inputs_dir: Path,
    out_dir: Path,
    config: Config,
) -> EditResult:
    """Run the edit pipeline on one sample. Expects ``input.step`` in ``inputs_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work = config.work_root / sample
    work.mkdir(parents=True, exist_ok=True)

    desc = yaml.safe_load((inputs_dir / "description.yaml").read_text()) or {}
    task_description = desc.get("task_description", "")
    input_step = inputs_dir / "input.step"
    input_png = inputs_dir / "input.png"
    output_step = out_dir / "output.step"

    if not input_step.exists():
        # Some sample variants use .stp
        alt = inputs_dir / "input.stp"
        if alt.exists():
            input_step = alt
        else:
            raise FileNotFoundError(f"no input.step in {inputs_dir}")

    drawing = Image.open(input_png).convert("RGB") if input_png.exists() else Image.new("RGB", (256, 256))

    try:
        # 1. step -> stl
        logger.info("[%s] step -> stl", sample)
        input_stl = work / "input.stl"
        _step_to_stl(input_step, input_stl, config)

        # 2. CADFit -> CadQuery code that approximates input.step
        logger.info("[%s] CADFit on input", sample)
        cf = cad_fit.run_cadfit(input_stl, work / "cadfit", config)

        # 3. Claude patches the code per instruction
        logger.info("[%s] Claude patches CadQuery", sample)
        patched = code_patch.patch_cadquery(cf.cadquery_code, task_description, drawing, config)
        (work / "patched.py").write_text(patched.cadquery_code)

        # 4. Execute patched code -> output.step
        runners.execute_cadquery(
            patched.cadquery_code, output_step, (1.0, 1.0, 1.0), config, work,
        )
        return EditResult(sample=sample, status="ok", output_path=output_step)

    except Exception as exc:
        logger.warning("[%s] edit pipeline failed: %s; falling back to input.step verbatim",
                       sample, exc)
        shutil.copy(input_step, output_step)
        return EditResult(
            sample=sample, status="noop_fallback", output_path=output_step,
            notes=str(exc)[:300],
        )
