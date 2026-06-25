"""CADFit subprocess wrapper.

Runs ``third_party/CADFit/run_pipeline.py`` in CADFit's own venv against a directory
containing ``mesh.stl``. Returns the recovered CadQuery code, recon STL, and IoU.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)


class CadFitError(RuntimeError):
    pass


@dataclass(frozen=True)
class CadFitResult:
    cadquery_code: str
    recon_stl: Path
    iou: float
    work_dir: Path


def run_cadfit(stl_path: Path, work_dir: Path, config: Config) -> CadFitResult:
    """Run CADFit on a single STL.

    CADFit ingests a folder of ``*.stl`` files and writes outputs to
    ``<output_folder>/<stl_id>/best_greedy_parallel_iterative.{py,stl}``.
    We stage the input as ``<work_dir>/input/mesh.stl`` and target
    ``<work_dir>/output/`` so outputs land at
    ``<work_dir>/output/mesh/best_greedy_parallel_iterative.*``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    staged = input_dir / "mesh.stl"
    if staged.resolve() != stl_path.resolve():
        shutil.copy(stl_path, staged)

    cmd = [
        str(config.cadfit_python),
        str(config.cadfit_dir / "run_pipeline.py"),
        str(input_dir),
        "--output-folder",
        str(output_dir),
        "--max-iterations",
        str(config.cadfit_max_iterations),
    ]
    if config.cadfit_fillet_chamfer:
        cmd.append("--fillet-chamfer")

    logger.info("CADFit: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=config.cadfit_dir,
            capture_output=True,
            text=True,
            timeout=config.cadfit_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise CadFitError(f"CADFit timed out after {config.cadfit_timeout_s}s") from exc

    if proc.returncode != 0:
        logger.error("CADFit stdout:\n%s", proc.stdout[-2000:])
        logger.error("CADFit stderr:\n%s", proc.stderr[-2000:])
        raise CadFitError(f"CADFit exited {proc.returncode}")

    stl_dir = output_dir / "mesh"
    code_path = stl_dir / "best_greedy_parallel_iterative.py"
    recon_path = stl_dir / "best_greedy_parallel_iterative.stl"
    iou_path = stl_dir / "final_iou.json"

    if not code_path.exists() or not recon_path.exists():
        raise CadFitError(
            f"CADFit did not emit expected outputs in {stl_dir}: "
            f"code={code_path.exists()} recon={recon_path.exists()}"
        )

    code = code_path.read_text()
    try:
        iou = float(json.loads(iou_path.read_text()).get("final_iou", 0.0))
    except Exception:
        iou = 0.0

    return CadFitResult(cadquery_code=code, recon_stl=recon_path, iou=iou, work_dir=stl_dir)
