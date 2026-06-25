"""Generation pipeline: drawing -> isometric crop -> mesh -> CadQuery -> scaled STEP."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import trimesh
import yaml
from PIL import Image

from . import cad_fit, cad_simplify, mesh_generate, runners, scaling, view_extract
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    sample: str
    status: str            # "ok", "stl_fallback", "failed"
    output_path: Path
    cadfit_iou: float | None = None
    scale: tuple[float, float, float] | None = None
    notes: str = ""


def _load_description(inputs_dir: Path) -> dict:
    desc = yaml.safe_load((inputs_dir / "description.yaml").read_text())
    return desc or {}


def run_generation(
    sample: str,
    inputs_dir: Path,
    out_dir: Path,
    config: Config,
) -> SampleResult:
    """Run the full generation pipeline on one sample.

    Args:
        sample: fixture name.
        inputs_dir: directory containing ``description.yaml`` + ``input.png``.
        out_dir: where ``output.step`` (or ``output.stl`` fallback) goes.
        config: pipeline config.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    work = config.work_root / sample
    work.mkdir(parents=True, exist_ok=True)

    desc = _load_description(inputs_dir)
    task_description = desc.get("task_description", "")
    input_png = inputs_dir / "input.png"
    if not input_png.exists():
        # Some samples might use other extensions
        candidates = list(inputs_dir.glob("input.*"))
        candidates = [c for c in candidates if c.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if not candidates:
            raise FileNotFoundError(f"no input image in {inputs_dir}")
        input_png = candidates[0]

    drawing = Image.open(input_png).convert("RGB")

    # 1. view extraction
    logger.info("[%s] view extraction", sample)
    extracted = view_extract.extract_isometric_view(drawing, config)
    (work / "iso_crop.png").write_bytes(_pil_to_png_bytes(extracted.image))
    (work / "view_response.txt").write_text(extracted.raw_response)

    # 2. mesh generation
    logger.info("[%s] mesh generation", sample)
    mesh_stl = mesh_generate.image_to_mesh(extracted.image, work / "mesh.stl", config)

    # 3. CAD fitting. If CADFit can't fit primitives to the part, retry once on
    # a more aggressively simplified mesh (coarser alpha-wrap + largest body
    # only) — this fixes parts where Hunyuan3D produces multi-body / high-genus
    # output (e.g. sheet metal). Only if both attempts fail do we route to STL.
    logger.info("[%s] CADFit", sample)
    cadfit = None
    try:
        cadfit = cad_fit.run_cadfit(mesh_stl, work / "cadfit", config)
    except cad_fit.CadFitError as exc:
        logger.warning("[%s] CADFit failed (%s); retrying on simplified mesh", sample, exc)
        try:
            simplified = mesh_generate.simplify_for_retry(mesh_stl, work / "mesh_simplified.stl")
            cadfit = cad_fit.run_cadfit(simplified, work / "cadfit_retry", config)
            logger.info("[%s] CADFit retry succeeded, IoU=%.3f", sample, cadfit.iou)
        except cad_fit.CadFitError as exc2:
            logger.warning("[%s] CADFit retry also failed (%s); using Hunyuan mesh", sample, exc2)
            scale = scaling.compute_scale(drawing, task_description, mesh_stl, config)
            (work / "scale.json").write_text(json.dumps(asdict(scale), indent=2))
            output_stl = out_dir / "output.stl"
            _scale_stl_and_save(mesh_stl, output_stl, (scale.sx, scale.sy, scale.sz))
            return SampleResult(
                sample=sample, status="hunyuan_only", output_path=output_stl,
                cadfit_iou=None, scale=(scale.sx, scale.sy, scale.sz),
                notes=f"cadfit: {exc2}",
            )

    # 4. scaling
    logger.info("[%s] scaling", sample)
    scale = scaling.compute_scale(drawing, task_description, cadfit.recon_stl, config)
    (work / "scale.json").write_text(json.dumps(asdict(scale), indent=2))

    # 5. simplify CADFit's loops so the extruded BREP is tessellation-friendly
    # (CADFit's per-loop polylines are 300+ tiny segments; without DP simplification
    # the OCC tessellator takes 10+ minutes per part). Then execute -> output.step.
    simplified_code, simp_stats = cad_simplify.simplify_cadfit_code(cadfit.cadquery_code)
    logger.info(
        "[%s] cad_simplify: %d loops, %d → %d segments",
        sample, simp_stats["loops"], simp_stats["segments_before"], simp_stats["segments_after"],
    )
    (work / "cadfit_simplified.py").write_text(simplified_code)

    output_step = out_dir / "output.step"
    try:
        runners.execute_cadquery(
            simplified_code, output_step,
            (scale.sx, scale.sy, scale.sz), config, work,
        )
        return SampleResult(
            sample=sample, status="ok", output_path=output_step,
            cadfit_iou=cadfit.iou, scale=(scale.sx, scale.sy, scale.sz),
        )
    except runners.RunnerError as exc:
        logger.warning("[%s] cadquery exec failed (%s); falling back to scaled STL", sample, exc)
        output_stl = out_dir / "output.stl"
        _scale_stl_and_save(cadfit.recon_stl, output_stl, (scale.sx, scale.sy, scale.sz))
        return SampleResult(
            sample=sample, status="stl_fallback", output_path=output_stl,
            cadfit_iou=cadfit.iou, scale=(scale.sx, scale.sy, scale.sz),
            notes=f"cq error: {exc}",
        )


def _pil_to_png_bytes(image: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _scale_stl_and_save(src: Path, dst: Path, scale: tuple[float, float, float]) -> None:
    mesh = trimesh.load(src, force="mesh")
    mesh.apply_scale(scale)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)
