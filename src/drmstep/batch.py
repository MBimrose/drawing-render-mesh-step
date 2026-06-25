"""Batch runner: serial GPU, parallel CPU.

The pipeline has two phases per sample:
  1. GPU phase: view extraction (LocateAnything-3B) + mesh generation (Hunyuan3D-2).
     These share one GPU and cannot run in parallel without OOM.
  2. CPU phase: CADFit (subprocess) + scaling (VLM HTTP) + simplify + execute.
     CADFit is CPU-heavy and embarrassingly parallel across samples.

This module runs phase 1 serially for all samples, accumulating (sample, mesh_path,
drawing, work_dir) tuples. It then dispatches phase 2 as a ThreadPoolExecutor (HTTP
for VLM, subprocess for CADFit — both release the GIL).

Usage:
    python -m drmstep.batch --samples 101,102,103 --out runs/batch
    python -m drmstep.batch --all --out runs/batch --parallel 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import yaml
from PIL import Image
from rich.console import Console
from rich.table import Table

from .config import Config, load_config
from . import cad_fit, cad_simplify, mesh_generate, runners, scaling, view_extract

logger = logging.getLogger("drmstep.batch")
console = Console()


def _resolve_inputs_dir(inputs: Path | None) -> Path:
    if inputs is not None:
        return inputs.resolve()
    try:
        from cadgenbench.common.paths import data_inputs_dir
        return data_inputs_dir()
    except Exception as exc:
        raise FileNotFoundError(f"--inputs not given and cadgenbench.data_inputs_dir() failed: {exc}") from exc


def _discover_samples(inputs_root: Path, names: list[str] | None, limit: int | None) -> list[str]:
    all_samples = sorted(
        p.name for p in inputs_root.iterdir() if (p / "description.yaml").exists()
    )
    if names:
        wanted = {s.strip() for s in names if s.strip()}
        all_samples = [s for s in all_samples if s in wanted]
    if limit:
        all_samples = all_samples[:limit]
    return all_samples


def _gpu_phase(sample: str, inputs_dir: Path, work_dir: Path, config: Config) -> dict:
    """Run view extraction + Hunyuan3D mesh generation (GPU-bound, serial)."""
    work_dir.mkdir(parents=True, exist_ok=True)

    desc = yaml.safe_load((inputs_dir / "description.yaml").read_text()) or {}
    task_description = desc.get("task_description", desc.get("description", ""))
    input_png = inputs_dir / "input.png"
    if not input_png.exists():
        candidates = [c for c in inputs_dir.glob("input.*")
                      if c.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if not candidates:
            raise FileNotFoundError(f"no input image in {inputs_dir}")
        input_png = candidates[0]

    drawing = Image.open(input_png).convert("RGB")

    # 1. view extraction (GPU)
    t = time.time()
    extracted = view_extract.extract_isometric_view(drawing, config)
    import io
    buf = io.BytesIO()
    extracted.image.save(buf, format="PNG")
    (work_dir / "iso_crop.png").write_bytes(buf.getvalue())
    (work_dir / "view_response.txt").write_text(extracted.raw_response)
    logger.info("[%s] view extraction: %.1fs", sample, time.time() - t)

    # 2. mesh generation (GPU)
    t = time.time()
    mesh_stl = mesh_generate.image_to_mesh(extracted.image, work_dir / "mesh.stl", config)
    logger.info("[%s] mesh generation: %.1fs", sample, time.time() - t)

    return {
        "sample": sample,
        "mesh_stl": str(mesh_stl),
        "drawing_path": str(input_png),
        "task_description": task_description,
        "work_dir": str(work_dir),
    }


def _cpu_phase(gpu_result: dict, out_dir: Path, config: Config) -> dict:
    """Run CADFit + scaling + simplify + execute (CPU + HTTP, parallelizable)."""
    sample = gpu_result["sample"]
    mesh_stl = Path(gpu_result["mesh_stl"])
    work_dir = Path(gpu_result["work_dir"])
    drawing_path = Path(gpu_result["drawing_path"])
    task_description = gpu_result["task_description"]

    from PIL import Image
    drawing = Image.open(drawing_path).convert("RGB")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. CADFit (CPU subprocess)
    t = time.time()
    try:
        cadfit = cad_fit.run_cadfit(mesh_stl, work_dir / "cadfit", config)
        logger.info("[%s] CADFit: %.1fs, IoU=%.3f", sample, time.time() - t, cadfit.iou)
    except cad_fit.CadFitError as exc:
        logger.warning("[%s] CADFit failed (%s); using Hunyuan mesh", sample, exc)
        scale = scaling.compute_scale(drawing, task_description, mesh_stl, config)
        (work_dir / "scale.json").write_text(json.dumps(asdict(scale), indent=2))
        output_stl = out_dir / "output.stl"
        _scale_stl_and_save(mesh_stl, output_stl, (scale.sx, scale.sy, scale.sz))
        return {"sample": sample, "status": "hunyuan_only",
                "output_path": str(output_stl), "cadfit_iou": None,
                "scale": (scale.sx, scale.sy, scale.sz), "notes": str(exc)[:200]}

    # 4. scaling (HTTP VLM)
    t = time.time()
    scale = scaling.compute_scale(drawing, task_description, cadfit.recon_stl, config)
    (work_dir / "scale.json").write_text(json.dumps(asdict(scale), indent=2))
    logger.info("[%s] scaling: %.1fs, s=%.2f", sample, time.time() - t, scale.sx)

    # 5. simplify + execute
    simplified, stats = cad_simplify.simplify_cadfit_code(cadfit.cadquery_code)
    (work_dir / "cadfit_simplified.py").write_text(simplified)
    logger.info("[%s] simplify: %d → %d segments", sample,
                stats["segments_before"], stats["segments_after"])

    output_step = out_dir / "output.step"
    try:
        runners.execute_cadquery(simplified, output_step,
                                 (scale.sx, scale.sy, scale.sz), config, work_dir)
        return {"sample": sample, "status": "ok",
                "output_path": str(output_step), "cadfit_iou": cadfit.iou,
                "scale": (scale.sx, scale.sy, scale.sz), "notes": ""}
    except runners.RunnerError as exc:
        logger.warning("[%s] cq exec failed; STL fallback", sample)
        output_stl = out_dir / "output.stl"
        _scale_stl_and_save(cadfit.recon_stl, output_stl, (scale.sx, scale.sy, scale.sz))
        return {"sample": sample, "status": "stl_fallback",
                "output_path": str(output_stl), "cadfit_iou": cadfit.iou,
                "scale": (scale.sx, scale.sy, scale.sz), "notes": str(exc)[:200]}


def _scale_stl_and_save(src: Path, dst: Path, scale: tuple[float, float, float]) -> None:
    import trimesh
    mesh = trimesh.load(src, force="mesh")
    mesh.apply_scale(scale)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)


def run_batch(
    samples: list[str],
    inputs_root: Path,
    out_root: Path,
    config: Config,
    parallel: int = 4,
) -> list[dict]:
    """Run the full pipeline on multiple samples.

    Phase 1 (GPU): serial — view extraction + Hunyuan3D for each sample.
    Phase 2 (CPU): parallel — CADFit + scaling + simplify + execute.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    # Phase 1: GPU serial
    gpu_results: list[dict] = []
    console.print(f"[bold]Phase 1: GPU (serial) — {len(samples)} samples[/bold]")
    for i, sample in enumerate(samples):
        sample_inputs = inputs_root / sample
        sample_out = out_root / sample
        work_dir = config.work_root / sample
        console.print(f"  [{i+1}/{len(samples)}] {sample}: view + mesh... ", end="")
        try:
            gr = _gpu_phase(sample, sample_inputs, work_dir, config)
            gpu_results.append(gr)
            console.print(f"done (mesh: {Path(gr['mesh_stl']).stat().st_size // 1024}KB)")
        except Exception as exc:
            console.print(f"FAILED: {exc}")
            results.append({"sample": sample, "status": "gpu_failed", "error": str(exc)[:300]})

    # Phase 2: CPU parallel
    console.print(f"\n[bold]Phase 2: CPU (parallel={parallel}) — {len(gpu_results)} samples[/bold]")
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {}
        for gr in gpu_results:
            sample = gr["sample"]
            sample_out = out_root / sample
            future = pool.submit(_cpu_phase, gr, sample_out, config)
            futures[future] = sample

        for future in as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
                results.append(result)
                console.print(f"  {sample}: {result['status']}"
                             f" (IoU={result.get('cadfit_iou', '?')})"
                             f" → {Path(result['output_path']).name}")
            except Exception as exc:
                console.print(f"  {sample}: CRASHED: {exc}")
                results.append({"sample": sample, "status": "crashed", "error": str(exc)[:300]})

    # Summary
    summary_path = out_root / "_batch_results.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))

    table = Table(title="Batch Summary")
    table.add_column("sample")
    table.add_column("status")
    table.add_column("IoU")
    table.add_column("output")
    for r in sorted(results, key=lambda x: x["sample"]):
        table.add_row(
            r["sample"],
            r["status"],
            f"{r.get('cadfit_iou', 0):.3f}" if r.get("cadfit_iou") else "-",
            Path(r["output_path"]).name if "output_path" in r else "-",
        )
    console.print(table)
    console.print(f"\nresults: {summary_path}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch pipeline: serial GPU, parallel CPU")
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated fixture names (e.g. 101,102,103)")
    parser.add_argument("--all", action="store_true", help="Process all fixtures")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of fixtures")
    parser.add_argument("--inputs", type=Path, default=None, help="Override inputs dir")
    parser.add_argument("--out", type=Path, required=True, help="Output dir")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Parallel CADFit workers (CPU phase)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    inputs_root = _resolve_inputs_dir(args.inputs)
    samples = _discover_samples(inputs_root, args.samples.split(",") if args.samples else None,
                                args.limit if not args.all else None)
    if not samples:
        console.print("[red]No samples matched.[/red]")
        return 1

    config = load_config()
    console.print(f"Running {len(samples)} samples, parallel={args.parallel}")
    run_batch(samples, inputs_root, args.out, config, parallel=args.parallel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
