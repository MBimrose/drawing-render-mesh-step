"""drmstep CLI: run-sample, run-bench, score."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from . import edit_pipeline, pipeline
from .config import load_config

app = typer.Typer(add_completion=False, help="drawing-render-mesh-step pipeline CLI")
console = Console()
logger = logging.getLogger("drmstep")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _resolve_inputs_dir(inputs: Optional[Path]) -> Path:
    if inputs is not None:
        return inputs.resolve()
    try:
        from cadgenbench.common.paths import data_inputs_dir
        return data_inputs_dir()
    except Exception as exc:
        raise typer.BadParameter(
            f"--inputs not given and cadgenbench.data_inputs_dir() failed: {exc}"
        ) from exc


def _task_type(inputs_dir: Path, name: str) -> str:
    desc = inputs_dir / name / "description.yaml"
    data = yaml.safe_load(desc.read_text()) or {}
    return str(data.get("task_type", "generation"))


def _run_one(sample: str, inputs_root: Path, out_root: Path) -> dict:
    config = load_config()
    sample_inputs = inputs_root / sample
    sample_out = out_root / sample
    ttype = _task_type(inputs_root, sample)
    try:
        if ttype == "editing":
            res = edit_pipeline.run_edit(sample, sample_inputs, sample_out, config)
        else:
            res = pipeline.run_generation(sample, sample_inputs, sample_out, config)
        return {"task_type": ttype, **{k: str(v) if isinstance(v, Path) else v
                                      for k, v in asdict(res).items()}}
    except Exception as exc:
        logger.exception("[%s] pipeline crashed", sample)
        return {"sample": sample, "task_type": ttype, "status": "crashed", "error": str(exc)}


@app.command("run-sample")
def run_sample(
    sample: str = typer.Option(..., "--sample", help="Fixture name"),
    inputs: Optional[Path] = typer.Option(None, "--inputs", help="Override inputs dir"),
    out: Path = typer.Option(..., "--out", help="Output dir (gets <sample>/output.step)"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Run the pipeline on a single sample."""
    _setup_logging(verbose)
    inputs_root = _resolve_inputs_dir(inputs)
    out.mkdir(parents=True, exist_ok=True)
    result = _run_one(sample, inputs_root, out)
    print(json.dumps(result, indent=2, default=str))


@app.command("run-bench")
def run_bench(
    inputs: Optional[Path] = typer.Option(None, "--inputs", help="Override inputs dir"),
    out: Path = typer.Option(..., "--out", help="Submission output dir"),
    parallel: int = typer.Option(1, "--parallel", min=1, help="Concurrent samples"),
    samples: Optional[str] = typer.Option(None, "--samples", help="Comma-sep fixture names"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Cap number of fixtures"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Run the pipeline on the whole benchmark (or a subset)."""
    _setup_logging(verbose)
    inputs_root = _resolve_inputs_dir(inputs)
    all_samples = sorted(p.name for p in inputs_root.iterdir()
                          if (p / "description.yaml").exists())
    if samples:
        wanted = {s.strip() for s in samples.split(",") if s.strip()}
        all_samples = [s for s in all_samples if s in wanted]
    if limit:
        all_samples = all_samples[:limit]
    if not all_samples:
        raise typer.BadParameter("no samples matched")

    console.print(f"running {len(all_samples)} samples with parallel={parallel}")
    out.mkdir(parents=True, exist_ok=True)

    results = []
    if parallel == 1:
        for s in all_samples:
            results.append(_run_one(s, inputs_root, out))
            console.print(f"[bold]{s}[/bold]: {results[-1].get('status')}")
    else:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_run_one, s, inputs_root, out): s for s in all_samples}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {"sample": s, "status": "crashed", "error": str(exc)}
                results.append(res)
                console.print(f"[bold]{s}[/bold]: {res.get('status')}")

    summary_path = out / "_drmstep_run.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    console.print(f"\nwrote {summary_path}")


@app.command("score")
def score(
    submission: Path = typer.Option(..., "--submission", help="Submission dir"),
    inputs: Optional[Path] = typer.Option(None, "--inputs"),
    gt: Optional[Path] = typer.Option(None, "--gt"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Run ``cadgenbench evaluate`` against a submission dir and print highlights."""
    _setup_logging(verbose)
    cmd = ["cadgenbench", "evaluate", str(submission)]
    env = None  # let user set CADGENBENCH_DATA_REPO / DIR
    console.print(f"running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise typer.Exit(proc.returncode)

    summary_path = submission / "run_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        table = Table(title="run_summary.json highlights")
        table.add_column("metric")
        table.add_column("value")
        table.add_row("aggregate_score", f"{data.get('aggregate_score', 0):.3f}")
        table.add_row("validity_rate", f"{data.get('validity_rate', 0):.3f}")
        table.add_row("n_samples", str(data.get('n_samples', '?')))
        console.print(table)


if __name__ == "__main__":
    app()
