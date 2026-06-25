"""Execute CADFit's CadQuery code in a subprocess with a scale + STEP-export postlude."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)

_POSTLUDE_TEMPLATE = """
# --- drmstep postlude: scale + export STEP ---
import cadquery as _cq
import sys as _sys

_candidates = []
for _name in list(globals()):
    if _name.startswith('_'):
        continue
    _obj = globals()[_name]
    if isinstance(_obj, (_cq.Workplane, _cq.Shape, _cq.Compound, _cq.Solid, _cq.Assembly)):
        _candidates.append((_name, _obj))

if not _candidates:
    print("drmstep: no cadquery result found in module scope", file=_sys.stderr)
    _sys.exit(2)

_preferred = ['result', 'solid', 'shape', 'part', 'model', 'final', 'output']
_picked = None
for _pn in _preferred:
    for _n, _o in _candidates:
        if _n == _pn:
            _picked = _o
            break
    if _picked is not None:
        break
if _picked is None:
    _picked = _candidates[-1][1]

# Coerce to a cq.Shape.
if isinstance(_picked, _cq.Assembly):
    _shape = _picked.toCompound()
elif isinstance(_picked, _cq.Workplane):
    _shape = _picked.val()
else:
    _shape = _picked
if not isinstance(_shape, _cq.Shape):
    _shape = _cq.Shape(_shape.wrapped) if hasattr(_shape, 'wrapped') else _shape

_sx, _sy, _sz = {sx}, {sy}, {sz}

_scaled = _shape
if abs(_sx - 1.0) > 1e-9 or abs(_sy - 1.0) > 1e-9 or abs(_sz - 1.0) > 1e-9:
    try:
        if abs(_sx - _sy) < 1e-9 and abs(_sy - _sz) < 1e-9:
            # Uniform: cq.Shape.scale exists and is robust.
            _scaled = _shape.scale(_sx) if hasattr(_shape, 'scale') else _shape
        else:
            # Non-uniform: apply gp_GTrsf via BRepBuilderAPI_GTransform.
            from OCP.gp import gp_GTrsf, gp_Mat, gp_XYZ
            from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
            _mat = gp_Mat(_sx, 0, 0, 0, _sy, 0, 0, 0, _sz)
            _trsf = gp_GTrsf()
            _trsf.SetVectorialPart(_mat)
            _trsf.SetTranslationPart(gp_XYZ(0, 0, 0))
            _builder = BRepBuilderAPI_GTransform(_shape.wrapped, _trsf, True)
            _builder.Build()
            if _builder.IsDone():
                _scaled = _cq.Shape(_builder.Shape())
            else:
                print("drmstep: GTransform did not complete; exporting unscaled", file=_sys.stderr)
    except Exception as _exc:
        print(f"drmstep: scale failed ({{_exc}}); exporting unscaled", file=_sys.stderr)

_cq.exporters.export(_scaled, "{output_step}")
print("drmstep: wrote {output_step}")
"""


class RunnerError(RuntimeError):
    pass


def execute_cadquery(
    code: str,
    output_step: Path,
    scale: tuple[float, float, float],
    config: Config,
    work_dir: Path,
) -> Path:
    """Append a scale + STEP-export postlude to ``code`` and execute in a subprocess.

    Returns the produced STEP path. Raises RunnerError on failure.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    sx, sy, sz = scale
    augmented = code + _POSTLUDE_TEMPLATE.format(
        sx=sx, sy=sy, sz=sz, output_step=str(output_step).replace("\\", "\\\\")
    )
    script = work_dir / "_drmstep_run.py"
    script.write_text(augmented)

    cmd = [str(config.cadquery_python), str(script)]
    logger.info("execute cadquery: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, cwd=work_dir, capture_output=True, text=True,
            timeout=config.cadquery_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RunnerError(f"cadquery execution timed out: {exc}") from exc

    if proc.returncode != 0:
        logger.error("cadquery stdout:\n%s", proc.stdout[-2000:])
        logger.error("cadquery stderr:\n%s", proc.stderr[-2000:])
        raise RunnerError(f"cadquery exited {proc.returncode}")

    if not output_step.exists():
        raise RunnerError(f"expected {output_step} not produced")
    return output_step
