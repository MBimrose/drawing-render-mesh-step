"""Execute CADFit's CadQuery code in a subprocess with a scale + STEP-export postlude."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)

_POSTLUDE_TEMPLATE = """
# --- drmstep postlude: uniform-scale + export STEP ---
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

# Coerce to a list of cq.Shape solids so we can boolean-union them.
# CADFit emits `result = solid_1.add(solid_2).add(solid_3)`, which collects
# overlapping solids in a Workplane WITHOUT fusing them. The resulting BREP has
# coincident/intersecting faces that turn OCC's tessellator into a black hole.
# So we explicitly fuse here.
def _collect_solids(_obj):
    if isinstance(_obj, _cq.Workplane):
        _items = [v for v in _obj.vals() if isinstance(v, _cq.Shape)]
        return [s for v in _items for s in (v.Solids() if hasattr(v, 'Solids') else [v])]
    if isinstance(_obj, _cq.Assembly):
        return _obj.toCompound().Solids()
    if isinstance(_obj, _cq.Shape):
        solids = _obj.Solids() if hasattr(_obj, 'Solids') else []
        return solids if solids else [_obj]
    if hasattr(_obj, 'wrapped'):
        return [_cq.Shape(_obj.wrapped)]
    return []

_solids = _collect_solids(_picked)
if not _solids:
    print("drmstep: result yielded no solids", file=_sys.stderr)
    _sys.exit(2)

_shape = _solids[0]
for _s2 in _solids[1:]:
    try:
        _shape = _shape.fuse(_s2)
    except Exception as _exc:
        print(f"drmstep: fuse failed ({{_exc}}); appending without union", file=_sys.stderr)
        _shape = _cq.Compound.makeCompound([_shape, _s2])

# Simplify the BREP. CADFit's emitted code traces each sketch loop with hundreds
# of tiny lineTo + threePointArc segments — after extrusion that's hundreds of
# near-tangent side faces, which torch OCC's tessellator (10+ minutes per part).
# ShapeUpgrade_UnifySameDomain merges G1-continuous adjacent faces, dropping the
# face count by ~10x without changing the shape.
try:
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    _unify = ShapeUpgrade_UnifySameDomain(
        _shape.wrapped,
        True,   # UnifyEdges
        True,   # UnifyFaces
        True,   # ConcatBSplines
    )
    _unify.SetLinearTolerance(1e-4)
    _unify.SetAngularTolerance(1e-3)
    _unify.Build()
    _unified = _unify.Shape()
    if _unified is not None:
        _shape = _cq.Shape(_unified)
        print("drmstep: ShapeUpgrade_UnifySameDomain applied")
except Exception as _exc:
    print(f"drmstep: UnifySameDomain skipped ({{_exc}})", file=_sys.stderr)

# Uniform scale via gp_Trsf — preserves aspect ratio and produces a clean BREP
# the evaluator can tessellate (non-uniform gp_GTrsf warps shapes in ways that
# frequently trip the tessellation timeout).
_s = float({s})
if abs(_s - 1.0) > 1e-9:
    try:
        _shape = _shape.scale(_s)
    except Exception as _exc:
        print(f"drmstep: uniform scale failed ({{_exc}}); exporting unscaled",
              file=_sys.stderr)

_cq.exporters.export(_shape, "{output_step}")
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
    """Append a uniform-scale + STEP-export postlude to ``code`` and run it.

    ``scale`` is a 3-tuple for backwards compatibility, but only ``sx`` is used —
    the postlude applies a single uniform scale via ``cq.Shape.scale``. Returns
    the produced STEP path. Raises RunnerError on failure.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    sx, sy, sz = scale
    # If the caller passed a non-uniform tuple, collapse to the largest factor
    # (we no longer emit non-uniform scales, but legacy callers might).
    s = max(abs(sx), abs(sy), abs(sz)) if abs(sx) > 0 else 1.0
    output_step = output_step.resolve()
    augmented = code + _POSTLUDE_TEMPLATE.format(
        s=s, output_step=str(output_step).replace("\\", "\\\\")
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
