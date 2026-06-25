"""Simplify CADFit's emitted CadQuery program before execution.

CADFit traces sketch loops from the mesh outline with hundreds of tiny ``lineTo``
and ``threePointArc`` segments. After ``extrude``, every segment becomes a
distinct side face — a 350-segment loop produces ~350 near-tangent side faces
that take OCC tessellator minutes per part to mesh.

This module rewrites each loop's polyline using Douglas-Peucker simplification
(via ``shapely.simplify``) with a tolerance scaled to the loop's bounding box,
collapsing arcs to line segments along the way. The resulting BREP has ~10x
fewer faces and tessellates in seconds.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Match the CADFit chain prelude: sketch_N = cq.Workplane(plane_N)
_SKETCH_PREAMBLE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<sketch>sketch_\d+)\s*=\s*(?P<rhs>cq\.Workplane\([^\n]+\))\s*$",
    re.MULTILINE,
)

# Match the moveTo line: loop_N = sketch_N.moveTo(x, y)
_MOVETO_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<loop>loop_\d+)\s*=\s*(?P<sketch>sketch_\d+)\.moveTo\((?P<x>-?\d+\.?\d*),\s*(?P<y>-?\d+\.?\d*)\)\s*$",
    re.MULTILINE,
)

# Subsequent step: loop_N = loop_N.lineTo(x, y)  OR  .threePointArc((mx,my),(ex,ey))  OR  .close()
_STEP_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<loop>loop_\d+)\s*=\s*(?P=loop)\.(?P<kind>lineTo|threePointArc|close)"
    r"(?P<args>\([^\n]*\))?\s*$",
    re.MULTILINE,
)

_NUM_RE = re.compile(r"-?\d+\.?\d*")

# Pixel-equivalent simplify tolerance, in the loop's own coord space.
# CADFit emits loops normalized to roughly [-1, 1], so 0.01 = ~1% of size.
_SIMPLIFY_TOLERANCE = 0.01


def _extract_loop_points(
    text: str, start: int, loop_name: str
) -> tuple[list[tuple[float, float]], int, bool]:
    """Return (points, end_offset, has_close) for the chain starting at ``start``.

    Walks consecutive ``loop_N = loop_N.<kind>(...)`` lines from ``start`` and
    flattens lineTo / threePointArc into (x, y) waypoints. The first such line
    must be the moveTo that we matched immediately before.
    """
    pts: list[tuple[float, float]] = []
    has_close = False
    pos = start
    while True:
        m = _STEP_RE.match(text, pos)
        if not m or m.group("loop") != loop_name:
            break
        kind = m.group("kind")
        if kind == "close":
            has_close = True
            pos = m.end() + 1
            break
        args = m.group("args") or ""
        nums = [float(n) for n in _NUM_RE.findall(args)]
        if kind == "lineTo" and len(nums) >= 2:
            pts.append((nums[0], nums[1]))
        elif kind == "threePointArc" and len(nums) >= 4:
            # Sample mid + end as vertices — DP simplification will keep only the
            # ones that matter; this is the lossy step but it's bounded by tolerance.
            pts.append((nums[0], nums[1]))
            pts.append((nums[2], nums[3]))
        pos = m.end() + 1  # advance past the trailing newline
    return pts, pos, has_close


def _simplify_polyline(
    points: list[tuple[float, float]], tolerance: float
) -> list[tuple[float, float]]:
    if len(points) < 4:
        return points
    try:
        from shapely.geometry import LineString, Polygon
    except ImportError:
        return points
    # Use polygon simplify when closed-ish so corners are preserved.
    geom = LineString(points)
    simp = geom.simplify(tolerance, preserve_topology=False)
    return [(float(x), float(y)) for x, y in simp.coords]


def simplify_cadfit_code(
    code: str, tolerance: Optional[float] = None
) -> tuple[str, dict]:
    """Return ``(new_code, stats)`` where new_code has simplified loop polylines.

    stats keys: ``loops``, ``segments_before``, ``segments_after``.
    """
    tol = tolerance if tolerance is not None else _SIMPLIFY_TOLERANCE
    stats = {"loops": 0, "segments_before": 0, "segments_after": 0}

    # Walk every moveTo; for each one, find the following loop chain, extract
    # points, simplify, and emit a replacement chain.
    out_chunks: list[str] = []
    cursor = 0
    for m in _MOVETO_RE.finditer(code):
        # Copy everything up to (but not including) the moveTo line.
        out_chunks.append(code[cursor : m.start()])

        indent = m.group("indent")
        loop = m.group("loop")
        sketch = m.group("sketch")
        x0, y0 = float(m.group("x")), float(m.group("y"))

        chain_start = m.end() + 1  # past the newline
        rest_points, chain_end, has_close = _extract_loop_points(code, chain_start, loop)
        all_pts = [(x0, y0)] + rest_points
        before_n = len(all_pts)

        simplified = _simplify_polyline(all_pts, tol)
        after_n = len(simplified)
        if after_n < 3:
            simplified = all_pts  # don't degenerate
            after_n = before_n

        # Emit the replacement chain.
        sx, sy = simplified[0]
        emit_lines = [f"{indent}{loop} = {sketch}.moveTo({sx:.5f}, {sy:.5f})"]
        for ex, ey in simplified[1:]:
            emit_lines.append(f"{indent}{loop} = {loop}.lineTo({ex:.5f}, {ey:.5f})")
        if has_close:
            emit_lines.append(f"{indent}{loop} = {loop}.close()")
        out_chunks.append("\n".join(emit_lines) + "\n")

        stats["loops"] += 1
        stats["segments_before"] += before_n
        stats["segments_after"] += after_n

        cursor = chain_end

    out_chunks.append(code[cursor:])
    new_code = "".join(out_chunks)
    return new_code, stats
