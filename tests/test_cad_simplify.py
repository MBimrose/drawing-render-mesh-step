"""Tests for CADFit loop simplification (parser + Douglas-Peucker pass)."""

from drmstep import cad_simplify


def test_simplify_collinear_lineTo_drops_midpoints():
    code = (
        "import cadquery as cq\n"
        "plane_1 = cq.Plane(origin=(0,0,0), normal=(0,0,1), xDir=(1,0,0))\n"
        "sketch_1 = cq.Workplane(plane_1)\n"
        "loop_1 = sketch_1.moveTo(0.0, 0.0)\n"
        "loop_1 = loop_1.lineTo(1.0, 0.0)\n"
        "loop_1 = loop_1.lineTo(2.0, 0.0)\n"
        "loop_1 = loop_1.lineTo(3.0, 0.0)\n"
        "loop_1 = loop_1.lineTo(3.0, 1.0)\n"
        "loop_1 = loop_1.lineTo(0.0, 1.0)\n"
        "loop_1 = loop_1.close()\n"
        "sketch_1 = sketch_1.add(loop_1)\n"
        "solid_1 = sketch_1.extrude(0.5)\n"
        "result = solid_1\n"
    )
    new_code, stats = cad_simplify.simplify_cadfit_code(code, tolerance=0.01)
    assert stats["loops"] == 1
    # Original: 6 vertices (moveTo + 5 lineTo). Three midpoints on the bottom edge
    # are collinear with start+end so DP should drop them.
    assert stats["segments_before"] == 6
    assert stats["segments_after"] < stats["segments_before"]
    # Result still mentions loop_1.close() and the corner (3.0, 1.0)
    assert ".close()" in new_code
    assert "3.00000, 1.00000" in new_code


def test_simplify_preserves_chain_outside_loop():
    code = (
        "x = 5\n"
        "sketch_2 = cq.Workplane(plane_2)\n"
        "loop_2 = sketch_2.moveTo(0.0, 0.0)\n"
        "loop_2 = loop_2.lineTo(0.0, 1.0)\n"
        "loop_2 = loop_2.close()\n"
        "solid_2 = sketch_2.extrude(1.0)\n"
        "y = 7\n"
    )
    new_code, stats = cad_simplify.simplify_cadfit_code(code)
    assert stats["loops"] == 1
    assert "x = 5" in new_code
    assert "y = 7" in new_code
    assert "solid_2 = sketch_2.extrude(1.0)" in new_code


def test_simplify_handles_threepoint_arc_as_waypoints():
    code = (
        "sketch_3 = cq.Workplane(plane_3)\n"
        "loop_3 = sketch_3.moveTo(0.0, 0.0)\n"
        "loop_3 = loop_3.threePointArc((1.0, 0.5), (2.0, 0.0))\n"
        "loop_3 = loop_3.lineTo(2.0, 1.0)\n"
        "loop_3 = loop_3.close()\n"
    )
    new_code, stats = cad_simplify.simplify_cadfit_code(code, tolerance=0.001)
    # moveTo + (arc mid, arc end) + lineTo = 4 waypoints
    assert stats["segments_before"] == 4
    # threePointArc becomes lineTo's in the output
    assert "threePointArc" not in new_code
    assert ".close()" in new_code
