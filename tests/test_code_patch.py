"""Unit tests for code_patch extraction (no Claude required)."""

from drmstep import code_patch


def test_extract_code_from_python_block():
    text = "Here you go:\n```python\nimport cadquery as cq\nresult = cq.Workplane('XY').box(1,1,1)\n```\nDone."
    out = code_patch._extract_code(text)
    assert out is not None
    assert "result" in out and "cadquery" in out


def test_extract_code_from_bare_block():
    text = "```\nimport cadquery as cq\nresult = cq.Workplane().box(2,2,2)\n```"
    out = code_patch._extract_code(text)
    assert out is not None and "result" in out


def test_extract_code_returns_none_on_noise():
    text = "I cannot do that."
    assert code_patch._extract_code(text) is None
