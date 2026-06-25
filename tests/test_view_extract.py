"""Unit tests for view_extract (no model calls — bbox-parsing only)."""

from drmstep import view_extract


def test_parse_json_bbox_array_form():
    text = '{"bbox": [100, 50, 400, 300]}'
    assert view_extract._parse_json_bbox(text) == (100, 50, 400, 300)


def test_parse_json_bbox_x1y1_form():
    text = '{"x1": 100, "y1": 50, "x2": 400, "y2": 300}'
    assert view_extract._parse_json_bbox(text) == (100, 50, 400, 300)


def test_parse_json_bbox_with_fence():
    text = '```json\n{"bbox": [10, 20, 30, 40]}\n```'
    assert view_extract._parse_json_bbox(text) == (10, 20, 30, 40)


def test_parse_json_bbox_embedded_in_prose():
    text = 'The isometric is at {"bbox": [1, 2, 5, 9]} done'
    assert view_extract._parse_json_bbox(text) == (1, 2, 5, 9)


def test_parse_json_bbox_bare_array_fallback():
    text = 'The isometric is roughly [750, 120, 1080, 500].'
    assert view_extract._parse_json_bbox(text) == (750, 120, 1080, 500)


def test_parse_json_bbox_handles_mangled_form():
    """Qwen sometimes emits malformed JSON like {\"x1\": [a, b, c, d}.
    The bare-array fallback should still pick it up."""
    text = '{"x1": [750, 120, 1080, 500}'
    assert view_extract._parse_json_bbox(text) == (750, 120, 1080, 500)


def test_parse_json_bbox_rejects_degenerate():
    text = '{"bbox": [0, 0, 0, 0]}'
    assert view_extract._parse_json_bbox(text) is None


def test_parse_json_bbox_returns_none_on_nonsense():
    assert view_extract._parse_json_bbox("no json here") is None


def test_parse_la_bbox_angle_bracket_format():
    resp = "<box><314><170><863><650></box>"
    box = view_extract._parse_la_bbox(resp, 1000, 1000)
    assert box == (314, 170, 863, 650)


def test_parse_la_bbox_picks_largest_nondegenerate():
    """Real responses interleave one real box with many <0 0 999 999> spam boxes."""
    resp = "<box><314><170><863><650></box>" + "<box><0><0><999><999></box>" * 30
    box = view_extract._parse_la_bbox(resp, 1000, 1000)
    assert box == (314, 170, 863, 650)


def test_parse_la_bbox_whitespace_form_still_works():
    resp = "<box>0 0 100 100</box> and <box>0 0 500 500</box>"
    box = view_extract._parse_la_bbox(resp, 1000, 1000)
    assert box == (0, 0, 500, 500)


def test_pad_bbox_clamps_to_image():
    box = view_extract._pad_bbox((10, 10, 90, 90), 100, 100, frac=0.5)
    assert box == (0, 0, 100, 100)
