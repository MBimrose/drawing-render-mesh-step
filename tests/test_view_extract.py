"""Unit tests for view_extract (no LocateAnything-3B required — bbox-parsing only)."""

from drmstep import view_extract


def test_parse_bbox_picks_largest():
    resp = "found things <box>0 0 100 100</box> and <box>0 0 500 500</box> done"
    box = view_extract._parse_bbox(resp, width=1000, height=1000)
    assert box == (0, 0, 500, 500)


def test_parse_bbox_normalizes_to_pixel_space():
    resp = "<box>250 500 750 1000</box>"
    box = view_extract._parse_bbox(resp, width=800, height=400)
    assert box == (200, 200, 600, 400)


def test_parse_bbox_returns_none_on_no_match():
    assert view_extract._parse_bbox("no boxes here", 100, 100) is None


def test_pad_bbox_clamps_to_image():
    box = view_extract._pad_bbox((10, 10, 90, 90), 100, 100, frac=0.5)
    assert box == (0, 0, 100, 100)
