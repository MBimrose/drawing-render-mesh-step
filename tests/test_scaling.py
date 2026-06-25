"""Unit tests for scaling JSON parsing (no Claude required)."""

from drmstep import scaling


def test_parse_json_plain():
    assert scaling._parse_json('{"sx": 2.0, "sy": 1.5, "sz": 1.0}') == {
        "sx": 2.0, "sy": 1.5, "sz": 1.0,
    }


def test_parse_json_with_code_fence():
    text = '```json\n{"sx": 1.0, "sy": 1.0, "sz": 1.0, "unit": "mm"}\n```'
    out = scaling._parse_json(text)
    assert out and out["sx"] == 1.0 and out["unit"] == "mm"


def test_parse_json_embedded_in_prose():
    text = 'Sure, here is your scale: {"sx": 3.0, "sy": 2.0, "sz": 1.0} thanks!'
    out = scaling._parse_json(text)
    assert out == {"sx": 3.0, "sy": 2.0, "sz": 1.0}


def test_fallback_uniform():
    res = scaling._fallback_uniform(target_max=100.0, candidate_dims=(2.0, 1.0, 0.5))
    assert abs(res.sx - 50.0) < 1e-6
    assert res.sx == res.sy == res.sz
