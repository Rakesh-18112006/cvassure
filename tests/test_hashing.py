import json

import pytest

from cvassure.core.hashing import canonical_json, file_digest, sha256_hex


def test_key_order_does_not_change_bytes():
    a = {"zebra": 1, "alpha": {"y": [1, 2], "x": "s"}, "mid": 3.5}
    b = {}
    b["mid"] = 3.5
    b["alpha"] = {}
    b["alpha"]["x"] = "s"
    b["alpha"]["y"] = [1, 2]
    b["zebra"] = 1
    assert canonical_json(a) == canonical_json(b)
    assert sha256_hex(a) == sha256_hex(b)


def test_no_whitespace_and_utf8():
    out = canonical_json({"k": "vä", "n": 1})
    assert b" " not in out
    assert out.decode("utf-8") == '{"k":"vä","n":1}'


def test_nested_list_order_is_preserved():
    assert canonical_json({"l": [1, 2]}) != canonical_json({"l": [2, 1]})


def test_negative_zero_collapses():
    assert canonical_json({"x": -0.0}) == canonical_json({"x": 0.0})


def test_float_roundtrips_exactly():
    value = 0.1 + 0.2
    restored = json.loads(canonical_json({"x": value}))["x"]
    assert restored == value


def test_non_finite_floats_are_representable():
    out = canonical_json({"a": float("nan"), "b": float("inf")})
    assert json.loads(out) == {"a": "NaN", "b": "Infinity"}


def test_bytes_and_tuples():
    assert canonical_json({"b": b"\x00\xff"}) == b'{"b":"00ff"}'
    assert canonical_json({"t": (1, 2)}) == canonical_json({"t": [1, 2]})


def test_unserialisable_type_raises():
    with pytest.raises(TypeError):
        canonical_json({"x": object()})


def test_sha256_hex_accepts_bytes_str_and_objects():
    assert sha256_hex(b"abc") == sha256_hex("abc")
    assert len(sha256_hex({"a": 1})) == 64


def test_file_digest_matches_sha256(tmp_path):
    p = tmp_path / "weights.bin"
    p.write_bytes(b"x" * (3 * 1024 * 1024 + 7))
    assert file_digest(p) == sha256_hex(p.read_bytes())
