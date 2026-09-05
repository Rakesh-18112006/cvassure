"""Canonical serialisation and hashing.

Everything in cvassure that is signed, chained or digested goes through
``canonical_json`` first. Two dictionaries with the same content must produce
byte-identical output regardless of how they were built, otherwise signatures
and hash chains become order-dependent and the whole provenance story falls
apart.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def _stabilise(obj: Any) -> Any:
    """Recursively replace values with a stable, JSON-safe representation.

    Floats are the tricky part: ``repr`` of a float is platform stable in
    CPython, but ``-0.0``/``0.0`` and integral floats must not flip between
    runs, and NaN/Infinity are not valid JSON at all.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if obj == 0.0:
            return 0.0  # collapse -0.0
        # 17 significant digits round-trips an IEEE-754 double exactly.
        return float(f"{obj:.17g}")
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, dict):
        return {str(k): _stabilise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stabilise(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_stabilise(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return _stabilise(obj.to_dict())
    # numpy scalars and anything else that quacks like a number
    if hasattr(obj, "item"):
        return _stabilise(obj.item())
    raise TypeError(f"canonical_json cannot serialise {type(obj).__name__}")


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, no whitespace, UTF-8."""
    return json.dumps(
        _stabilise(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: Any) -> str:
    """SHA-256 of raw bytes, a string, or any canonical-JSON-able object."""
    if isinstance(data, (bytes, bytearray)):
        payload = bytes(data)
    elif isinstance(data, str):
        payload = data.encode("utf-8")
    else:
        payload = canonical_json(data)
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: str | Path) -> str:
    """Streamed SHA-256 — model weight files are too big to read at once."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()
