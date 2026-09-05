"""Tampering with a signed inference log.

Each function returns the record it attacked, so the tamper-detection table
can be scored per attack type rather than in aggregate.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from cvassure.provenance.receipts import read_receipts, write_receipts


def _rng(seed: int, tag: str) -> np.random.Generator:
    return np.random.default_rng(abs(hash((seed, tag))) % (2**32))


def alter_field(
    receipts_path: str | Path,
    out_path: str | Path,
    *,
    index: int | None = None,
    field: str = "output",
    seed: int = 0,
) -> dict[str, Any]:
    """Change the recorded answer after the fact — the classic 'the model said
    what I needed it to say' attack."""
    log = read_receipts(receipts_path)
    idx = index if index is not None else int(_rng(seed, "alter").integers(0, len(log)))
    target = log[idx]
    before = copy.deepcopy(target.get(field))
    if field == "output" and isinstance(target.get("output"), dict):
        out = dict(target["output"])
        if "label" in out:
            out["label"] = f"not_{out['label']}"
        elif "confidence" in out:
            out["confidence"] = float(out["confidence"]) * 0.5
        else:
            out["tampered"] = True
        target["output"] = out
    else:
        target[field] = f"tampered_{target.get(field)}"
    write_receipts(log, out_path)
    return {
        "attack_class": "receipt_alter",
        "expected_code": "SIGNATURE_INVALID",
        "path": str(out_path),
        "index": idx,
        "seq": target.get("seq"),
        "field": field,
        "before": before,
        "after": target.get(field),
    }


def replay_receipt(
    receipts_path: str | Path, out_path: str | Path, *, index: int | None = None, seed: int = 0
) -> dict[str, Any]:
    """Re-submit an old record so an inference appears to have happened twice."""
    log = read_receipts(receipts_path)
    idx = index if index is not None else int(_rng(seed, "replay").integers(0, len(log)))
    log.append(copy.deepcopy(log[idx]))
    write_receipts(log, out_path)
    return {
        "attack_class": "receipt_replay",
        "expected_code": "SEQUENCE_REPLAY",
        "path": str(out_path),
        "index": idx,
        "seq": log[idx].get("seq"),
    }


def delete_receipt(
    receipts_path: str | Path, out_path: str | Path, *, index: int | None = None, seed: int = 0
) -> dict[str, Any]:
    """Remove an inconvenient record entirely."""
    log = read_receipts(receipts_path)
    idx = index if index is not None else int(_rng(seed, "delete").integers(1, len(log) - 1))
    removed = log.pop(idx)
    write_receipts(log, out_path)
    return {
        "attack_class": "receipt_delete",
        "expected_code": "SEQUENCE_GAP",
        "path": str(out_path),
        "index": idx,
        "seq": removed.get("seq"),
    }


def reorder(
    receipts_path: str | Path,
    out_path: str | Path,
    *,
    index: int | None = None,
    distance: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Move a record so the history reads differently."""
    log = read_receipts(receipts_path)
    idx = index if index is not None else int(
        _rng(seed, "reorder").integers(distance, len(log) - 1)
    )
    other = max(0, idx - distance)
    log[idx], log[other] = log[other], log[idx]
    write_receipts(log, out_path)
    return {
        "attack_class": "receipt_reorder",
        "expected_code": "SEQUENCE_REPLAY",
        "path": str(out_path),
        "index": idx,
        "seq": log[other].get("seq"),
    }


ATTACKS = {
    "alter_field": alter_field,
    "replay_receipt": replay_receipt,
    "delete_receipt": delete_receipt,
    "reorder": reorder,
}


def apply(
    spec: dict[str, Any],
    *,
    receipts_path: str | Path | None,
    out_dir: str | Path,
    seed: int,
    key_path: str | Path | None = None,
) -> dict[str, Any]:
    kind = spec.get("type")
    if kind not in ATTACKS:
        raise ValueError(f"unknown receipt attack {kind!r}; have {sorted(ATTACKS)}")
    if receipts_path is None:
        raise ValueError(f"receipt attack {kind!r} needs --receipts")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{kind}.jsonl"
    params = {k: v for k, v in spec.items() if k != "type"}
    return ATTACKS[kind](receipts_path, dest, seed=seed, **params)
