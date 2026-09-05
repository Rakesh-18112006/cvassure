"""Verifying a receipt log, and saying in plain words what went wrong.

Six failure modes are named explicitly. Cryptography either matches or it does
not, so every row of the tamper-detection results table should read 100%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from cvassure.core.hashing import canonical_json
from cvassure.core.schemas import Finding
from cvassure.provenance.receipts import (
    GENESIS,
    SIGNED_FIELDS,
    load_public_key,
    read_receipts,
    receipt_hash,
)

SIGNATURE_INVALID = "SIGNATURE_INVALID"
CHAIN_BROKEN = "CHAIN_BROKEN"
SEQUENCE_GAP = "SEQUENCE_GAP"
SEQUENCE_REPLAY = "SEQUENCE_REPLAY"
NONCE_REUSE = "NONCE_REUSE"
TIMESTAMP_REGRESSION = "TIMESTAMP_REGRESSION"
MALFORMED = "MALFORMED"

#: When several failures fire on one record we still report them all, but the
#: first one in this order is the headline — it is the most specific
#: explanation of what an attacker actually did.
FAILURE_PRIORITY: tuple[str, ...] = (
    MALFORMED,
    SEQUENCE_REPLAY,
    NONCE_REUSE,
    SEQUENCE_GAP,
    SIGNATURE_INVALID,
    TIMESTAMP_REGRESSION,
    CHAIN_BROKEN,
)

#: One sentence per failure mode, written for somebody who has never heard the
#: word "nonce".
EXPLANATIONS: dict[str, str] = {
    SIGNATURE_INVALID: "The recorded answer was changed after it was signed.",
    CHAIN_BROKEN: "This record does not point back at the one before it, so the "
    "history has been rewritten.",
    SEQUENCE_GAP: "A record is missing — the numbering jumps.",
    SEQUENCE_REPLAY: "An old record was submitted again under a number that had "
    "already been used.",
    NONCE_REUSE: "This record reuses a one-time marker from an earlier record, "
    "which means it is a copy, not a new inference.",
    TIMESTAMP_REGRESSION: "This record claims to have happened before the record "
    "in front of it. Time does not run backwards.",
    MALFORMED: "This record is missing fields it must have, so it cannot be checked.",
}


@dataclass(frozen=True, slots=True)
class Failure:
    seq: int | None
    index: int
    code: str
    detail: str
    receipt_id: str | None = None

    @property
    def explanation(self) -> str:
        return EXPLANATIONS.get(self.code, "Unknown failure.")


@dataclass
class VerificationResult:
    total: int
    failures: list[Failure] = field(default_factory=list)
    first_bad_index: int | None = None

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def codes(self) -> set[str]:
        return {f.code for f in self.failures}

    @property
    def n_verified(self) -> int:
        return self.total if self.ok else (self.first_bad_index or 0)

    def has(self, code: str) -> bool:
        return code in self.codes

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "ok": self.ok,
            "n_verified": self.n_verified,
            "failures": [
                {
                    "seq": f.seq,
                    "index": f.index,
                    "code": f.code,
                    "detail": f.detail,
                    "explanation": f.explanation,
                }
                for f in self.failures
            ],
        }

    # -- human output ----------------------------------------------------

    def render(self) -> str:
        """Output you can read from the back of a room."""
        if self.ok:
            if self.total == 0:
                return "PASS  No receipts to check."
            return f"PASS  Receipts 1-{self.total} verified."

        lines = []
        if self.n_verified:
            lines.append(f"PASS  Receipts 1-{self.n_verified} verified.")
        by_seq: dict[Any, list[Failure]] = {}
        for f in self.failures:
            by_seq.setdefault(f.seq if f.seq is not None else f"line {f.index + 1}", []).append(f)
        for seq, fails in by_seq.items():
            fails = sorted(fails, key=lambda f: FAILURE_PRIORITY.index(f.code))
            head = fails[0]
            label = f"#{seq}" if isinstance(seq, int) else seq
            lines.append(f"FAIL  Receipt {label}: {head.code}")
            lines.append(f"      {head.explanation}")
            for f in fails:
                lines.append(f"      {f.detail}")
        lines.append("")
        lines.append(
            f"SUMMARY  {len(by_seq)} of {self.total} records failed verification."
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify_log(
    receipts: Sequence[dict[str, Any]] | str | Path,
    public_key: Ed25519PublicKey | str | Path,
) -> VerificationResult:
    """Check signatures, chain links, ordering, replays and timestamps."""
    if isinstance(receipts, (str, Path)):
        receipts = read_receipts(receipts)
    if isinstance(public_key, (str, Path)):
        public_key = load_public_key(public_key)

    result = VerificationResult(total=len(receipts))
    seen_nonces: dict[str, int] = {}
    seen_seqs: dict[int, int] = {}
    prev_hash = GENESIS
    prev_seq: int | None = None
    prev_ts: str | None = None

    def fail(idx: int, seq: Any, code: str, detail: str, rid: Any = None) -> None:
        result.failures.append(
            Failure(seq=seq if isinstance(seq, int) else None, index=idx, code=code,
                    detail=detail, receipt_id=rid)
        )
        if result.first_bad_index is None:
            result.first_bad_index = idx

    for idx, raw in enumerate(receipts):
        missing = [f for f in (*SIGNED_FIELDS, "signature") if f not in raw]
        if missing:
            fail(idx, raw.get("seq"), MALFORMED, f"Missing fields: {', '.join(missing)}.")
            continue

        seq = raw["seq"]
        rid = raw.get("receipt_id")

        # -- signature ---------------------------------------------------
        payload = canonical_json({k: raw[k] for k in SIGNED_FIELDS})
        try:
            public_key.verify(bytes.fromhex(raw["signature"]), payload)
        except (InvalidSignature, ValueError):
            stored = raw.get("output")
            fail(
                idx,
                seq,
                SIGNATURE_INVALID,
                f'Stored answer: {_short(stored)}   The signature no longer matches '
                f"what this record says.",
                rid,
            )

        # -- chain link --------------------------------------------------
        if raw["prev_receipt_hash"] != prev_hash:
            fail(
                idx,
                seq,
                CHAIN_BROKEN,
                f"Expected link {prev_hash[:12]}…, found {raw['prev_receipt_hash'][:12]}….",
                rid,
            )

        # -- ordering ----------------------------------------------------
        if seq in seen_seqs:
            fail(
                idx,
                seq,
                SEQUENCE_REPLAY,
                f"Number {seq} was already used at line {seen_seqs[seq] + 1}.",
                rid,
            )
        elif prev_seq is not None and seq < prev_seq:
            fail(
                idx,
                seq,
                SEQUENCE_REPLAY,
                f"Number {seq} appears after number {prev_seq}; the log has been reordered.",
                rid,
            )
        elif prev_seq is not None and seq > prev_seq + 1:
            n_missing = seq - prev_seq - 1
            missing_list = ", ".join(str(s) for s in range(prev_seq + 1, seq))
            fail(
                idx,
                seq,
                SEQUENCE_GAP,
                f"{n_missing} record(s) missing between {prev_seq} and {seq}: {missing_list}.",
                rid,
            )

        # -- nonce -------------------------------------------------------
        nonce = raw["nonce"]
        if nonce in seen_nonces:
            fail(
                idx,
                seq,
                NONCE_REUSE,
                f"One-time marker {nonce[:12]}… was already used at line "
                f"{seen_nonces[nonce] + 1}.",
                rid,
            )
        else:
            seen_nonces[nonce] = idx

        # -- timestamp ---------------------------------------------------
        ts = raw["timestamp_utc"]
        if prev_ts is not None and isinstance(ts, str) and ts < prev_ts:
            fail(idx, seq, TIMESTAMP_REGRESSION, f"Recorded at {ts}, after {prev_ts}.", rid)

        seen_seqs[seq] = idx
        prev_hash = receipt_hash(raw)
        prev_seq = seq if prev_seq is None else max(prev_seq, seq)
        prev_ts = ts if prev_ts is None else max(prev_ts, ts)

    return result


def _short(value: Any, width: int = 60) -> str:
    text = str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


# --------------------------------------------------------------------------
# Bridge into the rest of the pipeline
# --------------------------------------------------------------------------

_CODE_TO_ATTACK = {
    SIGNATURE_INVALID: "receipt_alter",
    NONCE_REUSE: "receipt_replay",
    SEQUENCE_REPLAY: "receipt_replay",
    SEQUENCE_GAP: "receipt_delete",
    TIMESTAMP_REGRESSION: "receipt_reorder",
    CHAIN_BROKEN: "receipt_alter",
    MALFORMED: "receipt_alter",
}


def findings_from_result(result: VerificationResult, access_tier: int = 0) -> list[Finding]:
    """Turn a verification result into Findings so the receipt log appears in
    the same report and the same scoring tables as everything else."""
    out: list[Finding] = []
    if result.ok:
        out.append(
            Finding(
                asset_ref="inference_log",
                asset_type="receipt",
                attack_class="clean",
                detector_id="provenance.verify",
                access_tier=access_tier,
                raw_score=0.0,
                severity="low",
                disposition="accept",
                reason=(
                    f"All {result.total} inference records are signed, in order, and each "
                    f"one still points correctly at the one before it — nothing has been "
                    f"changed since they were written."
                ),
                evidence={"total": result.total, "failures": 0},
            )
        )
        return out

    grouped: dict[Any, list[Failure]] = {}
    for f in result.failures:
        grouped.setdefault(f.seq if f.seq is not None else f"line{f.index}", []).append(f)

    for key, fails in grouped.items():
        fails = sorted(fails, key=lambda f: FAILURE_PRIORITY.index(f.code))
        head = fails[0]
        label = f"#{key}" if isinstance(key, int) else str(key)
        out.append(
            Finding(
                asset_ref=f"receipt:{key}",
                asset_type="receipt",
                attack_class=_CODE_TO_ATTACK[head.code],
                detector_id="provenance.verify",
                access_tier=access_tier,
                raw_score=1.0,
                severity="critical",
                disposition="quarantine",
                reason=(
                    f"Inference record {label} does not check out: {head.explanation} "
                    f"{head.detail} This is proof, not an estimate — the maths either "
                    f"matches or it does not, and here it does not."
                ),
                evidence={
                    "codes": [f.code for f in fails],
                    "details": [f.detail for f in fails],
                    "seq": head.seq,
                },
            )
        )
    return out


def verify_file(receipts_path: str | Path, pubkey_path: str | Path) -> VerificationResult:
    return verify_log(read_receipts(receipts_path), load_public_key(pubkey_path))


__all__ = [
    "CHAIN_BROKEN",
    "Failure",
    "NONCE_REUSE",
    "SEQUENCE_GAP",
    "SEQUENCE_REPLAY",
    "SIGNATURE_INVALID",
    "TIMESTAMP_REGRESSION",
    "VerificationResult",
    "findings_from_result",
    "verify_file",
    "verify_log",
]
