"""Chain of custody: a signed record of every organisation that touched a
dataset or model before it reached you.

The problem statement's provenance clause (2.2.3) is about *inference*
records: binding one answer to the exact input and model that produced it.
This module answers a question one step earlier in the same pipeline, using
the same two cryptographic tools (Ed25519 signatures, a hash-linked log):
when a dataset is collected by one organisation, labelled by a second, and
trained into a model by a third, can the organisation that finally deploys it
prove nothing was substituted at any handoff between them?

A custody chain differs from an inference-receipt chain (``receipts.py``) in
the one place that matters: every hop can be signed by a *different*
organisation's own key, and each hop must declare a content digest of what it
received — which has to equal the previous hop's declared output digest. That
equality check, not the signature alone, is the actual chain-of-custody link:
a valid signature only proves an organisation signed *something*; matching
digests prove it signed off on the *same bytes* the previous organisation
produced, not a substituted copy.

Nothing here retrains or inspects model behaviour — it only hashes and signs,
exactly like the inference-receipt chain it borrows its primitives from.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from cvassure.core.hashing import canonical_json, file_digest, sha256_hex
from cvassure.core.schemas import Finding, Sample
from cvassure.provenance.receipts import (
    GENESIS,
    NONCE_BYTES,
    init_keys,
    load_private_key,
    load_public_key,
)

#: Suggested stage names — not enforced. A custom stage string is accepted so
#: this is never a bottleneck for a pipeline shaped differently from the
#: usual collect -> label -> train -> deploy sequence.
SUGGESTED_STAGES: tuple[str, ...] = ("data_collection", "labelling", "training", "deployment")

SIGNED_FIELDS: tuple[str, ...] = (
    "record_id",
    "hop",
    "stage",
    "actor_id",
    "input_digest",
    "output_digest",
    "description",
    "timestamp_utc",
    "nonce",
    "prev_hash",
)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="microseconds")


def _clip(text: str, limit: int = 380) -> str:
    """Cut to a length a ``Finding.reason`` is guaranteed to accept, rather
    than trusting every string folded into it to already be short."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Content digests — what actually gets signed off on at each hop
# --------------------------------------------------------------------------


def dataset_content_digest(samples: Sequence[Sample], *, include_labels: bool = True) -> str:
    """A digest that changes if any image's bytes change, any image is added
    or removed, or — when asked — any label changes. It does not change if a
    file is only moved on disk, because the digest is keyed on content, not
    path.

    ``include_labels=False`` gives the digest a data-collection stage should
    sign: the images alone, before anyone has labelled them. The labelling
    stage then signs the ``include_labels=True`` digest of the same images,
    so a verifier can see that the pictures a labeller worked from are
    byte-identical to the ones the collector actually handed over — only the
    labels are new.
    """
    rows = []
    for s in sorted(samples, key=lambda s: s.sample_id):
        row: tuple[Any, ...] = (s.sample_id, file_digest(s.image_path))
        if include_labels:
            row = row + (s.label or "",)
        rows.append(row)
    return sha256_hex(rows)


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CustodyRecord:
    """One organisation's signed statement: 'I am ``actor_id``, at stage
    ``stage``; what I received digests to ``input_digest``; what I produced
    and am handing on digests to ``output_digest``.'"""

    hop: int
    stage: str
    actor_id: str
    input_digest: str
    output_digest: str
    description: str
    prev_hash: str
    timestamp_utc: str = field(default_factory=utc_now)
    nonce: str = field(default_factory=lambda: os.urandom(NONCE_BYTES).hex())
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    signature: str | None = None

    def signing_payload(self) -> bytes:
        d = asdict(self)
        return canonical_json({k: d[k] for k in SIGNED_FIELDS})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CustodyRecord":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def record_hash(record: CustodyRecord | dict[str, Any]) -> str:
    """The link value the *next* hop records. Covers the signature too, so a
    hop cannot be re-signed without breaking the link that follows it."""
    d = record.to_dict() if isinstance(record, CustodyRecord) else dict(record)
    return sha256_hex(canonical_json(d))


def sign_record(record: CustodyRecord, private_key: Ed25519PrivateKey) -> CustodyRecord:
    sig = private_key.sign(record.signing_payload())
    return CustodyRecord.from_dict({**record.to_dict(), "signature": sig.hex()})


# --------------------------------------------------------------------------
# Building a chain — a different key is allowed at every hop
# --------------------------------------------------------------------------


class CustodyChain:
    """Append-only, hash-linked log where each hop can be signed by a
    different organisation's own private key.

    This is the one structural difference from :class:`ReceiptChain`: that
    class is handed one key at construction because one party signs every
    record. Here, the key is supplied per hop, because the whole point is
    that different organisations — who do not trust each other and do not
    share a key — each sign only the hop they were responsible for.
    """

    def __init__(self, *, start_hop: int = 1):
        self._next_hop = start_hop
        self._prev_hash = GENESIS
        self.records: list[CustodyRecord] = []

    @classmethod
    def from_file(cls, path: str | Path) -> "CustodyChain":
        chain = cls()
        for raw in read_custody_chain(path):
            chain.records.append(CustodyRecord.from_dict(raw))
        if chain.records:
            chain._next_hop = chain.records[-1].hop + 1
            chain._prev_hash = record_hash(chain.records[-1])
        return chain

    @property
    def last_output_digest(self) -> str | None:
        return self.records[-1].output_digest if self.records else None

    def add_hop(
        self,
        *,
        stage: str,
        actor_id: str,
        output_digest: str,
        description: str,
        private_key: Ed25519PrivateKey,
        input_digest: str | None = None,
        timestamp_utc: str | None = None,
    ) -> CustodyRecord:
        """Add and sign one hop.

        ``input_digest`` defaults to the previous hop's output digest — the
        normal case, where this organisation is truthfully declaring it
        received exactly what the last organisation produced. It can be
        overridden explicitly, but doing so honestly only makes sense for the
        very first hop (which has nothing before it, so its input is
        ``GENESIS``) or for deliberately constructing a test case.
        """
        resolved_input = (
            input_digest if input_digest is not None else (self.last_output_digest or GENESIS)
        )
        r = CustodyRecord(
            hop=self._next_hop,
            stage=stage,
            actor_id=actor_id,
            input_digest=resolved_input,
            output_digest=output_digest,
            description=description,
            prev_hash=self._prev_hash,
            timestamp_utc=timestamp_utc or utc_now(),
        )
        signed = sign_record(r, private_key)
        self.records.append(signed)
        self._prev_hash = record_hash(signed)
        self._next_hop += 1
        return signed

    def write(self, path: str | Path) -> Path:
        return write_custody_chain(self.records, path)


# --------------------------------------------------------------------------
# JSONL IO
# --------------------------------------------------------------------------


def write_custody_chain(records: Sequence[CustodyRecord | dict[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            d = r.to_dict() if isinstance(r, CustodyRecord) else r
            fh.write(canonical_json(d).decode("utf-8") + "\n")
    return p


def read_custody_chain(path: str | Path) -> list[dict[str, Any]]:
    """Read raw dicts, not records. Verification must see exactly what is on
    disk, including fields it does not recognise."""
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    return out


# --------------------------------------------------------------------------
# Keyrings — every organisation's public key, gathered in one place so a
# chain touching several of them can be verified in a single pass
# --------------------------------------------------------------------------


def init_actor_keys(out_dir: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Generate one organisation's Ed25519 keypair. A thin, named wrapper
    around the same key generation the inference-receipt chain uses — a
    custody actor and a receipt signer are the same kind of key, used at a
    different point in the pipeline."""
    return init_keys(out_dir, overwrite=overwrite)


def load_keyring_dir(root: str | Path) -> dict[str, Ed25519PublicKey]:
    """Build a keyring from ``<root>/<actor_id>/pub.pem`` for every actor
    directory found — the layout ``init_actor_keys`` writes."""
    root = Path(root)
    keyring: dict[str, Ed25519PublicKey] = {}
    if not root.exists():
        return keyring
    for sub in sorted(root.iterdir()):
        pub = sub / "pub.pem"
        if sub.is_dir() and pub.exists():
            keyring[sub.name] = load_public_key(pub)
    return keyring


def load_keyring_json(path: str | Path) -> dict[str, Ed25519PublicKey]:
    """Build a keyring from ``{"actor_id": "-----BEGIN PUBLIC KEY-----...", ...}``
    — one portable file a verifier can be handed instead of a directory of
    individual ``pub.pem`` files."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        actor: serialization.load_pem_public_key(pem.encode("utf-8"))
        for actor, pem in raw.items()
    }


def keyring_to_json(pubkey_paths: dict[str, str | Path]) -> dict[str, str]:
    """The inverse of :func:`load_keyring_json`: gather several actors'
    ``pub.pem`` files into the one-file mapping it reads."""
    return {actor: Path(p).read_text(encoding="utf-8") for actor, p in pubkey_paths.items()}


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

SIGNATURE_INVALID = "SIGNATURE_INVALID"
UNKNOWN_ACTOR = "UNKNOWN_ACTOR"
CHAIN_BROKEN = "CHAIN_BROKEN"
HOP_GAP = "HOP_GAP"
HOP_REPLAY = "HOP_REPLAY"
HANDOFF_MISMATCH = "HANDOFF_MISMATCH"
TIMESTAMP_REGRESSION = "TIMESTAMP_REGRESSION"
MALFORMED = "MALFORMED"

#: When several failures fire on one hop, this is the order that decides
#: which one is the headline — the most specific explanation of what
#: actually went wrong, printed first.
FAILURE_PRIORITY: tuple[str, ...] = (
    MALFORMED,
    UNKNOWN_ACTOR,
    HOP_REPLAY,
    HOP_GAP,
    SIGNATURE_INVALID,
    HANDOFF_MISMATCH,
    TIMESTAMP_REGRESSION,
    CHAIN_BROKEN,
)

#: Kept to one short clause each, deliberately — these get folded into a
#: Finding's ``reason``, which the schema enforces must read as one sentence
#: (see ``check_plain_english``), not a paragraph. The fuller explanation
#: lives in ``.render()``'s surrounding text and in ``evidence``, where there
#: is no length budget to respect.
EXPLANATIONS: dict[str, str] = {
    SIGNATURE_INVALID: "This hop's record was changed after the organisation signed it.",
    UNKNOWN_ACTOR: "This hop is signed by an organisation we have no public key for.",
    CHAIN_BROKEN: "This record does not point back at the one before it.",
    HOP_GAP: "A hop is missing — the numbering jumps.",
    HOP_REPLAY: "A hop number was reused, or the hops are out of order.",
    HANDOFF_MISMATCH: "What this organisation received does not match what was actually "
    "produced before it.",
    TIMESTAMP_REGRESSION: "This hop claims to have happened before the one in front of it.",
    MALFORMED: "This hop is missing fields it must have.",
}


@dataclass(frozen=True, slots=True)
class CustodyFailure:
    hop: int | None
    index: int
    code: str
    detail: str
    actor_id: str | None = None

    @property
    def explanation(self) -> str:
        return EXPLANATIONS.get(self.code, "Unknown failure.")


@dataclass
class CustodyVerification:
    total: int
    failures: list[CustodyFailure] = field(default_factory=list)
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
                    "hop": f.hop,
                    "index": f.index,
                    "code": f.code,
                    "detail": f.detail,
                    "explanation": f.explanation,
                    "actor_id": f.actor_id,
                }
                for f in self.failures
            ],
        }

    def render(self) -> str:
        if self.ok:
            if self.total == 0:
                return "PASS  No custody hops to check."
            return f"PASS  Hops 1-{self.total} verified. Nothing was substituted at any handoff."

        lines = []
        if self.n_verified:
            lines.append(f"PASS  Hops 1-{self.n_verified} verified.")
        by_hop: dict[Any, list[CustodyFailure]] = {}
        for f in self.failures:
            by_hop.setdefault(f.hop if f.hop is not None else f"line {f.index + 1}", []).append(f)
        for hop, fails in by_hop.items():
            fails = sorted(fails, key=lambda f: FAILURE_PRIORITY.index(f.code))
            head = fails[0]
            label = f"#{hop}" if isinstance(hop, int) else hop
            lines.append(f"FAIL  Hop {label}: {head.code}")
            lines.append(f"      {head.explanation}")
            for f in fails:
                lines.append(f"      {f.detail}")
        lines.append("")
        lines.append(f"SUMMARY  {len(by_hop)} of {self.total} hops failed verification.")
        return "\n".join(lines)


def verify_custody_chain(
    records: Sequence[dict[str, Any]] | str | Path,
    keyring: dict[str, Ed25519PublicKey],
) -> CustodyVerification:
    """Check every hop's signature against its own actor's key, check that
    each hop's declared input equals the previous hop's declared output, and
    check the log itself has not been reordered, replayed or rewritten."""
    if isinstance(records, (str, Path)):
        records = read_custody_chain(records)

    result = CustodyVerification(total=len(records))
    seen_hops: dict[int, int] = {}
    prev_hash = GENESIS
    prev_hop: int | None = None
    prev_ts: str | None = None
    prev_output_digest = GENESIS

    def fail(idx: int, hop: Any, code: str, detail: str, actor: str | None = None) -> None:
        result.failures.append(
            CustodyFailure(
                hop=hop if isinstance(hop, int) else None, index=idx, code=code,
                detail=detail, actor_id=actor,
            )
        )
        if result.first_bad_index is None:
            result.first_bad_index = idx

    for idx, raw in enumerate(records):
        missing = [f for f in (*SIGNED_FIELDS, "signature") if f not in raw]
        if missing:
            fail(idx, raw.get("hop"), MALFORMED, f"Missing fields: {', '.join(missing)}.")
            continue

        hop = raw["hop"]
        actor_id = raw["actor_id"]

        # -- signature, against this hop's own actor's key ----------------
        pubkey = keyring.get(actor_id)
        if pubkey is None:
            fail(
                idx, hop, UNKNOWN_ACTOR,
                f"No public key on file for '{actor_id}'.", actor_id,
            )
        else:
            payload = canonical_json({k: raw[k] for k in SIGNED_FIELDS})
            try:
                pubkey.verify(bytes.fromhex(raw["signature"]), payload)
            except (InvalidSignature, ValueError):
                fail(
                    idx, hop, SIGNATURE_INVALID,
                    f"'{actor_id}' signed this hop, but the record no longer matches what "
                    f"was signed.",
                    actor_id,
                )

        # -- the log's own hash chain --------------------------------------
        if raw["prev_hash"] != prev_hash:
            fail(
                idx, hop, CHAIN_BROKEN,
                f"Expected link {prev_hash[:12]}…, found {raw['prev_hash'][:12]}….", actor_id,
            )

        # -- the actual custody link: input must equal the prior output ---
        if raw["input_digest"] != prev_output_digest:
            if idx == 0:
                detail = (
                    "The first hop should declare it received nothing before it "
                    f"(the origin marker), but it declares receiving {raw['input_digest'][:12]}…."
                )
            else:
                detail = (
                    f"'{actor_id}' says it received something that digests to "
                    f"{raw['input_digest'][:12]}…, but the organisation before it signed off "
                    f"on producing {prev_output_digest[:12]}… — those do not match."
                )
            fail(idx, hop, HANDOFF_MISMATCH, detail, actor_id)

        # -- hop numbering -------------------------------------------------
        if hop in seen_hops:
            fail(
                idx, hop, HOP_REPLAY,
                f"Hop {hop} was already used at line {seen_hops[hop] + 1}.", actor_id,
            )
        elif prev_hop is not None and hop < prev_hop:
            fail(
                idx, hop, HOP_REPLAY,
                f"Hop {hop} appears after hop {prev_hop}; the log has been reordered.", actor_id,
            )
        elif prev_hop is not None and hop > prev_hop + 1:
            n_missing = hop - prev_hop - 1
            missing_list = ", ".join(str(h) for h in range(prev_hop + 1, hop))
            fail(
                idx, hop, HOP_GAP,
                f"{n_missing} hop(s) missing between {prev_hop} and {hop}: {missing_list}.",
                actor_id,
            )

        # -- timestamps ------------------------------------------------------
        ts = raw["timestamp_utc"]
        if prev_ts is not None and isinstance(ts, str) and ts < prev_ts:
            fail(idx, hop, TIMESTAMP_REGRESSION, f"Recorded at {ts}, after {prev_ts}.", actor_id)

        seen_hops[hop] = idx
        prev_hash = record_hash(raw)
        prev_hop = hop if prev_hop is None else max(prev_hop, hop)
        prev_ts = ts if prev_ts is None else max(prev_ts, ts)
        prev_output_digest = raw["output_digest"]

    return result


# --------------------------------------------------------------------------
# Bridge into the rest of the pipeline
# --------------------------------------------------------------------------

_CODE_TO_ATTACK = {
    SIGNATURE_INVALID: "custody_break",
    UNKNOWN_ACTOR: "custody_unknown_actor",
    CHAIN_BROKEN: "custody_break",
    HOP_GAP: "custody_break",
    HOP_REPLAY: "custody_break",
    HANDOFF_MISMATCH: "custody_substitution",
    TIMESTAMP_REGRESSION: "custody_break",
    MALFORMED: "custody_break",
}


def findings_from_custody_result(result: CustodyVerification, access_tier: int = 0) -> list[Finding]:
    """Turn a verification result into Findings, so a custody chain appears in
    the same report and the same JSON shape as every other check."""
    out: list[Finding] = []
    if result.ok:
        out.append(
            Finding(
                asset_ref="custody_chain",
                asset_type="custody",
                attack_class="clean",
                detector_id="provenance.custody",
                access_tier=access_tier,
                raw_score=0.0,
                severity="low",
                disposition="accept",
                reason=(
                    f"All {result.total} custody hops are signed by the organisation that "
                    f"claims them, and each one's declared input matches exactly what the "
                    f"organisation before it signed off on producing — nothing was "
                    f"substituted at any handoff."
                ),
                evidence={"total": result.total, "failures": 0},
            )
        )
        return out

    grouped: dict[Any, list[CustodyFailure]] = {}
    for f in result.failures:
        grouped.setdefault(f.hop if f.hop is not None else f"line{f.index}", []).append(f)

    for key, fails in grouped.items():
        fails = sorted(fails, key=lambda f: FAILURE_PRIORITY.index(f.code))
        head = fails[0]
        label = f"#{key}" if isinstance(key, int) else str(key)
        out.append(
            Finding(
                asset_ref=f"custody_hop:{key}",
                asset_type="custody",
                attack_class=_CODE_TO_ATTACK[head.code],
                detector_id="provenance.custody",
                access_tier=access_tier,
                raw_score=1.0,
                severity="critical",
                disposition="quarantine",
                # A hard length cap, not just short EXPLANATIONS text, because
                # ``head.detail`` embeds ``actor_id`` — free text typed into the
                # web form with no length limit of its own — so nothing here
                # may assume the resulting sentence stays short on trust alone.
                reason=_clip(
                    f"Custody hop {label} does not check out: {head.explanation} {head.detail}"
                ),
                evidence={
                    "codes": [f.code for f in fails],
                    "details": [f.detail for f in fails],
                    "hop": head.hop,
                    "actor_id": head.actor_id,
                },
            )
        )
    return out


def verify_custody_file(path: str | Path, keyring: dict[str, Ed25519PublicKey]) -> CustodyVerification:
    return verify_custody_chain(read_custody_chain(path), keyring)


__all__ = [
    "CHAIN_BROKEN",
    "HANDOFF_MISMATCH",
    "HOP_GAP",
    "HOP_REPLAY",
    "MALFORMED",
    "SIGNATURE_INVALID",
    "SUGGESTED_STAGES",
    "TIMESTAMP_REGRESSION",
    "UNKNOWN_ACTOR",
    "CustodyChain",
    "CustodyFailure",
    "CustodyRecord",
    "CustodyVerification",
    "dataset_content_digest",
    "findings_from_custody_result",
    "init_actor_keys",
    "keyring_to_json",
    "load_keyring_dir",
    "load_keyring_json",
    "read_custody_chain",
    "record_hash",
    "sign_record",
    "verify_custody_chain",
    "verify_custody_file",
    "write_custody_chain",
]
