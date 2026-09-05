"""Inference receipts: cryptographically binding input, model and output.

PS clause 2.2.3 asks that an inference output be tied to the exact input, the
exact weights and the exact configuration that produced it, in a way that
cannot be quietly edited afterwards. A receipt is that binding; the chain of
receipts is what makes editing *detectable*.
"""

from __future__ import annotations

import datetime as _dt
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from cvassure.core.hashing import canonical_json, file_digest, sha256_hex

GENESIS = "0" * 64
NONCE_BYTES = 16

#: Fields that are covered by the signature. ``signature`` itself is excluded,
#: obviously; everything else is bound.
SIGNED_FIELDS: tuple[str, ...] = (
    "receipt_id",
    "seq",
    "nonce",
    "input_sha256",
    "input_phash",
    "model_weight_digest",
    "preproc_config_hash",
    "inference_config_hash",
    "output",
    "output_sha256",
    "timestamp_utc",
    "prev_receipt_hash",
)


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------


def init_keys(out_dir: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Generate a local Ed25519 keypair. Nothing is ever transmitted."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    priv_path, pub_path = out / "priv.pem", out / "pub.pem"
    if priv_path.exists() and not overwrite:
        raise FileExistsError(f"{priv_path} already exists; pass --overwrite to replace it")

    private_key = Ed25519PrivateKey.generate()
    priv_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    os.chmod(priv_path, 0o600)
    pub_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return priv_path, pub_path


def load_private_key(path: str | Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("cvassure signs with Ed25519 keys only")
    return key


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("cvassure verifies Ed25519 keys only")
    return key


# --------------------------------------------------------------------------
# The receipt
# --------------------------------------------------------------------------


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="microseconds")


def perceptual_hash(image_path: str | Path) -> str:
    """pHash of the input image, so a visually identical but re-encoded input
    is still recognisable. Falls back to an empty string if the file is not a
    readable image — a receipt for a non-image input is still valid."""
    try:
        import imagehash
        from PIL import Image

        with Image.open(image_path) as im:
            return str(imagehash.phash(im.convert("RGB")))
    except Exception:
        return ""


@dataclass(frozen=True, slots=True)
class Receipt:
    seq: int
    input_sha256: str
    model_weight_digest: str
    preproc_config_hash: str
    inference_config_hash: str
    output: dict[str, Any]
    output_sha256: str
    prev_receipt_hash: str
    input_phash: str = ""
    timestamp_utc: str = field(default_factory=utc_now)
    nonce: str = field(default_factory=lambda: os.urandom(NONCE_BYTES).hex())
    receipt_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    signature: str | None = None

    def signing_payload(self) -> bytes:
        d = asdict(self)
        return canonical_json({k: d[k] for k in SIGNED_FIELDS})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Receipt":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def receipt_hash(receipt: Receipt | dict[str, Any]) -> str:
    """The link value the *next* receipt records. Covers the signature too, so
    an attacker cannot re-sign a record without breaking the following link."""
    d = receipt.to_dict() if isinstance(receipt, Receipt) else dict(receipt)
    return sha256_hex(canonical_json(d))


def sign_receipt(receipt: Receipt, private_key: Ed25519PrivateKey) -> Receipt:
    sig = private_key.sign(receipt.signing_payload())
    return Receipt.from_dict({**receipt.to_dict(), "signature": sig.hex()})


# --------------------------------------------------------------------------
# Building a chain
# --------------------------------------------------------------------------


class ReceiptChain:
    """Append-only, signed, hash-linked log of inferences."""

    def __init__(self, private_key: Ed25519PrivateKey, *, start_seq: int = 1):
        self._key = private_key
        self._next_seq = start_seq
        self._prev_hash = GENESIS
        self.receipts: list[Receipt] = []

    @classmethod
    def from_key_file(cls, path: str | Path) -> "ReceiptChain":
        return cls(load_private_key(path))

    def append(
        self,
        *,
        input_sha256: str,
        model_weight_digest: str,
        preproc_config: dict[str, Any] | str,
        inference_config: dict[str, Any] | str,
        output: dict[str, Any],
        input_phash: str = "",
        timestamp_utc: str | None = None,
    ) -> Receipt:
        r = Receipt(
            seq=self._next_seq,
            input_sha256=input_sha256,
            input_phash=input_phash,
            model_weight_digest=model_weight_digest,
            preproc_config_hash=_hash_config(preproc_config),
            inference_config_hash=_hash_config(inference_config),
            output=output,
            output_sha256=sha256_hex(output),
            prev_receipt_hash=self._prev_hash,
            timestamp_utc=timestamp_utc or utc_now(),
        )
        signed = sign_receipt(r, self._key)
        self.receipts.append(signed)
        self._prev_hash = receipt_hash(signed)
        self._next_seq += 1
        return signed

    def append_for_image(
        self,
        image_path: str | Path,
        *,
        model_weight_digest: str,
        preproc_config: dict[str, Any],
        inference_config: dict[str, Any],
        output: dict[str, Any],
    ) -> Receipt:
        return self.append(
            input_sha256=file_digest(image_path),
            input_phash=perceptual_hash(image_path),
            model_weight_digest=model_weight_digest,
            preproc_config=preproc_config,
            inference_config=inference_config,
            output=output,
        )

    def write(self, path: str | Path) -> Path:
        return write_receipts(self.receipts, path)


def _hash_config(cfg: dict[str, Any] | str) -> str:
    return cfg if isinstance(cfg, str) else sha256_hex(cfg)


# --------------------------------------------------------------------------
# JSONL IO
# --------------------------------------------------------------------------


def write_receipts(receipts: Iterable[Receipt | dict[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in receipts:
            d = r.to_dict() if isinstance(r, Receipt) else r
            fh.write(canonical_json(d).decode("utf-8") + "\n")
    return p


def read_receipts(path: str | Path) -> list[dict[str, Any]]:
    """Read raw dicts, not Receipts. Verification must see exactly what is on
    disk, including fields we do not recognise."""
    import json

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


def iter_receipts(path: str | Path) -> Iterator[dict[str, Any]]:
    yield from read_receipts(path)


def sign_unsigned_file(
    in_path: str | Path, out_path: str | Path, private_key: Ed25519PrivateKey
) -> list[Receipt]:
    """Take a file of raw (unsigned, unchained) inference records and turn it
    into a proper chain. Used by ``cvassure provenance sign``."""
    chain = ReceiptChain(private_key)
    for rec in read_receipts(in_path):
        chain.append(
            input_sha256=rec.get("input_sha256", ""),
            input_phash=rec.get("input_phash", ""),
            model_weight_digest=rec.get("model_weight_digest", ""),
            preproc_config=rec.get("preproc_config_hash") or rec.get("preproc_config", {}),
            inference_config=rec.get("inference_config_hash") or rec.get("inference_config", {}),
            output=rec.get("output", {}),
            timestamp_utc=rec.get("timestamp_utc"),
        )
    write_receipts(chain.receipts, out_path)
    return chain.receipts
