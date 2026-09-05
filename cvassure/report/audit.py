"""The audit's own audit trail (PS clause 2.2.5).

The system checks other people's integrity, so it has to be able to answer the
same question about itself: did this report really come from this data, this
model and this configuration? Every detector run appends a record, and the
records are hash-chained exactly like the inference receipts, so a changed
entry breaks every link after it.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cvassure.core.hashing import canonical_json, sha256_hex

GENESIS = "0" * 64


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="microseconds")


def environment() -> dict[str, Any]:
    """Recorded once per audit so a result can be reproduced years later."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


@dataclass
class AuditLog:
    """Append-only, hash-chained record of everything the audit did."""

    path: Path
    entries: list[dict[str, Any]] = field(default_factory=list)
    _prev: str = GENESIS

    @classmethod
    def open(cls, path: str | Path) -> "AuditLog":
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        log = cls(path=p)
        if p.exists():
            log.entries = [
                json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line
            ]
            if log.entries:
                log._prev = entry_hash(log.entries[-1])
        return log

    def append(
        self,
        *,
        detector_id: str,
        config_hash: str,
        input_digests: dict[str, Any],
        seed: int,
        started: str,
        ended: str,
        output_digest: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "seq": len(self.entries) + 1,
            "detector_id": detector_id,
            "config_hash": config_hash,
            "input_digests": input_digests,
            "seed": int(seed),
            "started_utc": started,
            "ended_utc": ended,
            "output_digest": output_digest,
            "prev_hash": self._prev,
            "extra": extra or {},
        }
        self.entries.append(entry)
        self._prev = entry_hash(entry)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(entry).decode("utf-8") + "\n")
        return entry

    def record_run(
        self,
        result,
        *,
        input_digests: dict[str, Any],
        config: dict[str, Any],
        seed: int,
        started: str,
    ) -> dict[str, Any]:
        from cvassure.core.hashing import sha256_hex as _sha

        return self.append(
            detector_id=result.detector_id,
            config_hash=_sha(config),
            input_digests=input_digests,
            seed=seed,
            started=started,
            ended=_now(),
            output_digest=_sha([f.to_dict() for f in result.findings]),
            extra={
                "n_findings": len(result.findings),
                "seconds": round(result.seconds, 4),
                "peak_ram_mb": round(result.peak_ram_mb, 2),
                "error": result.error,
            },
        )


def entry_hash(entry: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(entry))


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


@dataclass
class AuditVerification:
    total: int
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        if self.ok:
            if self.total == 0:
                return "PASS  The audit log is empty — nothing has been run yet."
            return (
                f"PASS  All {self.total} audit records are intact and in order. "
                f"Nothing in this audit's own history has been altered."
            )
        return "FAIL  The audit log has been tampered with.\n      " + "\n      ".join(
            self.failures
        )


def verify_audit_log(path: str | Path) -> AuditVerification:
    p = Path(path)
    if not p.exists():
        return AuditVerification(total=0, failures=[f"There is no audit log at {p}."])

    entries = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
    result = AuditVerification(total=len(entries))
    prev = GENESIS
    for i, entry in enumerate(entries):
        if entry.get("prev_hash") != prev:
            result.failures.append(
                f"Record #{entry.get('seq', i + 1)} does not follow on from the one "
                f"before it — the history has been rewritten."
            )
        if entry.get("seq") != i + 1:
            result.failures.append(
                f"Record #{i + 1} is numbered {entry.get('seq')}; a record has been "
                f"inserted or removed."
            )
        prev = entry_hash(entry)
    return result
