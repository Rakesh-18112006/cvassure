"""The model integrity suite (PS clause 2.2.2).

No retraining happens anywhere in this module — PS clause 2.2.6 forbids it for
baseline assessment. Every check here reads the model; none of them writes to
it.
"""

from __future__ import annotations

from cvassure.core.schemas import Finding
from cvassure.detect.base import AuditContext, DetectorResult
from cvassure.detect.fingerprint import FingerprintDetector
from cvassure.detect.trigger_recon import TriggerReconDetector
from cvassure.detect.weight_digest import WeightDigestDetector
from cvassure.detect.weight_stats import WeightStatsDetector


def build_detectors() -> list:
    return [
        FingerprintDetector(),   # tier 0 — works black-box
        WeightDigestDetector(),  # tier 1
        WeightStatsDetector(),   # tier 1
        TriggerReconDetector(),  # tier 2
    ]


def run_all(ctx: AuditContext) -> tuple[list[Finding], list[DetectorResult]]:
    findings: list[Finding] = []
    results: list[DetectorResult] = []
    for detector in build_detectors():
        result = detector.safe_run(ctx)
        results.append(result)
        findings.extend(result.findings)
    return findings, results


def enrol(model, samples=None, *, n_probes: int = 200, seed: int = 20260101,
          input_shape: tuple[int, ...] = (3, 64, 64)) -> dict:
    """Record everything a future audit will need to prove this model is the
    one that was accepted.

    Run this once, when a model is first taken into service. It is the model
    equivalent of signing an inference receipt.
    """
    from cvassure.detect.fingerprint import compute_fingerprint
    from cvassure.detect.weight_digest import canonical_weight_digest
    from cvassure.ingest.models import AccessDenied

    record = compute_fingerprint(
        model, n_probes=n_probes, seed=seed, input_shape=input_shape
    )
    record["file_digest"] = model.file_digest()
    try:
        record["weight_digest"] = canonical_weight_digest(model.weights())
        record["layer_stats"] = [s.to_dict() for s in model.layer_stats()]
    except AccessDenied:
        # Enrolling from black-box access is still worth doing: the
        # fingerprint alone catches substitution.
        record["weight_digest"] = None
        record["layer_stats"] = None
        record["note"] = (
            "enrolled without access to the weights, so later audits can prove the "
            "model was swapped but not which layers were edited"
        )
    return record
