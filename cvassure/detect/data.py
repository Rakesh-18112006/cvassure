"""The training-data integrity suite (PS clause 2.2.1).

Runs each data detector, collects Findings, and then aggregates them to source
level. Every detector runs behind ``safe_run``, so one failing check never
takes the audit down with it.
"""

from __future__ import annotations

from cvassure.core.schemas import Finding
from cvassure.detect.base import AuditContext, DetectorResult
from cvassure.detect.contributor import ContributorDetector
from cvassure.detect.label_noise import LabelNoiseDetector
from cvassure.detect.near_duplicate import NearDuplicateDetector
from cvassure.detect.ood import OODDetector
from cvassure.detect.spectral_signature import SpectralSignatureDetector
from cvassure.detect.trigger_freq import TriggerFrequencyDetector


def build_detectors() -> list:
    """Ordered easiest-first, which is also cheapest-first."""
    return [
        NearDuplicateDetector(),
        OODDetector(),
        LabelNoiseDetector(),
        TriggerFrequencyDetector(),
        SpectralSignatureDetector(),
    ]


def run_all(ctx: AuditContext) -> tuple[list[Finding], list[DetectorResult]]:
    findings: list[Finding] = []
    results: list[DetectorResult] = []

    for detector in build_detectors():
        result = detector.safe_run(ctx)
        results.append(result)
        findings.extend(result.findings)

    # Source-level aggregation runs last: it reads what the others produced.
    contributor = ContributorDetector()
    import time

    start = time.perf_counter()
    contributor_findings = contributor.aggregate(ctx, findings)
    results.append(
        DetectorResult(
            contributor.detector_id,
            contributor_findings,
            time.perf_counter() - start,
            0.0,
        )
    )
    findings.extend(contributor_findings)
    return findings, results
