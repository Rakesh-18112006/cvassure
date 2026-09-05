"""``cvassure audit`` — the one command shown to judges.

Loads the data and the model, runs every check the declared access tier
allows, verifies the inference log, records what it did in a tamper-evident
audit trail, and writes a single self-contained HTML report.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import time
from pathlib import Path

from cvassure.core.hashing import file_digest, sha256_hex
from cvassure.core.schemas import Finding
from cvassure.detect import data as data_suite
from cvassure.detect import model as model_suite
from cvassure.detect.base import AuditContext, DetectorResult
from cvassure.detect.embed import EmbeddingStore
from cvassure.ingest.dataset import load_dataset
from cvassure.ingest.models import load_model
from cvassure.report import build as report_build
from cvassure.report import html as report_html
from cvassure.report.audit import AuditLog, environment, verify_audit_log
from cvassure.score.evaluate import write_findings


class Progress:
    """A progress bar that works on a plain terminal and stays quiet when the
    output is being piped somewhere."""

    def __init__(self, total: int, quiet: bool = False):
        self.total = max(1, total)
        self.n = 0
        self.quiet = quiet or not sys.stdout.isatty()

    def step(self, label: str) -> None:
        self.n += 1
        if self.quiet:
            print(f"  [{self.n}/{self.total}] {label}", flush=True)
            return
        filled = int(28 * self.n / self.total)
        bar = "█" * filled + "░" * (28 - filled)
        sys.stdout.write(f"\r  {bar} {self.n}/{self.total}  {label:<28}")
        sys.stdout.flush()

    def done(self) -> None:
        if not self.quiet:
            sys.stdout.write("\r" + " " * 78 + "\r")
            sys.stdout.flush()


def run_audit(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    quiet = getattr(args, "quiet", False)

    print("cvassure — integrity audit")
    print(f"  output: {out_dir}")
    print()

    # -- load ------------------------------------------------------------
    dataset = load_dataset(
        args.dataset,
        fmt=getattr(args, "format", "auto"),
        contributors=getattr(args, "contributors", None),
        contributor_from_path=getattr(args, "contributor_from_path", None),
    )
    print(f"  {len(dataset)} images, {len(dataset.labels)} classes, "
          f"{len(dataset.contributors)} contributors")

    model = None
    if getattr(args, "model", None):
        try:
            model = load_model(args.model, args.access_tier)
            print(f"  model: {model.kind}, access tier {args.access_tier}")
        except Exception as exc:
            print(f"  model could not be loaded ({exc}); continuing with the data checks")

    enrolled = None
    if getattr(args, "enrolled_fingerprint", None):
        enrolled = json.loads(Path(args.enrolled_fingerprint).read_text(encoding="utf-8"))

    reference = None
    if getattr(args, "reference", None):
        from cvassure.detect.shift import load_profile

        reference = load_profile(args.reference)

    ctx = AuditContext(
        samples=dataset.samples,
        access_tier=args.access_tier,
        model=model,
        embeddings=EmbeddingStore(cache_dir=out_dir / ".cache"),
        reference_profile=reference,
        enrolled_fingerprint=enrolled,
        out_dir=out_dir,
        seed=getattr(args, "seed", 0),
        limitations=list(dataset.limitations),
    )

    log = AuditLog.open(out_dir / "audit_log.jsonl")
    input_digests = {
        "dataset": sha256_hex([s.sample_id for s in dataset.samples]),
        "dataset_root": str(dataset.root),
        "n_samples": len(dataset),
        "model_file": model.file_digest() if model else None,
        "environment": environment(),
    }

    # -- run ---------------------------------------------------------------
    print()
    n_steps = len(data_suite.build_detectors()) + len(model_suite.build_detectors()) + 3
    progress = Progress(n_steps, quiet=quiet)

    findings: list[Finding] = []
    results: list[DetectorResult] = []

    for detector in data_suite.build_detectors():
        progress.step(detector.detector_id)
        started = _dt.datetime.now(_dt.timezone.utc).isoformat()
        result = detector.safe_run(ctx)
        results.append(result)
        findings.extend(result.findings)
        log.record_run(result, input_digests=input_digests,
                       config={"detector": detector.detector_id}, seed=ctx.seed,
                       started=started)

    # distribution shift
    progress.step("shift")
    from cvassure.detect.shift import ShiftDetector

    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    shift_result = ShiftDetector().safe_run(ctx)
    results.append(shift_result)
    findings.extend(shift_result.findings)
    log.record_run(shift_result, input_digests=input_digests, config={"detector": "shift"},
                   seed=ctx.seed, started=started)

    # model checks
    if model is not None:
        for detector in model_suite.build_detectors():
            progress.step(detector.detector_id)
            started = _dt.datetime.now(_dt.timezone.utc).isoformat()
            result = detector.safe_run(ctx)
            results.append(result)
            findings.extend(result.findings)
            log.record_run(result, input_digests=input_digests,
                           config={"detector": detector.detector_id}, seed=ctx.seed,
                           started=started)
    else:
        for _ in model_suite.build_detectors():
            progress.step("model checks skipped")

    # contributor aggregation, after everything that feeds it
    progress.step("contributor")
    from cvassure.detect.contributor import ContributorDetector

    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    t0 = time.perf_counter()
    contributor_findings = ContributorDetector().aggregate(ctx, findings)
    contributor_result = DetectorResult(
        "contributor", contributor_findings, time.perf_counter() - t0, 0.0
    )
    results.append(contributor_result)
    findings.extend(contributor_findings)
    log.record_run(contributor_result, input_digests=input_digests,
                   config={"detector": "contributor"}, seed=ctx.seed, started=started)

    # inference receipts
    progress.step("provenance")
    if getattr(args, "receipts", None):
        findings.extend(_verify_receipts(args, ctx, log, input_digests))
    progress.done()

    # -- report -------------------------------------------------------------
    findings_path = write_findings(findings, out_dir / "findings.jsonl")

    # Quote the measured numbers from the results run when they exist, so the
    # coverage table in an operational report says how well each check actually
    # performed rather than merely that it ran.
    coverage = report_build.build_coverage(
        findings,
        access_tier=args.access_tier,
        measured=report_build.measured_from_sweep(Path("results") / "sweep_raw.jsonl"),
    )
    verdict = report_build.overall_verdict(findings, n_samples=len(dataset))

    limitations = list(dataset.limitations)
    if model is not None:
        limitations += [f"This audit could not check by {n}" for n in model.unavailable_checks()]
    encoder = ctx.embeddings.info()
    limitations.append(
        f"Images were compared using {encoder.name}: {encoder.note}."
    )

    report_path = report_html.write(
        out_dir / "report.html",
        findings=findings,
        verdict=verdict,
        coverage=coverage,
        inputs={
            "Dataset": f"{dataset.root} ({dataset.fmt.upper()}, {len(dataset)} images)",
            "Model": f"{args.model} ({model.kind})" if model else "none supplied",
            "Access granted": f"tier {args.access_tier}",
            "Inference log": getattr(args, "receipts", None) or "none supplied",
            "Reference profile": getattr(args, "reference", None) or "none supplied",
            "Network access": "none — this audit ran fully offline",
        },
        limitations=limitations,
        audit_log_result=verify_audit_log(log.path),
    )

    timings = [
        {"module": r.detector_id, "n_samples": len(dataset), "seconds": r.seconds,
         "peak_ram_mb": r.peak_ram_mb}
        for r in results
    ]
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")

    # -- the console summary a judge reads across the room ------------------
    print()
    print("=" * 66)
    print(verdict.render())
    print("=" * 66)
    print()
    errors = [r for r in results if r.error]
    if errors:
        print(f"  {len(errors)} check(s) could not finish: "
              f"{', '.join(r.detector_id for r in errors)}")
    print(f"  Report:      {report_path}")
    print(f"  Findings:    {findings_path}")
    print(f"  Audit trail: {log.path}  ({len(log.entries)} records, hash-chained)")

    return 0 if verdict.colour == "green" else 2


def _verify_receipts(args, ctx, log, input_digests) -> list[Finding]:
    from cvassure.provenance.verify import findings_from_result, verify_log
    from cvassure.provenance.receipts import load_public_key, read_receipts

    pubkey = Path(getattr(args, "pubkey", "keys/pub.pem"))
    if not pubkey.exists():
        return [
            Finding.unavailable(
                asset_ref="inference_log",
                asset_type="receipt",
                attack_class="clean",
                detector_id="provenance.verify",
                access_tier=args.access_tier,
                reason=(
                    "We could not check the inference log because the public key it "
                    "was signed with was not supplied."
                ),
                unavailable_reason=f"public key not found at {pubkey}",
            )
        ]
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    t0 = time.perf_counter()
    result = verify_log(read_receipts(args.receipts), load_public_key(pubkey))
    findings = findings_from_result(result, access_tier=args.access_tier)
    log.record_run(
        DetectorResult("provenance.verify", findings, time.perf_counter() - t0, 0.0),
        input_digests={**input_digests, "receipts": file_digest(args.receipts)},
        config={"detector": "provenance.verify"},
        seed=ctx.seed,
        started=started,
    )
    return findings
