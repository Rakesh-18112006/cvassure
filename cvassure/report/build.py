"""Assembling the assurance report, including the coverage statement.

The coverage matrix is generated from what actually ran and what it actually
measured. Never hand-written: a hand-written coverage table drifts out of
sync with the code within a week, and the one thing it must never do is
overstate what the system can do.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from cvassure.core.schemas import Finding, severity_for
from cvassure.score.tables import Table

#: Which detector answers for which attack, and what a human should call it.
ATTACK_LABELS: dict[str, str] = {
    "badnets_patch": "A marker pasted onto the image",
    "blended_trigger": "A faint pattern laid over the whole image",
    "label_flip": "The picture is fine, the label is wrong",
    "systematic_mislabel": "One contributor labels everything the same wrong way",
    "near_duplicate_flood": "The same photograph submitted many times",
    "ood_insertion": "Images from a completely different source",
    "model_substitute": "The model file was swapped",
    "model_perturb": "The model's weights were edited",
    "model_backdoor": "A hidden trigger was trained into the model",
    "receipt_alter": "An inference record was edited afterwards",
    "receipt_replay": "An old inference record was submitted again",
    "receipt_delete": "An inference record was removed",
    "receipt_reorder": "The inference records were shuffled",
    "distribution_shift": "The incoming data has drifted from normal",
}

SUPPORTED = "supported"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
NOT_MEASURED = "not measured"


@dataclass
class CoverageRow:
    attack_class: str
    access_tier: int
    status: str
    detector: str | None
    measured: str | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class Coverage:
    rows: list[CoverageRow] = field(default_factory=list)

    def table(self) -> Table:
        t = Table(
            name="coverage",
            title="Coverage — what this system can and cannot detect",
            columns=["what an attacker did", "access tier", "status", "check",
                     "measured", "note"],
            notes=[
                "Generated from the checks that actually ran in this audit and what "
                "they measured. Nothing in this table is written by hand.",
                "'not measured' is not the same as 'works' — it means we have no "
                "number for it and you should not rely on it.",
            ],
        )
        for r in sorted(self.rows, key=lambda r: (r.attack_class, r.access_tier)):
            t.add(
                **{
                    "what an attacker did": ATTACK_LABELS.get(r.attack_class, r.attack_class),
                    "access tier": r.access_tier,
                    "status": r.status,
                    "check": r.detector or "—",
                    "measured": r.measured or "—",
                    "note": r.note,
                }
            )
        return t

    def unsupported(self) -> list[CoverageRow]:
        return [r for r in self.rows if r.status in (UNSUPPORTED, NOT_MEASURED)]

    def statement(self) -> list[str]:
        """The plain-English coverage statement, including the bad news."""
        by_status: dict[str, list[CoverageRow]] = defaultdict(list)
        for r in self.rows:
            by_status[r.status].append(r)

        lines: list[str] = []
        if by_status[SUPPORTED]:
            names = sorted({ATTACK_LABELS.get(r.attack_class, r.attack_class)
                            for r in by_status[SUPPORTED]})
            lines.append(
                "We can detect these reliably, and we have measured it: "
                + "; ".join(names)
                + "."
            )
        if by_status[PARTIAL]:
            names = sorted({ATTACK_LABELS.get(r.attack_class, r.attack_class)
                            for r in by_status[PARTIAL]})
            lines.append(
                "We catch these some of the time, and you should treat a clean result "
                "here as weak evidence: " + "; ".join(names) + "."
            )
        if by_status[UNSUPPORTED]:
            names = sorted({ATTACK_LABELS.get(r.attack_class, r.attack_class)
                            for r in by_status[UNSUPPORTED]})
            lines.append(
                "We cannot detect these at the access level this audit was given: "
                + "; ".join(names)
                + ". If you need them covered, the system needs deeper access to the "
                "model, or a different technique."
            )
        if by_status[NOT_MEASURED]:
            names = sorted({ATTACK_LABELS.get(r.attack_class, r.attack_class)
                            for r in by_status[NOT_MEASURED]})
            lines.append(
                "These checks ran, but nothing of that kind was present in this batch, "
                "so we have no measurement of how well they work here: "
                + "; ".join(names)
                + "."
            )
        return lines


def build_coverage(
    findings: Sequence[Finding],
    *,
    access_tier: int,
    measured: dict[str, dict[str, Any]] | None = None,
) -> Coverage:
    """Derive the coverage matrix from the findings and, when available, from
    the measured numbers in ``results/RESULTS.md``.

    ``measured`` maps attack_class -> {"tpr_at_1pct": float, "verdict": str}.
    """
    from cvassure.score.evaluate import PRIMARY_DETECTOR

    measured = measured or {}
    ran: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        ran[f.detector_id].append(f)

    coverage = Coverage()
    for attack, detectors in PRIMARY_DETECTOR.items():
        available = [d for d in detectors if d in ran and any(not f.is_unavailable
                                                              for f in ran[d])]
        blocked = [d for d in detectors if d in ran and all(f.is_unavailable for f in ran[d])]

        if not available:
            why = "no check for this ran in this audit"
            if blocked:
                reasons = sorted({
                    f.unavailable_reason for d in blocked for f in ran[d]
                    if f.unavailable_reason
                })
                why = reasons[0] if reasons else why
            coverage.rows.append(
                CoverageRow(attack, access_tier, UNSUPPORTED, None, None, why)
            )
            continue

        detector = available[0]
        stats = measured.get(attack)
        if stats and stats.get("tpr_at_1pct") is not None:
            tpr = stats["tpr_at_1pct"]
            verdict = stats.get("verdict", "")
            status = (
                SUPPORTED if verdict in ("strong", "good")
                else PARTIAL if verdict == "partial"
                else UNSUPPORTED
            )
            coverage.rows.append(
                CoverageRow(
                    attack, access_tier, status, detector,
                    f"{100 * tpr:.0f}% caught at a 1% false-alarm budget",
                    f"graded '{verdict}' against a known answer key",
                )
            )
        else:
            coverage.rows.append(
                CoverageRow(
                    attack, access_tier, NOT_MEASURED, detector, None,
                    "the check ran, but this batch contained no example of this to "
                    "measure against",
                )
            )
    return coverage


def measured_from_sweep(path: str | Any = "results/sweep_raw.jsonl") -> dict[str, dict[str, Any]]:
    """Read the results run so a report can quote what was actually measured.

    Graded on the median across all swept cells rather than the best one, for
    the same reason ``aggregate.py`` does: quoting the best cell turns a
    detector that usually catches a fifth of an attack into a row marked
    "good". Returns an empty mapping when the sweep has not been run, which
    leaves every row honestly marked "not measured".
    """
    import json
    from pathlib import Path as _Path
    from statistics import median

    p = _Path(path)
    if not p.exists():
        return {}

    from cvassure.score.metrics import verdict as grade

    per: dict[str, list[float]] = defaultdict(list)
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        value = row.get("tpr_at_1pct")
        if "error" in row or value is None or value != value:  # NaN-safe
            continue
        per[row["attack"]].append(float(value))

    out: dict[str, dict[str, Any]] = {}
    for attack, values in per.items():
        typical = float(median(values))
        out[attack] = {
            "tpr_at_1pct": typical,
            "verdict": grade(typical),
            "n_cells": len(values),
        }
    return out


def measured_from_tables(table1) -> dict[str, dict[str, Any]]:
    """Pull the headline numbers out of Table 1 so coverage cannot disagree
    with the results section."""
    out: dict[str, dict[str, Any]] = {}
    for row in getattr(table1, "rows", []):
        if row.get("detector") != "any detector":
            continue
        text = str(row.get("TPR@1%FPR [95% CI]", ""))
        try:
            value = float(text.split()[0])
        except (ValueError, IndexError):
            continue
        attack = row["attack_class"]
        # keep the best tier for each attack
        if attack not in out or value > out[attack]["tpr_at_1pct"]:
            out[attack] = {"tpr_at_1pct": value, "verdict": row.get("verdict", "")}
    return out


# --------------------------------------------------------------------------
# The overall verdict
# --------------------------------------------------------------------------


@dataclass
class Verdict:
    headline: str
    colour: str
    lines: list[str] = field(default_factory=list)

    def render(self) -> str:
        return "\n".join([f"VERDICT: {self.headline}", *self.lines])


#: The two checks that can actually establish "this is the model we were given".
#: `weight_stats` looks for a lopsided layer and `spectral_signature` looks at
#: activations — both are worth running, but neither can tell you the file is
#: the same one, so neither may be used to justify the words "verified
#: unchanged".
IDENTITY_CHECKS = ("fingerprint", "weight_digest")


def overall_verdict(findings: Sequence[Finding], *, n_samples: int = 0) -> Verdict:
    """The traffic light at the top of the report."""
    contributors = [f for f in findings if f.asset_type == "contributor" and not f.is_unavailable]
    models = [f for f in findings if f.asset_type == "model" and not f.is_unavailable]
    receipts = [f for f in findings if f.asset_type == "receipt" and not f.is_unavailable]

    # A check that could not run is not a check that passed. Keeping the blocked
    # model findings lets the headline say "we did not look" instead of
    # "we looked and it was fine".
    models_blocked = [f for f in findings if f.asset_type == "model" and f.is_unavailable]

    quarantined = [f for f in contributors if f.disposition == "quarantine"]
    review = [f for f in contributors if f.disposition == "review"]
    model_bad = [f for f in models if f.disposition == "quarantine"]
    model_review = [f for f in models if f.disposition == "review"]
    receipt_bad = [f for f in receipts if f.disposition == "quarantine"]

    lines: list[str] = []
    if contributors:
        clean = len(contributors) - len(quarantined) - len(review)
        parts = [f"{clean} of {len(contributors)} contributors clean"]
        if quarantined:
            parts.append(
                f"{len(quarantined)} contributor{'s' if len(quarantined) > 1 else ''} "
                f"quarantined ({', '.join(f.asset_ref for f in quarantined)})"
            )
        if review:
            parts.append(f"{len(review)} to review")
        lines.append(".  ".join(parts) + ".")

    model_unverified = False
    if models or models_blocked:
        identity_ran = [f for f in models if f.detector_id in IDENTITY_CHECKS]
        if model_bad:
            lines.append("Model: SUBSTITUTED OR EDITED.")
        elif model_review:
            lines.append("Model: NEEDS REVIEW.")
        elif identity_ran:
            lines.append("Model: verified unchanged.")
        else:
            # Nothing that could establish identity was able to run. Saying
            # "verified unchanged" here would be a false assurance — exactly the
            # claim this system exists to avoid making.
            model_unverified = True
            why = next(
                (f.unavailable_reason for f in models_blocked
                 if f.detector_id in IDENTITY_CHECKS and f.unavailable_reason),
                "the checks that establish model identity could not run",
            )
            lines.append(
                f"Model: NOT VERIFIED — {why}. A swapped or edited model would "
                "not have been caught."
            )

    if receipts:
        n_bad = len(receipt_bad)
        total = max(
            (int(f.evidence.get("total", 0)) for f in receipts if "total" in f.evidence),
            default=0,
        )
        if n_bad:
            which = ", ".join(
                f"#{f.evidence.get('seq')}" for f in receipt_bad[:3] if f.evidence.get("seq")
            )
            lines.append(
                f"Inference log: {n_bad} record{'s' if n_bad > 1 else ''} tampered"
                + (f" ({which})." if which else ".")
            )
        else:
            lines.append(f"Inference log: all {total} records verified.")

    if quarantined or model_bad or receipt_bad:
        return Verdict("QUARANTINE RECOMMENDED", "red", lines)
    if review or model_review:
        return Verdict("REVIEW BEFORE USE", "amber", lines)
    if model_unverified:
        # Green would read as "the model is fine". It is not that; it is
        # "we were not given what we needed to say".
        return Verdict("NO PROBLEMS FOUND IN WHAT WE COULD CHECK", "amber", lines)
    return Verdict("NO PROBLEMS FOUND", "green", lines)


# --------------------------------------------------------------------------
# The one-page summary: an assessment id, an overall risk word, an overall
# confidence number, and a single machine-readable record — the rollup layer
# an analyst skims before reading any of the sections above.
# --------------------------------------------------------------------------

#: The verdict's traffic-light colour already carries the risk judgement;
#: this just gives it the word a governance document expects.
RISK_FOR_COLOUR: dict[str, str] = {"green": "LOW", "amber": "MEDIUM", "red": "HIGH"}


def overall_risk(verdict: Verdict) -> str:
    return RISK_FOR_COLOUR.get(verdict.colour, "MEDIUM")


def make_assessment_id(seed: str, *, when: _dt.datetime | None = None) -> str:
    """A short, stable id for one run, derived from it rather than a counter.

    Two audits of the same input at the same moment would collide on a
    counter kept only in memory; hashing the seed (a run id, or the run's
    input digests) makes the id reproducible instead of merely unique.
    """
    when = when or _dt.datetime.now(_dt.timezone.utc)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:6].upper()
    return f"CVA-{when.strftime('%Y%m%d')}-{digest}"


def overall_confidence(findings: Sequence[Finding]) -> float | None:
    """The average confidence across every check that actually stated one.

    Not every detector can: a hash comparison is binary and a statistical
    shape argument is not, so ``Finding.confidence`` is often ``None``. This
    returns ``None`` rather than a number when nothing here reported one —
    a single invented aggregate would be exactly the kind of overstatement
    this system exists to avoid.
    """
    values = [
        f.confidence for f in findings
        if not f.is_unavailable and f.confidence is not None
    ]
    return float(sum(values) / len(values)) if values else None


#: Attack classes grouped the way an analyst thinks about training-data risk,
#: rather than by which detector happens to answer for them.
DATA_RISK_CATEGORIES: dict[str, tuple[str, ...]] = {
    "trigger_pattern": ("badnets_patch", "blended_trigger"),
    "label_anomaly": ("label_flip", "systematic_mislabel"),
    "duplicate_flood": ("near_duplicate_flood",),
    "ood_distribution": ("ood_insertion",),
}

DATA_RISK_LABELS: dict[str, str] = {
    "trigger_pattern": "Trigger / pattern risk",
    "label_anomaly": "Label anomaly risk",
    "duplicate_flood": "Duplicate / near-duplicate risk",
    "ood_distribution": "OOD / distribution risk",
}


def data_category_risk(findings: Sequence[Finding]) -> list[dict[str, Any]]:
    """Roll sample-level findings up into the four risk categories an
    operational report groups training-data problems into.

    ``level`` is the qualitative read (low/medium/high/critical) of the worst
    score seen in that category this run; ``status`` is the disposition an
    analyst would act on. A category with no matching samples in this run
    reports zero findings rather than being omitted, so the table always
    answers "did we look" as well as "what did we find".
    """
    samples = [f for f in findings if f.asset_type == "sample" and not f.is_unavailable]
    out = []
    for key, classes in DATA_RISK_CATEGORIES.items():
        members = [f for f in samples if f.attack_class in classes]
        flagged = [f for f in members if f.disposition != "accept"]
        top_score = max((f.score for f in members), default=0.0)
        status = (
            "quarantine" if any(f.disposition == "quarantine" for f in flagged)
            else "review" if flagged
            else "accept"
        )
        out.append(
            {
                "category": DATA_RISK_LABELS[key],
                "n_checked": len(members),
                "n_flagged": len(flagged),
                "level": severity_for(top_score) if members else "low",
                "status": status,
            }
        )
    return out


def _worst_disposition(items: Sequence[Finding]) -> str:
    order = {"accept": 0, "review": 1, "quarantine": 2}
    worst = "accept"
    for f in items:
        if not f.is_unavailable and order.get(f.disposition, 0) > order.get(worst, 0):
            worst = f.disposition
    return worst


def assurance_record(
    *,
    assessment_id: str,
    verdict: Verdict,
    findings: Sequence[Finding],
    dataset_summary: dict[str, Any] | None = None,
    model_info: dict[str, Any] | None = None,
    receipts_result: dict[str, Any] | None = None,
    limitations: Sequence[str] = (),
) -> dict[str, Any]:
    """The single, consolidated machine-readable record for one run.

    Everywhere else in cvassure deliberately keeps findings as a flat list —
    that is what lets a scoring harness and an HTML report both consume it
    without agreeing on anything else. This is the one place that rolls that
    list up into the shape a downstream system integration actually wants:
    one object, one verdict, one risk word, one confidence number.
    """
    samples = [f for f in findings if f.asset_type == "sample"]
    models = [f for f in findings if f.asset_type == "model"]
    receipts = [f for f in findings if f.asset_type == "receipt"]
    contributors = {
        f.asset_ref: f.disposition
        for f in findings
        if f.asset_type == "contributor"
        and not f.is_unavailable
        and not str(f.asset_ref).startswith("batch:")
    }

    record: dict[str, Any] = {
        "assessment_id": assessment_id,
        "verdict": verdict.headline,
        "risk": overall_risk(verdict),
        "confidence": overall_confidence(findings),
    }
    if dataset_summary is not None:
        record["dataset"] = {
            "status": _worst_disposition(samples),
            "n_samples": dataset_summary.get("n_samples"),
            "n_flagged": sum(
                1 for f in samples if not f.is_unavailable and f.disposition != "accept"
            ),
        }
    if model_info is not None:
        record["model"] = {
            "status": _worst_disposition(models),
            "kind": model_info.get("kind"),
            "access_tier": model_info.get("declared_access_tier"),
        }
    if receipts_result is not None:
        record["inference"] = {
            "status": _worst_disposition(receipts),
            "total": receipts_result.get("total"),
            "tampered_records": len(receipts_result.get("failures", [])),
        }
    if contributors:
        record["contributors"] = contributors
    record["limitations"] = list(limitations)
    record["recommendation"] = verdict.lines[0] if verdict.lines else verdict.headline
    return record
