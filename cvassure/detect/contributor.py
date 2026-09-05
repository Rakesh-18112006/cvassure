"""Source-level assessment: which contributor is the problem?

This is the aggregation step, and it is the part an officer can actually act
on. A list of nine hundred suspect image IDs is not a decision; "quarantine
everything from Contributor D" is.

The statistics matter here for a specific reason: a contributor who sent three
images and had one flagged must not be accused at the same strength as one who
sent twelve hundred and had two hundred flagged. The Beta-Binomial posterior
handles that automatically — small samples produce wide intervals, and the
disposition rule reads the interval, not the point estimate.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from cvassure.core.schemas import Finding
from cvassure.detect.base import AuditContext, Detector
from cvassure.ingest.contributors import NO_CONTRIBUTOR_METADATA
from cvassure.score.metrics import beta_binomial_risk

# The rule, stated once. Printed in the report so the judgement is auditable.
DISPOSITION_RULE = (
    "Quarantine when we are more than 90% sure the contributor's true bad rate is "
    "above 5%, and they sent at least 30 images. Review when we are more than 60% "
    "sure. Otherwise accept."
)


class ContributorDetector(Detector):
    """Aggregates sample-level findings into a verdict per source."""

    detector_id = "contributor"
    required_tier = 0
    description = "It works out which contributor the suspect images came from."

    def __init__(self, threshold: float = 0.05, min_n_for_quarantine: int = 30):
        self.threshold = threshold
        self.min_n_for_quarantine = min_n_for_quarantine

    def run(self, ctx: AuditContext, sample_findings: Sequence[Finding] | None = None):
        raise NotImplementedError("use aggregate(); this detector runs after the others")

    # ``ContributorDetector`` is different from the rest: it consumes the other
    # detectors' output rather than the images, so it has its own entry point.
    def aggregate(
        self, ctx: AuditContext, sample_findings: Sequence[Finding], *,
        dimension: str = "contributor_id", asset_type: str = "contributor",
        noun: str = "Contributor",
    ) -> list[Finding]:
        """Roll sample-level evidence up to a source.

        ``dimension`` selects which piece of source metadata to group by. PS
        clause 2.2.1 asks for "contributor, batch or source metadata", and a
        batch is a genuinely different question from a contributor: one honest
        supplier can still send one bad consignment, and rolling that up per
        contributor averages it away.
        """
        attributed = [s for s in ctx.samples if getattr(s, dimension, None)]
        if not attributed:
            return [
                Finding.unavailable(
                    asset_ref="contributors",
                    asset_type="dataset",
                    attack_class="clean",
                    detector_id=self.detector_id,
                    access_tier=ctx.access_tier,
                    reason=(
                        "We cannot say which source is responsible for anything we "
                        "found, because this dataset does not record who supplied "
                        "each image."
                    ),
                    unavailable_reason=NO_CONTRIBUTOR_METADATA,
                )
            ] if dimension == "contributor_id" else []

        contributor_of = {s.sample_id: getattr(s, dimension) for s in attributed}
        totals: dict[str, int] = defaultdict(int)
        for cid in contributor_of.values():
            totals[cid] += 1

        # A sample counts once, no matter how many detectors flagged it —
        # otherwise a contributor is punished for the number of checks we run.
        flagged: dict[str, set[str]] = defaultdict(set)
        reasons: dict[str, defaultdict] = defaultdict(lambda: defaultdict(int))
        for f in sample_findings:
            if f.asset_type != "sample" or f.is_unavailable or f.disposition == "accept":
                continue
            cid = contributor_of.get(f.asset_ref)
            if cid is None:
                continue
            flagged[cid].add(f.asset_ref)
            reasons[cid][f.detector_id] += 1

        risks = {
            cid: beta_binomial_risk(
                len(flagged.get(cid, ())), totals[cid],
                threshold=self.threshold, contributor_id=cid,
            )
            for cid in sorted(totals)
        }
        rates = {cid: r.rate for cid, r in risks.items()}
        others_max = {
            cid: max([v for k, v in rates.items() if k != cid], default=0.0)
            for cid in rates
        }
        others_median = {
            cid: float(np.median([v for k, v in rates.items() if k != cid]))
            if len(rates) > 1
            else 0.0
            for cid in rates
        }

        findings: list[Finding] = []
        for cid in sorted(totals):
            risk = risks[cid]
            disposition = self._disposition(risk)
            severity = {
                "quarantine": "critical", "review": "high", "accept": "low"
            }[disposition]

            top_reason = max(reasons[cid].items(), key=lambda kv: kv[1], default=None)
            what = (
                f" Most of them were caught by the '{top_reason[0]}' check."
                if top_reason
                else ""
            )

            if disposition == "quarantine":
                # Only claim to be the worst source if that is actually true.
                # Telling an officer "every other contributor is under 45%"
                # about the contributor sitting at 12% is both wrong and
                # exactly the kind of overstatement that costs trust.
                if risk.rate >= others_max[cid]:
                    comparison = (
                        f"Every other source is under "
                        f"{100 * others_max[cid]:.0f}%. This is almost certainly "
                        f"deliberate, not bad luck."
                    )
                else:
                    comparison = (
                        f"That is well above the {100 * others_median[cid]:.0f}% we see "
                        f"across the other sources, though it is not the worst here."
                    )
                reason = (
                    f"{noun} {cid} sent {risk.n:,} images and {risk.n_flagged:,} of "
                    f"them look tampered with (about {100 * risk.rate:.0f}%). "
                    f"{comparison}{what} "
                    f"Recommend: quarantine all of {noun.lower()} {cid}'s data."
                )
            elif disposition == "review":
                reason = (
                    f"{noun} {cid} sent {risk.n:,} images and {risk.n_flagged:,} of "
                    f"them look wrong (about {100 * risk.rate:.0f}%). That is higher than "
                    f"we would like, but with this many images we cannot yet rule out "
                    f"bad luck — the true rate could be anywhere between "
                    f"{100 * risk.lo:.0f}% and {100 * risk.hi:.0f}%.{what} "
                    f"Recommend: have someone look at a sample of this {noun.lower()}'s work."
                )
            else:
                reason = (
                    f"{noun} {cid} sent {risk.n:,} images and only "
                    f"{risk.n_flagged:,} of them ({100 * risk.rate:.1f}%) look wrong, "
                    f"which is within what we would expect from an honest source. "
                    f"Recommend: accept."
                )

            findings.append(
                Finding(
                    asset_ref=cid if dimension == "contributor_id" else f"batch:{cid}",
                    asset_type=asset_type,
                    attack_class="systematic_mislabel",
                    detector_id=self.detector_id,
                    access_tier=ctx.access_tier,
                    raw_score=float(risk.p_above_threshold),
                    severity=severity,
                    disposition=disposition,
                    reason=reason,
                    evidence={
                        "n_samples": risk.n,
                        "n_flagged": risk.n_flagged,
                        "flagged_rate": round(risk.rate, 4),
                        "credible_interval_95": [round(risk.lo, 4), round(risk.hi, 4)],
                        "p_rate_above_threshold": round(risk.p_above_threshold, 4),
                        "threshold": self.threshold,
                        "flagged_by_detector": dict(reasons[cid]),
                        "rule": DISPOSITION_RULE,
                    },
                )
            )
        return findings

    def _disposition(self, risk) -> str:
        if risk.p_above_threshold > 0.9 and risk.n >= self.min_n_for_quarantine:
            return "quarantine"
        if risk.p_above_threshold > 0.6:
            return "review"
        return "accept"
