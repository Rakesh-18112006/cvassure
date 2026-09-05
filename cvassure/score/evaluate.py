"""Join findings to the answer key and produce the four results tables.

This is the only module that is allowed to see both sides at once. If the
numbers here look impossible — every AUROC near 0.5, say — the first thing to
check is the join, not the detectors.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from cvassure.core.schemas import Finding
from cvassure.score import metrics
from cvassure.score.calibrate import Splits, calibrate_findings, three_way_split
from cvassure.score.tables import Table

# Which detector is the designated answer for which attack. A detector is
# scored on every attack class, but the coverage statement quotes the number
# from the detector that is supposed to catch it.
PRIMARY_DETECTOR: dict[str, tuple[str, ...]] = {
    "badnets_patch": ("trigger_freq", "spectral_signature"),
    "blended_trigger": ("trigger_freq", "spectral_signature"),
    "label_flip": ("label_noise",),
    "systematic_mislabel": ("label_noise", "contributor"),
    "near_duplicate_flood": ("near_duplicate",),
    "ood_insertion": ("ood",),
    "model_substitute": ("fingerprint", "weight_digest"),
    "model_perturb": ("weight_digest", "weight_stats"),
    "model_backdoor": ("trigger_recon", "weight_stats"),
    "receipt_alter": ("provenance.verify",),
    "receipt_replay": ("provenance.verify",),
    "receipt_delete": ("provenance.verify",),
    "receipt_reorder": ("provenance.verify",),
    "distribution_shift": ("shift",),
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def read_findings(path: str | Path) -> list[Finding]:
    out: list[Finding] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Finding.from_dict(json.loads(line)))
    return out


def write_findings(findings: Iterable[Finding], path: str | Path) -> Path:
    from cvassure.core.hashing import canonical_json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for f in findings:
            fh.write(canonical_json(f.to_dict()).decode("utf-8") + "\n")
    return p


def read_truth(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "ground_truth.json"
    if not p.exists():
        raise FileNotFoundError(f"no answer key at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The join
# --------------------------------------------------------------------------


@dataclass
class Joined:
    """Findings paired with the truth about the asset they refer to."""

    truth: dict[str, Any]
    findings: list[Finding]
    truth_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)

    @property
    def sample_ids(self) -> list[str]:
        return sorted(self.truth_by_id)

    def label_of(self, asset_ref: str) -> int | None:
        row = self.truth_by_id.get(asset_ref)
        return None if row is None else int(row["is_poisoned"])

    def attack_of(self, asset_ref: str) -> str:
        row = self.truth_by_id.get(asset_ref)
        return row["attack_class"] if row else "clean"

    def binary_labels(self) -> dict[str, int]:
        return {k: int(v["is_poisoned"]) for k, v in self.truth_by_id.items()}


def join(truth: dict[str, Any], findings: Sequence[Finding]) -> Joined:
    """Match ``Finding.asset_ref`` to ``ground_truth.samples[].sample_id``.

    A silent mismatch here is the single most likely reason for a detector to
    look broken when it is fine, so the unmatched list is kept and reported
    rather than dropped.
    """
    truth_by_id = {r["sample_id"]: r for r in truth.get("samples", [])}
    sample_findings = [f for f in findings if f.asset_type == "sample"]
    unmatched = sorted({f.asset_ref for f in sample_findings} - set(truth_by_id))
    return Joined(
        truth=truth, findings=list(findings), truth_by_id=truth_by_id, unmatched=unmatched
    )


def _scores_for(
    joined: Joined,
    *,
    detector: str | None = None,
    tier: int | None = None,
    restrict_to: set[str] | None = None,
    attack_class: str | None = None,
    use_calibrated: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (y_true, y_score) over samples.

    A sample that a detector never scored counts as score 0 — silence is a
    prediction of "clean", and pretending otherwise would quietly inflate
    every number.
    """
    ids = set(joined.truth_by_id)
    if restrict_to is not None:
        ids &= restrict_to
    if attack_class is not None:
        # score this attack against the clean samples only, so that one
        # attack's numbers are not diluted by another's
        ids = {
            i
            for i in ids
            if joined.truth_by_id[i]["attack_class"] in (attack_class, "clean")
            or joined.truth_by_id[i]["is_poisoned"] == 0
        }

    best: dict[str, float] = {i: 0.0 for i in ids}
    for f in joined.findings:
        if f.asset_type != "sample" or f.is_unavailable or f.asset_ref not in best:
            continue
        if detector is not None and f.detector_id != detector:
            continue
        if tier is not None and f.access_tier != tier:
            continue
        value = f.score if use_calibrated else f.raw_score
        best[f.asset_ref] = max(best[f.asset_ref], float(value))

    order = sorted(best)
    y = np.array([joined.truth_by_id[i]["is_poisoned"] for i in order], dtype=np.int64)
    s = np.array([best[i] for i in order], dtype=np.float64)
    return y, s


# --------------------------------------------------------------------------
# Table 1 — detection by attack class and access tier
# --------------------------------------------------------------------------


def table1_detection(
    joined: Joined,
    *,
    test_ids: set[str] | None = None,
    n_bootstrap: int = 400,
    seed: int = 0,
) -> Table:
    t = Table(
        name="table1_detection",
        title="Table 1 — Detection performance by attack class and access tier",
        columns=[
            "attack_class", "access_tier", "detector", "n", "n_poisoned",
            "AUROC [95% CI]", "TPR@1%FPR [95% CI]", "ECE", "verdict",
        ],
        notes=[
            "TPR@1%FPR is the operational number: the share of poison still caught "
            "when only one clean image in a hundred may be falsely flagged.",
            "verdict: strong >=0.80, good >=0.60, partial >=0.35, unsupported <0.35, "
            "measured on TPR@1%FPR.",
        ],
    )

    tiers = sorted({f.access_tier for f in joined.findings}) or [0]
    attacks = sorted(
        {r["attack_class"] for r in joined.truth_by_id.values() if r["is_poisoned"]}
    )

    for attack in attacks:
        for tier in tiers:
            detectors = [
                d for d in PRIMARY_DETECTOR.get(attack, ())
                if any(f.detector_id == d and f.access_tier == tier for f in joined.findings)
            ]
            # "any detector" row first, then the designated ones
            for detector in [None, *detectors]:
                y, s = _scores_for(
                    joined,
                    detector=detector,
                    tier=tier,
                    restrict_to=test_ids,
                    attack_class=attack,
                )
                if y.size == 0 or y.sum() == 0:
                    continue
                auc = metrics.bootstrap_ci(metrics.roc_auc, y, s, n=n_bootstrap, seed=seed)
                tpr = metrics.bootstrap_ci(
                    lambda a, b: metrics.tpr_at_fpr(a, b, 0.01), y, s,
                    n=n_bootstrap, seed=seed + 1,
                )
                t.add(
                    attack_class=attack,
                    access_tier=tier,
                    detector=detector or "any detector",
                    n=int(y.size),
                    n_poisoned=int(y.sum()),
                    **{
                        "AUROC [95% CI]": str(auc),
                        "TPR@1%FPR [95% CI]": str(tpr),
                    },
                    ECE=f"{metrics.ece(y, s):.3f}",
                    verdict=metrics.verdict(tpr.value),
                )
    return t


# --------------------------------------------------------------------------
# Table 2 — contributor risk
# --------------------------------------------------------------------------


def table2_contributors(joined: Joined, *, threshold: float = 0.05) -> Table:
    t = Table(
        name="table2_contributors",
        title="Table 2 — Contributor risk against ground truth",
        columns=[
            "contributor", "n_samples", "n_flagged", "flagged_rate",
            "95% credible interval", f"P(rate > {threshold:.0%})", "disposition",
            "true_poison_rate", "matches_truth",
        ],
        notes=[
            "The interval is a Beta-Binomial posterior, so a contributor who sent "
            "three images gets a wide interval instead of an accusation.",
            "disposition: quarantine if P>0.9 and n>=30, review if P>0.6, else accept.",
        ],
    )

    flagged: dict[str, set[str]] = defaultdict(set)
    for f in joined.findings:
        if f.asset_type == "sample" and not f.is_unavailable and f.disposition != "accept":
            row = joined.truth_by_id.get(f.asset_ref)
            if row and row.get("contributor_id"):
                flagged[row["contributor_id"]].add(f.asset_ref)

    totals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined.truth_by_id.values():
        if row.get("contributor_id"):
            totals[row["contributor_id"]].append(row)

    for cid in sorted(totals):
        rows = totals[cid]
        n = len(rows)
        n_flagged = len(flagged.get(cid, ()))
        risk = metrics.beta_binomial_risk(
            n_flagged, n, threshold=threshold, contributor_id=cid
        )
        true_rate = float(np.mean([r["is_poisoned"] for r in rows]))
        disposition = contributor_disposition(risk)
        truly_bad = true_rate > threshold
        called_bad = disposition != "accept"
        t.add(
            contributor=cid,
            n_samples=n,
            n_flagged=n_flagged,
            flagged_rate=f"{risk.rate:.3f}",
            **{
                "95% credible interval": f"[{risk.lo:.3f}, {risk.hi:.3f}]",
                f"P(rate > {threshold:.0%})": f"{risk.p_above_threshold:.3f}",
            },
            disposition=disposition,
            true_poison_rate=f"{true_rate:.3f}",
            matches_truth="yes" if truly_bad == called_bad else "NO",
        )
    return t


def contributor_disposition(risk: metrics.ContributorRisk) -> str:
    """The rule, stated once and used everywhere it is quoted."""
    if risk.p_above_threshold > 0.9 and risk.n >= 30:
        return "quarantine"
    if risk.p_above_threshold > 0.6:
        return "review"
    return "accept"


# --------------------------------------------------------------------------
# Table 3 — tamper detection
# --------------------------------------------------------------------------


def table3_tamper(truth: dict[str, Any], results: Sequence[dict[str, Any]]) -> Table:
    """``results`` is one entry per attempted receipt attack, each recording
    whether verification caught it and with which code."""
    t = Table(
        name="table3_tamper",
        title="Table 3 — Tamper detection on the inference log",
        columns=["attack_type", "attempts", "detected", "rate", "codes_reported"],
        notes=[
            "This is cryptography, not statistics: the maths either matches or it "
            "does not, so anything below 100% is a bug to fix, not a limitation "
            "to report.",
        ],
    )
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_kind[r["attack_class"]].append(r)

    for kind in sorted(by_kind):
        entries = by_kind[kind]
        detected = sum(1 for e in entries if e.get("detected"))
        codes = sorted({c for e in entries for c in e.get("codes", [])})
        t.add(
            attack_type=kind,
            attempts=len(entries),
            detected=detected,
            rate=f"{100 * detected / max(1, len(entries)):.1f}%",
            codes_reported=", ".join(codes) or "—",
        )
    return t


# --------------------------------------------------------------------------
# Table 4 — runtime
# --------------------------------------------------------------------------


def table4_runtime(timings: Sequence[dict[str, Any]]) -> Table:
    t = Table(
        name="table4_runtime",
        title="Table 4 — Runtime and memory, per module, per 1000 samples",
        columns=["module", "n_samples", "seconds", "seconds_per_1000", "peak_RAM_MB"],
        notes=["Measured on the machine that produced this report; see RESULTS.md "
               "for the hardware."],
    )
    for row in timings:
        n = max(1, int(row.get("n_samples", 0)))
        secs = float(row.get("seconds", 0.0))
        t.add(
            module=row["module"],
            n_samples=n,
            seconds=f"{secs:.2f}",
            seconds_per_1000=f"{1000 * secs / n:.2f}",
            peak_RAM_MB=f"{float(row.get('peak_ram_mb', 0.0)):.1f}",
        )
    return t


# --------------------------------------------------------------------------
# Confusion of dispositions
# --------------------------------------------------------------------------


def disposition_confusion(joined: Joined, test_ids: set[str] | None = None) -> np.ndarray:
    """rows = truth (clean, poisoned), cols = accept / review / quarantine."""
    order = ["accept", "review", "quarantine"]
    m = np.zeros((2, 3), dtype=int)
    worst: dict[str, str] = {}
    for f in joined.findings:
        if f.asset_type != "sample" or f.is_unavailable:
            continue
        if test_ids is not None and f.asset_ref not in test_ids:
            continue
        if f.asset_ref not in joined.truth_by_id:
            continue
        current = worst.get(f.asset_ref, "accept")
        if order.index(f.disposition) > order.index(current):
            worst[f.asset_ref] = f.disposition
        worst.setdefault(f.asset_ref, f.disposition)
    scope = test_ids if test_ids is not None else set(joined.truth_by_id)
    for sid in scope & set(joined.truth_by_id):
        y = int(joined.truth_by_id[sid]["is_poisoned"])
        m[y, order.index(worst.get(sid, "accept"))] += 1
    return m


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class ScoreResult:
    tables: list[Table]
    joined: Joined
    splits: Splits
    calibrators: dict[str, Any]
    out_dir: Path
    plots: dict[str, Path] = field(default_factory=dict)
    ece_before: float = float("nan")
    ece_after: float = float("nan")


def score_all(
    *,
    truth: str | Path,
    findings: str | Path,
    out: str | Path,
    tamper_results: Sequence[dict[str, Any]] | None = None,
    timings: Sequence[dict[str, Any]] | None = None,
    n_bootstrap: int = 400,
    seed: int = 0,
    make_plots: bool = True,
) -> ScoreResult:
    truth_data = read_truth(truth)
    found = read_findings(findings) if not isinstance(findings, list) else findings
    joined = join(truth_data, found)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- splits and calibration -----------------------------------------
    labels = joined.binary_labels()
    ids = sorted(labels)
    splits = three_way_split(
        ids, stratify=[labels[i] for i in ids], seed=seed
    )
    raw_findings = list(joined.findings)
    calibrated, calibrators = calibrate_findings(raw_findings, labels, splits)
    test_ids = set(splits.test)

    joined_raw = join(truth_data, raw_findings)
    joined.findings = calibrated

    y_before, s_before = _scores_for(joined_raw, restrict_to=test_ids, use_calibrated=False)
    y_after, s_after = _scores_for(joined, restrict_to=test_ids, use_calibrated=True)
    ece_before = metrics.ece(y_before, s_before)
    ece_after = metrics.ece(y_after, s_after)

    # -- tables ----------------------------------------------------------
    tables = [
        table1_detection(joined, test_ids=test_ids, n_bootstrap=n_bootstrap, seed=seed),
        table2_contributors(joined),
        table3_tamper(truth_data, tamper_results or _tamper_from_truth(truth_data)),
        table4_runtime(timings or []),
    ]
    for t in tables:
        t.write(out_dir / "tables")

    result = ScoreResult(
        tables=tables,
        joined=joined,
        splits=splits,
        calibrators=calibrators,
        out_dir=out_dir,
        ece_before=ece_before,
        ece_after=ece_after,
    )

    if make_plots:
        from cvassure.score import plots

        result.plots = plots.render_all(result, joined_raw=joined_raw, test_ids=test_ids)

    _write_results_md(result)
    return result


def _tamper_from_truth(truth: dict[str, Any]) -> list[dict[str, Any]]:
    """If the caller did not run the receipt attacks, report zero attempts
    rather than an empty table with no explanation."""
    return [
        {"attack_class": r["attack_class"], "detected": False, "codes": []}
        for r in truth.get("receipt_attacks", [])
    ]


def _write_results_md(result: ScoreResult) -> Path:
    lines = [
        "# Results",
        "",
        f"Answer key: {len(result.joined.truth_by_id)} samples, "
        f"{sum(r['is_poisoned'] for r in result.joined.truth_by_id.values())} poisoned.",
        f"Findings: {len(result.joined.findings)}.",
        f"Split sizes: {result.splits.sizes()} (fit / calibrate / test — no sample "
        f"appears in more than one).",
        "",
        f"Calibration error before calibration: {result.ece_before:.3f}; "
        f"after: {result.ece_after:.3f}. Lower is better; under 0.05 is good.",
        "",
    ]
    if result.joined.unmatched:
        lines += [
            f"> **Warning:** {len(result.joined.unmatched)} findings refer to assets "
            f"that are not in the answer key, for example "
            f"`{result.joined.unmatched[0]}`. Check the join before reading anything "
            f"below.",
            "",
        ]
    for t in result.tables:
        lines.append(t.to_markdown())
        lines.append("")
    if result.plots:
        lines.append("### Figures")
        lines.append("")
        for name, path in sorted(result.plots.items()):
            lines.append(f"- `{name}` — {Path(path).name}")
        lines.append("")

    p = result.out_dir / "RESULTS.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
