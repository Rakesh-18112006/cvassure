"""Trust nothing here until the sanity tests pass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvassure.core.schemas import Finding
from cvassure.score import metrics
from cvassure.score.calibrate import (
    IsotonicCalibrator,
    SplitLeakage,
    Splits,
    calibrate_findings,
    three_way_split,
)
from cvassure.score.evaluate import (
    read_findings,
    score_all,
    write_findings,
)
from cvassure.score.tables import Table


# ==========================================================================
# THE SANITY TESTS — nothing below is meaningful until these two pass
# ==========================================================================


def test_random_scores_give_auroc_of_a_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    s = rng.random(4000)
    assert metrics.roc_auc(y, s) == pytest.approx(0.5, abs=0.03)


def test_perfect_scores_give_auroc_of_one():
    y = np.r_[np.zeros(500, int), np.ones(500, int)]
    s = np.r_[np.zeros(500), np.ones(500)] + np.linspace(0, 0.1, 1000)
    assert metrics.roc_auc(y, s) == pytest.approx(1.0)
    assert metrics.tpr_at_fpr(y, s, 0.01) == pytest.approx(1.0)


def test_inverted_scores_give_auroc_of_zero():
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    s = np.r_[np.ones(200), np.zeros(200)]
    assert metrics.roc_auc(y, s) == pytest.approx(0.0)


# ==========================================================================


def test_ties_are_scored_as_half_not_in_our_favour():
    y = np.array([0, 1, 0, 1])
    assert metrics.roc_auc(y, np.ones(4)) == pytest.approx(0.5)


def test_auroc_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    s = rng.random(500) + 0.4 * y
    assert metrics.roc_auc(y, s) == pytest.approx(
        sklearn_metrics.roc_auc_score(y, s), abs=1e-9
    )


def test_roc_curve_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    s = rng.random(300) + 0.5 * y
    fpr, tpr = metrics.roc_curve(y, s)
    # sklearn drops collinear points by default; we keep every threshold so the
    # curve can be interpolated at exactly 1% FPR.
    sk_fpr, sk_tpr, _ = sklearn_metrics.roc_curve(y, s, drop_intermediate=False)
    assert np.allclose(fpr, sk_fpr) and np.allclose(tpr, sk_tpr)


def test_average_precision_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 400)
    s = rng.random(400) + 0.6 * y
    assert metrics.average_precision(y, s) == pytest.approx(
        sklearn_metrics.average_precision_score(y, s), abs=0.02
    )


def test_tpr_at_fpr_is_interpolated_not_snapped():
    y = np.r_[np.zeros(1000, int), np.ones(100, int)]
    s = np.r_[np.linspace(0, 0.5, 1000), np.linspace(0.4, 1.0, 100)]
    v = metrics.tpr_at_fpr(y, s, 0.01)
    assert 0.0 <= v <= 1.0
    assert v >= metrics.tpr_at_fpr(y, s, 0.005)


def test_tpr_at_fpr_is_stricter_than_auroc():
    """A detector can have a fine AUROC and still be useless at a 1% budget."""
    rng = np.random.default_rng(7)
    y = np.r_[np.zeros(2000, int), np.ones(200, int)]
    s = np.r_[rng.normal(0.4, 0.2, 2000), rng.normal(0.6, 0.2, 200)]
    assert metrics.roc_auc(y, s) > 0.7
    assert metrics.tpr_at_fpr(y, s, 0.01) < 0.4


def test_metrics_return_nan_when_a_class_is_missing():
    y = np.zeros(50, int)
    s = np.random.default_rng(0).random(50)
    assert np.isnan(metrics.roc_auc(y, s))
    assert np.isnan(metrics.average_precision(y, s))


def test_precision_and_recall_at_k():
    y = np.r_[np.ones(10, int), np.zeros(90, int)]
    s = np.r_[np.ones(10), np.zeros(90)]
    assert metrics.precision_at_k(y, s, 10) == pytest.approx(1.0)
    assert metrics.recall_at_k(y, s, 10) == pytest.approx(1.0)
    assert metrics.precision_at_k(y, s, 100) == pytest.approx(0.1)


# -- calibration ------------------------------------------------------------


def test_ece_is_zero_for_a_perfect_forecaster():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert metrics.ece(y, p, bins=10) < 0.02


def test_ece_is_large_for_an_overconfident_forecaster():
    y = np.r_[np.zeros(500, int), np.ones(500, int)]
    p = np.r_[np.full(500, 0.9), np.full(500, 0.9)]
    assert metrics.ece(y, p) > 0.3


def test_isotonic_calibration_reduces_ece():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 3000)
    raw = np.clip(0.5 + 0.25 * y + rng.normal(0, 0.15, 3000), 0, 1)
    squashed = raw**3  # badly scaled but correctly ordered
    fit, test = slice(0, 1500), slice(1500, None)
    c = IsotonicCalibrator().fit(squashed[fit], y[fit])
    before = metrics.ece(y[test], squashed[test])
    after = metrics.ece(y[test], c.transform(squashed[test]))
    assert after < before
    assert after < 0.08


def test_calibration_never_reverses_the_ranking():
    """Isotonic calibration is monotone, so it may merge scores into ties but
    must never say that a less suspicious image is more suspicious."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 800)
    s = np.clip(rng.random(800) + 0.3 * y, 0, 1)
    out = IsotonicCalibrator().fit(s, y).transform(s)
    order = np.argsort(s)
    assert np.all(np.diff(out[order]) >= -1e-12)


def test_unfitted_calibrator_passes_scores_through():
    c = IsotonicCalibrator().fit([0.1, 0.2], [0, 1])
    assert not c.is_fitted
    assert c.transform([0.3])[0] == pytest.approx(0.3)


# -- splits -----------------------------------------------------------------


def test_three_way_split_is_disjoint_and_complete():
    ids = [f"s{i}" for i in range(1000)]
    s = three_way_split(ids, seed=0)
    assert set(s.fit) | set(s.calibrate) | set(s.test) == set(ids)
    assert not set(s.fit) & set(s.calibrate)
    assert not set(s.fit) & set(s.test)
    assert not set(s.calibrate) & set(s.test)


def test_split_is_deterministic():
    ids = [f"s{i}" for i in range(500)]
    assert three_way_split(ids, seed=3).test == three_way_split(ids, seed=3).test


def test_stratified_split_keeps_the_poison_rate_in_every_part():
    ids = [f"s{i}" for i in range(1000)]
    labels = [1 if i < 100 else 0 for i in range(1000)]
    # stratify is aligned to sorted(ids)
    order = sorted(range(1000), key=lambda i: f"s{i}")
    s = three_way_split(ids, stratify=[labels[i] for i in order], seed=0)
    pos = {f"s{i}" for i in range(100)}
    for part in (s.fit, s.calibrate, s.test):
        rate = len(set(part) & pos) / len(part)
        assert 0.05 < rate < 0.15


def test_leakage_fails_loudly():
    bad = Splits(fit=["a", "b"], calibrate=["b"], test=["c"])
    with pytest.raises(SplitLeakage, match="appear in both"):
        bad.check()


def test_calibrate_findings_only_fits_on_the_calibration_split():
    rng = np.random.default_rng(0)
    ids = [f"s{i}" for i in range(300)]
    truth = {sid: int(i < 60) for i, sid in enumerate(ids)}
    findings = [
        Finding(
            asset_ref=sid,
            asset_type="sample",
            attack_class="ood_insertion",
            detector_id="ood",
            access_tier=0,
            raw_score=float(np.clip(0.3 + 0.4 * truth[sid] + rng.normal(0, 0.1), 0, 1)),
            severity="low",
            disposition="accept",
            reason=f"This image sits {3.2:.1f} times further from its class than usual.",
        )
        for sid in ids
    ]
    splits = three_way_split(ids, seed=0)
    out, cals = calibrate_findings(findings, truth, splits)
    assert cals["ood"].is_fitted
    assert cals["ood"].n_fit == len(splits.calibrate)
    assert all(f.calibrated_score is not None for f in out)


# -- bootstrap --------------------------------------------------------------


def test_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = rng.random(400) + 0.5 * y
    est = metrics.bootstrap_ci(metrics.roc_auc, y, s, n=300, seed=0)
    assert est.lo <= est.value <= est.hi
    assert est.n == 400


def test_bootstrap_interval_narrows_with_more_data():
    rng = np.random.default_rng(0)

    def width(n):
        y = rng.integers(0, 2, n)
        s = rng.random(n) + 0.4 * y
        e = metrics.bootstrap_ci(metrics.roc_auc, y, s, n=300, seed=1)
        return e.hi - e.lo

    assert width(3000) < width(150)


def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    a = metrics.bootstrap_ci(metrics.roc_auc, y, s, n=200, seed=5)
    b = metrics.bootstrap_ci(metrics.roc_auc, y, s, n=200, seed=5)
    assert (a.value, a.lo, a.hi) == (b.value, b.lo, b.hi)


def test_estimate_prints_with_its_interval():
    assert str(metrics.Estimate(0.94, 0.91, 0.96, 100)) == "0.940 [0.910, 0.960]"


# -- contributor risk -------------------------------------------------------


def test_small_sample_gives_a_wide_interval():
    small = metrics.beta_binomial_risk(1, 3)
    large = metrics.beta_binomial_risk(100, 300)
    assert small.rate == pytest.approx(large.rate, abs=0.01)
    assert (small.hi - small.lo) > 4 * (large.hi - large.lo)


def test_a_clearly_bad_contributor_is_flagged_with_confidence():
    r = metrics.beta_binomial_risk(214, 1200, threshold=0.05)
    assert r.p_above_threshold > 0.99
    assert r.lo > 0.15


def test_a_clean_contributor_is_not_accused():
    r = metrics.beta_binomial_risk(3, 400, threshold=0.05)
    assert r.p_above_threshold < 0.05


# -- verdicts ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(0.95, "strong"), (0.80, "strong"), (0.7, "good"), (0.4, "partial"),
     (0.1, "unsupported"), (float("nan"), "not measured")],
)
def test_verdict_bands(value, expected):
    assert metrics.verdict(value) == expected


# -- tables -----------------------------------------------------------------


def test_table_writes_csv_and_markdown(tmp_path):
    t = Table(name="t", title="T", columns=["a", "b"])
    t.add(a=1, b="x")
    paths = t.write(tmp_path)
    assert paths["csv"].read_text().startswith("a,b")
    assert "| a | b |" in paths["markdown"].read_text()


def test_table_rejects_unknown_columns():
    t = Table(name="t", title="T", columns=["a"])
    with pytest.raises(KeyError):
        t.add(a=1, zzz=2)


# -- end to end -------------------------------------------------------------


def _fake_run(tmp_path, n=400, n_poisoned=40, separation=0.55, seed=0):
    rng = np.random.default_rng(seed)
    samples, findings = [], []
    for i in range(n):
        sid = f"class_a/img_{i:04d}.png"
        poisoned = i < n_poisoned
        samples.append(
            {
                "sample_id": sid,
                "is_poisoned": int(poisoned),
                "attack_class": "ood_insertion" if poisoned else "clean",
                "true_label": "foreign" if poisoned else "class_a",
                "stored_label": "class_a",
                "contributor_id": f"C{4 if poisoned and i % 5 else i % 5}",
            }
        )
        score = float(np.clip(rng.normal(0.25 + separation * poisoned, 0.12), 0, 1))
        findings.append(
            Finding(
                asset_ref=sid,
                asset_type="sample",
                attack_class="ood_insertion",
                detector_id="ood",
                access_tier=0,
                raw_score=score,
                severity="high" if score > 0.7 else "low",
                disposition="quarantine" if score > 0.6 else "accept",
                reason=f"This image sits {1 + 10 * score:.1f} times further away from "
                f"the other pictures in its folder than a typical one does.",
            )
        )
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    (truth_dir / "ground_truth.json").write_text(
        json.dumps({"seed": seed, "attack_config": {}, "samples": samples})
    )
    findings_path = tmp_path / "findings.jsonl"
    write_findings(findings, findings_path)
    return truth_dir, findings_path


def test_score_all_produces_four_tables_and_seven_plots(tmp_path):
    truth_dir, findings_path = _fake_run(tmp_path)
    result = score_all(
        truth=truth_dir, findings=findings_path, out=tmp_path / "out", n_bootstrap=60
    )
    assert len(result.tables) == 4
    assert len(result.plots) == 7
    for p in result.plots.values():
        assert Path(p).exists() and Path(p).stat().st_size > 5000
        assert Path(str(p).replace(".png", ".svg")).exists()
    for t in result.tables:
        assert (tmp_path / "out" / "tables" / f"{t.name}.csv").exists()
        assert (tmp_path / "out" / "tables" / f"{t.name}.md").exists()
    md = (tmp_path / "out" / "RESULTS.md").read_text()
    assert "Table 1" in md and "Table 4" in md


def test_a_good_detector_scores_well_end_to_end(tmp_path):
    truth_dir, findings_path = _fake_run(tmp_path, separation=0.6)
    result = score_all(truth=truth_dir, findings=findings_path, out=tmp_path / "out",
                       n_bootstrap=60, make_plots=False)
    t1 = result.tables[0]
    assert len(t1) > 0
    row = t1.rows[0]
    auroc = float(row["AUROC [95% CI]"].split()[0])
    assert auroc > 0.9
    assert row["verdict"] in {"strong", "good"}


def test_a_useless_detector_is_reported_as_useless(tmp_path):
    truth_dir, findings_path = _fake_run(tmp_path, separation=0.0)
    result = score_all(truth=truth_dir, findings=findings_path, out=tmp_path / "out",
                       n_bootstrap=60, make_plots=False)
    row = result.tables[0].rows[0]
    assert float(row["AUROC [95% CI]"].split()[0]) < 0.65
    assert row["verdict"] in {"unsupported", "partial"}


def test_contributor_table_marks_whether_it_matched_truth(tmp_path):
    truth_dir, findings_path = _fake_run(tmp_path, separation=0.6)
    result = score_all(truth=truth_dir, findings=findings_path, out=tmp_path / "out",
                       n_bootstrap=60, make_plots=False)
    t2 = result.tables[1]
    assert len(t2) == 5
    assert all(r["matches_truth"] in {"yes", "NO"} for r in t2.rows)


def test_unmatched_findings_are_reported_loudly(tmp_path):
    truth_dir, findings_path = _fake_run(tmp_path)
    extra = read_findings(findings_path)
    from dataclasses import replace

    write_findings(
        extra + [replace(extra[0], asset_ref="a_path_that_is_not_in_the_key.png")],
        findings_path,
    )
    result = score_all(truth=truth_dir, findings=findings_path, out=tmp_path / "out",
                       n_bootstrap=30, make_plots=False)
    assert result.joined.unmatched
    assert "Warning" in (tmp_path / "out" / "RESULTS.md").read_text()


def test_findings_round_trip_through_jsonl(tmp_path):
    _, findings_path = _fake_run(tmp_path)
    a = read_findings(findings_path)
    write_findings(a, tmp_path / "again.jsonl")
    assert read_findings(tmp_path / "again.jsonl") == a


def test_scoring_is_deterministic(tmp_path):
    truth_dir, findings_path = _fake_run(tmp_path)
    kw = dict(truth=truth_dir, findings=findings_path, n_bootstrap=50, make_plots=False)
    score_all(out=tmp_path / "a", **kw)
    score_all(out=tmp_path / "b", **kw)
    assert (tmp_path / "a" / "RESULTS.md").read_text() == (
        tmp_path / "b" / "RESULTS.md"
    ).read_text()


def test_silence_counts_as_a_prediction_of_clean(tmp_path):
    """A detector that never mentions a poisoned sample must not be rewarded."""
    truth_dir, findings_path = _fake_run(tmp_path, n=200, n_poisoned=20)
    kept = [f for f in read_findings(findings_path) if not f.asset_ref.endswith("0000.png")]
    write_findings(kept, findings_path)
    result = score_all(truth=truth_dir, findings=findings_path, out=tmp_path / "out",
                       n_bootstrap=30, make_plots=False)
    y, s = __import__(
        "cvassure.score.evaluate", fromlist=["_scores_for"]
    )._scores_for(result.joined)
    assert y.size == 200
