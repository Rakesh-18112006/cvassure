"""Metrics, with an interval on every headline number.

"0.94 [0.91, 0.96]" is a far stronger claim than a bare "0.94", because it
says how much of the number is real and how much is the size of the test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with a bootstrap confidence interval."""

    value: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        if not np.isfinite(self.value):
            return "n/a"
        if not np.isfinite(self.lo):
            return f"{self.value:.3f}"
        return f"{self.value:.3f} [{self.lo:.3f}, {self.hi:.3f}]"

    def to_dict(self) -> dict[str, float | int]:
        return {"value": self.value, "lo": self.lo, "hi": self.hi, "n": self.n}


def _clean(y_true: Sequence, y_score: Sequence) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y_true has {y.shape[0]} entries but y_score has {s.shape[0]}")
    ok = np.isfinite(s)
    return y[ok], s[ok]


# --------------------------------------------------------------------------
# Ranking metrics
# --------------------------------------------------------------------------


def roc_curve(y_true: Sequence, y_score: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) at every distinct threshold, ties handled correctly."""
    y, s = _clean(y_true, y_score)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y == 1)[idx]
    fps = np.cumsum(y == 0)[idx]
    return np.r_[0.0, fps / n_neg], np.r_[0.0, tps / n_pos]


def roc_auc(y_true: Sequence, y_score: Sequence) -> float:
    """Probability that a random poisoned sample outranks a random clean one.

    Computed from ranks, so ties count as half a point rather than being
    silently resolved in our favour.
    """
    y, s = _clean(y_true, y_score)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def tpr_at_fpr(y_true: Sequence, y_score: Sequence, max_fpr: float = 0.01) -> float:
    """The number that matters operationally.

    "If I will tolerate one false alarm per hundred clean images, what
    fraction of the real poison do I still catch?" Interpolated on the ROC
    curve, so it does not depend on where the thresholds happen to fall.
    """
    fpr, tpr = roc_curve(y_true, y_score)
    if fpr.size < 2:
        return float("nan")
    if max_fpr >= fpr[-1]:
        return float(tpr[-1])
    return float(np.interp(max_fpr, fpr, tpr))


def average_precision(y_true: Sequence, y_score: Sequence) -> float:
    y, s = _clean(y_true, y_score)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y).sum() / n_pos)


def precision_at_k(y_true: Sequence, y_score: Sequence, k: int = 100) -> float:
    """Of the k images we would put in front of an analyst first, how many are
    actually poisoned?"""
    y, s = _clean(y_true, y_score)
    k = min(k, y.size)
    if k == 0:
        return float("nan")
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].mean())


def recall_at_k(y_true: Sequence, y_score: Sequence, k: int = 100) -> float:
    y, s = _clean(y_true, y_score)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    k = min(k, y.size)
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].sum() / n_pos)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def ece(y_true: Sequence, y_prob: Sequence, bins: int = 10) -> float:
    """Expected Calibration Error.

    Like a weather forecaster: if the system says 0.9 a hundred times, about
    ninety of those should really be poisoned. Lower is better; under 0.05 is
    good.
    """
    y, p = _clean(y_true, y_prob)
    if y.size == 0:
        return float("nan")
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += (m.sum() / y.size) * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability_bins(
    y_true: Sequence, y_prob: Sequence, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bin centre, observed rate, count) — the reliability diagram's data."""
    y, p = _clean(y_true, y_prob)
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    centres, observed, counts = [], [], []
    for b in range(bins):
        m = idx == b
        centres.append((edges[b] + edges[b + 1]) / 2 if not m.any() else float(p[m].mean()))
        observed.append(float(y[m].mean()) if m.any() else np.nan)
        counts.append(int(m.sum()))
    return np.array(centres), np.array(observed), np.array(counts)


def brier(y_true: Sequence, y_prob: Sequence) -> float:
    y, p = _clean(y_true, y_prob)
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


def bootstrap_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence,
    y_score: Sequence,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Resample the test set to see how much of the number is really there.

    Stratified by class, so a resample can never accidentally contain no
    poisoned samples and produce a meaningless NaN.
    """
    y, s = _clean(y_true, y_score)
    point = float(fn(y, s))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if y.size == 0 or pos.size == 0 or neg.size == 0 or n <= 0:
        return Estimate(point, float("nan"), float("nan"), int(y.size))

    rng = np.random.default_rng(seed)
    draws = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = np.concatenate(
            [
                rng.choice(pos, size=pos.size, replace=True),
                rng.choice(neg, size=neg.size, replace=True),
            ]
        )
        try:
            draws[i] = fn(y[idx], s[idx])
        except Exception:
            draws[i] = np.nan
    draws = draws[np.isfinite(draws)]
    if draws.size < 10:
        return Estimate(point, float("nan"), float("nan"), int(y.size))
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point, float(lo), float(hi), int(y.size))


# --------------------------------------------------------------------------
# Contributor risk: Beta-Binomial
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContributorRisk:
    contributor_id: str
    n: int
    n_flagged: int
    rate: float
    lo: float
    hi: float
    p_above_threshold: float
    threshold: float

    def to_dict(self) -> dict[str, float | int | str]:
        from dataclasses import asdict

        return asdict(self)


def beta_binomial_risk(
    n_flagged: int, n: int, *, threshold: float = 0.05, prior=(1.0, 1.0), contributor_id: str = ""
) -> ContributorRisk:
    """Posterior over a contributor's true poison rate.

    A uniform Beta(1,1) prior keeps this honest: a contributor who sent three
    images and had one flagged gets a very wide interval rather than a 33%
    accusation.
    """
    from scipy.stats import beta as beta_dist

    a = prior[0] + n_flagged
    b = prior[1] + max(0, n - n_flagged)
    lo, hi = beta_dist.ppf([0.025, 0.975], a, b)
    return ContributorRisk(
        contributor_id=contributor_id,
        n=int(n),
        n_flagged=int(n_flagged),
        rate=float(n_flagged / n) if n else float("nan"),
        lo=float(lo),
        hi=float(hi),
        p_above_threshold=float(beta_dist.sf(threshold, a, b)),
        threshold=float(threshold),
    )


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

VERDICT_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.80, "strong"),
    (0.60, "good"),
    (0.35, "partial"),
    (0.00, "unsupported"),
)


def verdict(tpr_at_1pct: float) -> str:
    """Grade a detector on the operational number, not on AUROC."""
    if not np.isfinite(tpr_at_1pct):
        return "not measured"
    for cut, name in VERDICT_THRESHOLDS:
        if tpr_at_1pct >= cut:
            return name
    return "unsupported"


__all__ = [
    "ContributorRisk",
    "Estimate",
    "average_precision",
    "beta_binomial_risk",
    "bootstrap_ci",
    "brier",
    "ece",
    "precision_at_k",
    "recall_at_k",
    "reliability_bins",
    "roc_auc",
    "roc_curve",
    "tpr_at_fpr",
    "verdict",
]
