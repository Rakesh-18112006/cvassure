"""Turning a detector's raw score into a number that means something.

PS clause 2.2.4 asks for a *calibrated* score. A raw suspicion score of 0.9
means only "higher than 0.8"; a calibrated 0.9 means "about nine out of ten
samples I score this way really are poisoned". That is the difference between
a number an analyst can act on and a number they have to learn to interpret.

The split discipline here is not decoration. Fitting the calibrator on data it
is later evaluated on produces beautiful, meaningless numbers, so the split is
enforced with an assertion rather than a convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class SplitLeakage(AssertionError):
    """Raised when the same sample appears in more than one split."""


@dataclass(frozen=True)
class Splits:
    fit: list[str]
    calibrate: list[str]
    test: list[str]

    def check(self) -> None:
        """Loud failure, as specified — silence here would poison every number
        downstream."""
        pairs = (
            ("fit", "calibrate", set(self.fit) & set(self.calibrate)),
            ("fit", "test", set(self.fit) & set(self.test)),
            ("calibrate", "test", set(self.calibrate) & set(self.test)),
        )
        for a, b, overlap in pairs:
            if overlap:
                sample = sorted(overlap)[:5]
                raise SplitLeakage(
                    f"{len(overlap)} sample(s) appear in both the {a} split and the "
                    f"{b} split, for example {sample}. Every metric computed from "
                    f"these splits would be optimistic and worthless. Fix the split "
                    f"before trusting anything downstream."
                )

    def sizes(self) -> dict[str, int]:
        return {"fit": len(self.fit), "calibrate": len(self.calibrate), "test": len(self.test)}


def three_way_split(
    sample_ids: Sequence[str],
    *,
    fractions: tuple[float, float, float] = (0.4, 0.3, 0.3),
    seed: int = 0,
    stratify: Sequence[int] | None = None,
) -> Splits:
    """Deterministic fit/calibrate/test split, stratified when labels are given."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    ids = np.asarray(sorted(sample_ids), dtype=object)
    rng = np.random.default_rng(seed)

    if stratify is not None:
        # ``stratify[i]`` describes ``sorted(sample_ids)[i]`` — the same order
        # ``ids`` is in, so the two line up without reindexing.
        strat = np.asarray(stratify)
        if strat.size != ids.size:
            raise ValueError(
                f"stratify has {strat.size} entries but there are {ids.size} sample ids"
            )
        groups = [ids[strat == v] for v in np.unique(strat)]
    else:
        groups = [ids]

    fit, cal, test = [], [], []
    for g in groups:
        g = g.copy()
        rng.shuffle(g)
        n = g.size
        n_fit = int(round(fractions[0] * n))
        n_cal = int(round(fractions[1] * n))
        fit.extend(g[:n_fit].tolist())
        cal.extend(g[n_fit : n_fit + n_cal].tolist())
        test.extend(g[n_fit + n_cal :].tolist())

    s = Splits(fit=sorted(fit), calibrate=sorted(cal), test=sorted(test))
    s.check()
    return s


class IsotonicCalibrator:
    """Monotone mapping from raw score to probability of being poisoned.

    Isotonic rather than Platt scaling because detector scores are rarely
    shaped like a logistic curve — a Mahalanobis-style distance and a
    duplicate-cluster size have wildly different shapes, and isotonic only
    assumes "higher score means more suspicious", which is the one thing every
    detector in this system guarantees.
    """

    def __init__(self, out_of_range: str = "clip"):
        self._model = None
        self.out_of_range = out_of_range
        self.n_fit = 0

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        ok = np.isfinite(s) & np.isfinite(y)
        s, y = s[ok], y[ok]
        if s.size < 8 or len(np.unique(y)) < 2:
            # Not enough calibration data to say anything honest. Leaving the
            # calibrator unfitted means scores pass through unchanged and the
            # report says so, which is better than a made-up curve.
            self._model = None
            self.n_fit = int(s.size)
            return self
        self._model = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds=self.out_of_range
        ).fit(s, y)
        self.n_fit = int(s.size)
        return self

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def transform(self, scores: Sequence[float]) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64).ravel()
        if self._model is None:
            return np.clip(s, 0.0, 1.0)
        return np.clip(self._model.predict(s), 0.0, 1.0)

    def __call__(self, scores: Sequence[float]) -> np.ndarray:
        return self.transform(scores)


def calibrate_findings(
    findings: Iterable,
    truth: dict[str, int],
    splits: Splits,
    *,
    per_detector: bool = True,
) -> tuple[list, dict[str, IsotonicCalibrator]]:
    """Fit one calibrator per detector on the calibration split, then attach a
    calibrated score to every finding.

    Returns the findings with ``calibrated_score`` filled in, plus the fitted
    calibrators so the report can show what they learned.
    """
    findings = list(findings)
    splits.check()
    cal_ids = set(splits.calibrate)

    def key(f) -> str:
        return f.detector_id if per_detector else "_all"

    calibrators: dict[str, IsotonicCalibrator] = {}
    for k in sorted({key(f) for f in findings}):
        subset = [
            f
            for f in findings
            if key(f) == k and f.asset_ref in cal_ids and f.asset_ref in truth
            and not f.is_unavailable
        ]
        c = IsotonicCalibrator()
        if subset:
            c.fit([f.raw_score for f in subset], [truth[f.asset_ref] for f in subset])
        calibrators[k] = c

    out = []
    for f in findings:
        if f.is_unavailable:
            out.append(f)
            continue
        c = calibrators.get(key(f))
        value = float(c.transform([f.raw_score])[0]) if c else f.raw_score
        out.append(f.with_calibration(value))
    return out, calibrators


__all__ = [
    "IsotonicCalibrator",
    "SplitLeakage",
    "Splits",
    "calibrate_findings",
    "three_way_split",
]
