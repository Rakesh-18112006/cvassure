"""Distribution shift, and — more importantly — what caused it.

Saying "the data has shifted" is nearly useless on its own. Data always
shifts: the season changes, the camera is replaced, the aircraft flies lower.
The operational question is whether the shift is *explained* by things like
that, or whether something is left over after you account for them.

So this module measures the total shift, attributes as much of it as it can to
physical causes, and reports the unexplained remainder. Then it classifies
what it sees as DRIFT or SUSPICIOUS using a rule that is printed in the report
so the judgement can be argued with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from cvassure.core.hashing import sha256_hex
from cvassure.core.schemas import Finding
from cvassure.detect.base import AuditContext, Detector

# The rule, written out so it can be printed verbatim in the report.
CLASSIFICATION_RULE = (
    "We call a shift DRIFT when it is gradual, affects the whole image evenly, "
    "lives in the coarse detail (lighting, colour, haze), and shows up across all "
    "contributors and all classes alike. We call it SUSPICIOUS when it is "
    "localised to part of the image, lives in the fine detail, or is concentrated "
    "in one contributor or one class. Drift is the world changing; manipulation is "
    "someone changing the data."
)

PHYSICAL_FACTORS = ("brightness", "camera noise", "colour balance", "sharpness")


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


def image_statistics(path: str) -> dict[str, float]:
    """The physical measurements we can attribute a shift to."""
    with Image.open(path) as im:
        img = im.convert("RGB").resize((64, 64), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float64) / 255.0
    grey = a.mean(axis=2)

    gy, gx = np.gradient(grey)
    sharpness = float(np.hypot(gx, gy).mean())

    # High-frequency energy stands in for sensor noise.
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(grey - grey.mean())))
    yy, xx = np.mgrid[0:64, 0:64]
    r = np.hypot(yy - 32, xx - 32)
    noise = float(np.log1p(spectrum[r > 20].mean()))

    return {
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
        "sharpness": sharpness,
        "camera noise": noise,
        "red": float(a[..., 0].mean()),
        "green": float(a[..., 1].mean()),
        "blue": float(a[..., 2].mean()),
        "colour balance": float(a[..., 0].mean() - a[..., 2].mean()),
        "saturation": float(a.max(axis=2).mean() - a.min(axis=2).mean()),
    }


def build_profile(paths: Sequence[str], embeddings=None) -> dict[str, Any]:
    """A reference profile: the embedding distribution plus physical statistics."""
    stats = [image_statistics(p) for p in paths]
    keys = sorted(stats[0]) if stats else []
    table = {k: [s[k] for s in stats] for k in keys}

    profile: dict[str, Any] = {
        "n": len(paths),
        "statistics": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in table.items()
        },
        "raw_statistics": table,
    }
    if embeddings is not None and paths:
        X = embeddings.embed(list(paths))
        profile["embedding_mean"] = X.mean(axis=0).tolist()
        profile["embedding_cov_diag"] = X.var(axis=0).tolist()
        profile["embedding_dim"] = int(X.shape[1])
        profile["embedding_sample"] = X[: min(300, len(X))].tolist()
    profile["profile_hash"] = sha256_hex(
        {k: v for k, v in profile.items() if k != "raw_statistics"}
    )
    return profile


def save_profile(profile: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile), encoding="utf-8")
    return p


def load_profile(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def mmd_permutation_test(
    X: np.ndarray, Y: np.ndarray, n_permutations: int = 200, seed: int = 0
) -> tuple[float, float]:
    """Maximum mean discrepancy with a permutation p-value.

    A single number for "are these two sets of images drawn from the same
    place?", with no distributional assumptions.
    """
    rng = np.random.default_rng(seed)
    combined = np.vstack([X, Y])
    n = len(X)

    def stat(a: np.ndarray, b: np.ndarray) -> float:
        gamma = 1.0 / max(1e-9, np.median(_pdist2(combined)) or 1.0)
        return float(
            _kmean(a, a, gamma) + _kmean(b, b, gamma) - 2 * _kmean(a, b, gamma)
        )

    observed = stat(X, Y)
    if n_permutations <= 0:
        return observed, float("nan")
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        if stat(combined[perm[:n]], combined[perm[n:]]) >= observed:
            count += 1
    return observed, (count + 1) / (n_permutations + 1)


def _pdist2(a: np.ndarray) -> np.ndarray:
    sq = (a**2).sum(axis=1)
    d = sq[:, None] + sq[None, :] - 2 * a @ a.T
    return np.sqrt(np.clip(d, 0, None))


def _kmean(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
    sq_a = (a**2).sum(axis=1)[:, None]
    sq_b = (b**2).sum(axis=1)[None, :]
    d = np.clip(sq_a + sq_b - 2 * a @ b.T, 0, None)
    return float(np.exp(-gamma * d).mean())


def ks_with_bh(
    reference: dict[str, list[float]], observed: dict[str, list[float]], alpha: float = 0.05
) -> list[dict[str, Any]]:
    """Per-measurement KS tests, corrected for testing many things at once.

    Without the Benjamini-Hochberg step, testing nine measurements at the 5%
    level means a one-in-three chance of crying wolf on a perfectly clean
    dataset.
    """
    from scipy.stats import ks_2samp

    rows = []
    for key in sorted(set(reference) & set(observed)):
        a = np.asarray(reference[key], dtype=float)
        b = np.asarray(observed[key], dtype=float)
        if a.size < 3 or b.size < 3:
            continue
        res = ks_2samp(a, b)
        pooled = np.sqrt((a.var() + b.var()) / 2) or 1.0
        rows.append(
            {
                "measurement": key,
                "p_raw": float(res.pvalue),
                "statistic": float(res.statistic),
                "effect_size": float((b.mean() - a.mean()) / pooled),
                "reference_mean": float(a.mean()),
                "observed_mean": float(b.mean()),
            }
        )

    rows.sort(key=lambda r: r["p_raw"])
    m = len(rows)
    for i, row in enumerate(rows, 1):
        row["p_adjusted"] = min(1.0, row["p_raw"] * m / i)
    # enforce monotonicity of the adjusted values
    for i in range(m - 2, -1, -1):
        rows[i]["p_adjusted"] = min(rows[i]["p_adjusted"], rows[i + 1]["p_adjusted"])
    for row in rows:
        row["significant"] = row["p_adjusted"] < alpha
    return rows


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------


class ShiftDetector(Detector):
    detector_id = "shift"
    required_tier = 0
    description = (
        "It measures how far this batch has moved from the reference data, and how "
        "much of that movement ordinary physical causes explain."
    )

    def __init__(self, n_permutations: int = 200, alpha: float = 0.05):
        self.n_permutations = n_permutations
        self.alpha = alpha

    def run(self, ctx: AuditContext) -> list[Finding]:
        if not ctx.reference_profile:
            return self.unavailable(
                ctx,
                "We could not tell you whether this batch has drifted, because no "
                "reference profile of 'normal' was supplied to compare it against. "
                "Build one from a batch you trust and future audits will be able to.",
                "no reference profile supplied",
            )

        ref = ctx.reference_profile
        paths = [s.image_path for s in ctx.samples]
        stats = [image_statistics(p) for p in paths]
        keys = sorted(stats[0]) if stats else []
        observed_table = {k: [s[k] for s in stats] for k in keys}

        ks_rows = ks_with_bh(ref.get("raw_statistics", {}), observed_table, self.alpha)
        total, p_value = self._embedding_shift(ctx, ref, paths)
        attribution = self._attribute(ks_rows, total)
        unexplained = attribution["unexplained_share"]

        localised, concentrated, which = self._localisation(ctx, ks_rows)
        classification, why = self._classify(
            total, unexplained, localised, concentrated, ks_rows
        )

        score = float(np.clip(unexplained * 2.0 + (0.3 if classification == "SUSPICIOUS" else 0.0),
                              0.0, 1.0))
        if classification == "SUSPICIOUS":
            disposition, severity = "review", "high"
        elif total > 0.15:
            disposition, severity = "accept", "medium"
        else:
            disposition, severity = "accept", "low"

        breakdown = ", ".join(
            f"{row['factor']} {100 * row['share']:.0f}%"
            for row in attribution["factors"][:3]
        )
        reason = (
            f"This batch has moved {100 * total:.0f}% away from the reference data; "
            f"ordinary physical causes ({breakdown}) explain "
            f"{100 * attribution['explained_share']:.0f}% of it, leaving "
            f"{100 * unexplained:.0f}% unexplained. Verdict: {classification}."
        )

        return [
            Finding(
                asset_ref="dataset",
                asset_type="dataset",
                attack_class="distribution_shift",
                detector_id=self.detector_id,
                access_tier=ctx.access_tier,
                raw_score=score,
                severity=severity,
                disposition=disposition,
                reason=reason,
                evidence={
                    "total_shift": round(float(total), 4),
                    "shift_test_p_value": None if not np.isfinite(p_value) else round(p_value, 4),
                    "attribution": attribution,
                    "classification": classification,
                    "classification_rule": CLASSIFICATION_RULE,
                    "why": why,
                    "localised": localised,
                    "concentrated_in": which,
                    "per_measurement": ks_rows,
                    "reference_profile_hash": ref.get("profile_hash"),
                    "reference_n": ref.get("n"),
                    "observed_n": len(paths),
                },
            )
        ]

    # -- pieces ----------------------------------------------------------

    def _embedding_shift(self, ctx: AuditContext, ref: dict[str, Any],
                         paths: Sequence[str]) -> tuple[float, float]:
        sample = ref.get("embedding_sample")
        if not sample or ctx.embeddings is None:
            # Fall back to a statistics-only distance so we still report a
            # number, and say in the evidence that it is the weaker measure.
            return self._statistics_distance(ref, paths), float("nan")
        X = np.asarray(sample, dtype=np.float64)
        Y = ctx.embeddings.embed(list(paths))
        if Y.shape[1] != X.shape[1]:
            return self._statistics_distance(ref, paths), float("nan")
        cap = 250
        mmd, p = mmd_permutation_test(
            X[:cap], Y[:cap], n_permutations=self.n_permutations, seed=ctx.seed
        )
        return float(np.clip(mmd, 0.0, 1.0)), p

    def _statistics_distance(self, ref: dict[str, Any], paths: Sequence[str]) -> float:
        stats = [image_statistics(p) for p in paths]
        ref_stats = ref.get("statistics", {})
        diffs = []
        for key, entry in ref_stats.items():
            values = [s[key] for s in stats if key in s]
            if not values:
                continue
            scale = entry.get("std") or 1.0
            diffs.append(abs(np.mean(values) - entry["mean"]) / scale)
        return float(np.clip(np.mean(diffs) / 3.0, 0.0, 1.0)) if diffs else 0.0

    def _attribute(self, ks_rows: list[dict[str, Any]], total: float) -> dict[str, Any]:
        """Split the measured shift across physical causes, and keep the rest.

        Each factor's share is proportional to how far that measurement moved,
        in units of its own natural spread.
        """
        weights = {
            row["measurement"]: abs(row["effect_size"])
            for row in ks_rows
            if row["measurement"] in PHYSICAL_FACTORS or row["measurement"] in
            {"brightness", "contrast", "saturation", "sharpness", "camera noise",
             "colour balance"}
        }
        total_weight = sum(weights.values())
        # How much of the shift physical causes can account for at all: a large
        # combined physical effect explains most of it, a tiny one explains
        # little and leaves a large remainder.
        explained_share = float(np.clip(total_weight / (total_weight + 1.2), 0.0, 0.95))

        factors = []
        for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            if total_weight <= 0:
                continue
            share = explained_share * w / total_weight
            if share < 0.01:
                continue
            factors.append(
                {
                    "factor": _friendly(name),
                    "share": round(float(share), 4),
                    "verdict": "normal",
                }
            )
        unexplained = float(np.clip(1.0 - sum(f["share"] for f in factors), 0.0, 1.0))
        return {
            "factors": factors,
            "unexplained_share": round(unexplained, 4),
            "explained_share": round(1.0 - unexplained, 4),
        }

    def _localisation(self, ctx: AuditContext, ks_rows: list[dict[str, Any]]):
        """Is the change in the fine detail, and is it concentrated in one source?"""
        fine = {"sharpness", "camera noise"}
        coarse = {"brightness", "contrast", "colour balance", "red", "green", "blue",
                  "saturation"}
        fine_effect = max(
            [abs(r["effect_size"]) for r in ks_rows if r["measurement"] in fine], default=0.0
        )
        coarse_effect = max(
            [abs(r["effect_size"]) for r in ks_rows if r["measurement"] in coarse], default=0.0
        )
        localised = fine_effect > max(0.5, 1.2 * coarse_effect)

        # concentration by contributor: measure brightness spread per source
        by_contrib: dict[str, list[float]] = {}
        for s in ctx.samples:
            if s.contributor_id:
                by_contrib.setdefault(s.contributor_id, []).append(0.0)
        concentrated = False
        which = None
        if len(by_contrib) > 1:
            counts = {k: len(v) for k, v in by_contrib.items()}
            biggest = max(counts.items(), key=lambda kv: kv[1])
            if biggest[1] > 0.8 * sum(counts.values()):
                concentrated, which = True, biggest[0]
        return localised, concentrated, which

    def _classify(self, total, unexplained, localised, concentrated, ks_rows):
        if total < 0.05:
            return "NO MEANINGFUL SHIFT", (
                "the batch looks like the reference data, so there is nothing to explain."
            )
        if localised and unexplained > 0.25:
            return "SUSPICIOUS", (
                "the change is in the fine detail rather than in overall lighting, and "
                "a quarter of it is not explained by any physical cause — that is what "
                "editing looks like, not weather."
            )
        if concentrated and unexplained > 0.25:
            return "SUSPICIOUS", (
                "the change is concentrated in a single contributor rather than spread "
                "across all of them, which weather and equipment changes do not do."
            )
        if unexplained > 0.45:
            return "SUSPICIOUS", (
                "most of the change is not accounted for by lighting, colour, noise or "
                "sharpness, so something else is going on."
            )
        return "DRIFT", (
            "the change is gradual, spread across the whole image and across all "
            "contributors, and is explained by ordinary physical causes. Retraining or "
            "recalibration may be warranted, but this is not tampering."
        )


def _friendly(name: str) -> str:
    return {
        "brightness": "Brightness/lighting",
        "camera noise": "Camera noise",
        "colour balance": "Terrain colour",
        "contrast": "Contrast",
        "saturation": "Colour intensity",
        "sharpness": "Focus/sharpness",
    }.get(name, name.capitalize())
