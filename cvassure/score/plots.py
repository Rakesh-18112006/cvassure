"""The seven figures, at slide quality.

Rules applied throughout: colourblind-safe palette, no reliance on colour
alone, font sizes that survive being shrunk to a thumbnail, and 300 DPI PNG
plus SVG so they can be dropped straight into a deck.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cvassure.score import metrics  # noqa: E402

# Okabe-Ito: distinguishable under every common form of colour blindness.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":"]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    return png


def _empty(fig, ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# --------------------------------------------------------------------------
# 1. ROC curves, one panel per attack class, overlaid by access tier
# --------------------------------------------------------------------------


def roc_panels(result, out_dir: Path, test_ids: set[str] | None = None) -> Path:
    from cvassure.score.evaluate import _scores_for

    joined = result.joined
    attacks = sorted(
        {r["attack_class"] for r in joined.truth_by_id.values() if r["is_poisoned"]}
    )
    tiers = sorted({f.access_tier for f in joined.findings}) or [0]
    if not attacks:
        fig, ax = plt.subplots(figsize=(6, 4))
        _empty(fig, ax, "No poisoned samples in the answer key,\nso there is no ROC to draw.")
        return _save(fig, out_dir, "fig1_roc_by_attack")

    ncols = min(3, len(attacks))
    nrows = int(np.ceil(len(attacks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)

    for i, attack in enumerate(attacks):
        ax = axes[i // ncols][i % ncols]
        for j, tier in enumerate(tiers):
            y, s = _scores_for(joined, tier=tier, restrict_to=test_ids, attack_class=attack)
            if y.size == 0 or y.sum() == 0:
                continue
            fpr, tpr = metrics.roc_curve(y, s)
            auc = metrics.roc_auc(y, s)
            ax.plot(
                fpr, tpr,
                color=PALETTE[j % len(PALETTE)],
                linestyle=LINESTYLES[j % len(LINESTYLES)],
                linewidth=2.2,
                label=f"tier {tier} (AUROC {auc:.2f})",
            )
        ax.plot([0, 1], [0, 1], color="grey", linewidth=1, linestyle=":")
        ax.axvline(0.01, color="#444444", linewidth=1, linestyle="--")
        ax.set_title(attack.replace("_", " "))
        ax.set_xlabel("false alarms among clean images")
        ax.set_ylabel("poison caught")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", frameon=False)

    for k in range(len(attacks), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Detection performance by attack, at each level of model access", y=1.02)
    return _save(fig, out_dir, "fig1_roc_by_attack")


# --------------------------------------------------------------------------
# 2. Reliability diagram, before vs after calibration
# --------------------------------------------------------------------------


def reliability(result, out_dir: Path, joined_raw=None, test_ids=None) -> Path:
    from cvassure.score.evaluate import _scores_for

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    series = []
    if joined_raw is not None:
        y, s = _scores_for(joined_raw, restrict_to=test_ids, use_calibrated=False)
        series.append(("before calibration", y, s, PALETTE[1], MARKERS[1]))
    y, s = _scores_for(result.joined, restrict_to=test_ids, use_calibrated=True)
    series.append(("after calibration", y, s, PALETTE[0], MARKERS[0]))

    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1.5,
            label="perfectly calibrated")
    for label, yy, ss, colour, marker in series:
        if yy.size == 0:
            continue
        centres, observed, counts = metrics.reliability_bins(yy, ss, bins=10)
        keep = counts > 0
        ax.plot(centres[keep], observed[keep], marker=marker, color=colour,
                linewidth=2.0, markersize=8, label=f"{label} (ECE {metrics.ece(yy, ss):.3f})")

    ax.set_xlabel("score the system reported")
    ax.set_ylabel("fraction that really were poisoned")
    ax.set_title("Does a score of 0.9 mean 9 out of 10?")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    return _save(fig, out_dir, "fig2_reliability")


# --------------------------------------------------------------------------
# 3. Score histograms, clean vs poisoned, per detector
# --------------------------------------------------------------------------


def score_histograms(result, out_dir: Path, test_ids=None) -> Path:
    joined = result.joined
    detectors = sorted(
        {f.detector_id for f in joined.findings if f.asset_type == "sample"
         and not f.is_unavailable}
    )
    if not detectors:
        fig, ax = plt.subplots(figsize=(6, 4))
        _empty(fig, ax, "No sample-level findings to plot.")
        return _save(fig, out_dir, "fig3_score_histograms")

    ncols = min(3, len(detectors))
    nrows = int(np.ceil(len(detectors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    bins = np.linspace(0, 1, 21)

    for i, det in enumerate(detectors):
        ax = axes[i // ncols][i % ncols]
        clean, poisoned = [], []
        for f in joined.findings:
            if f.detector_id != det or f.asset_type != "sample" or f.is_unavailable:
                continue
            if test_ids is not None and f.asset_ref not in test_ids:
                continue
            row = joined.truth_by_id.get(f.asset_ref)
            if row is None:
                continue
            (poisoned if row["is_poisoned"] else clean).append(f.score)
        ax.hist(clean, bins=bins, color=PALETTE[0], alpha=0.75, label=f"clean ({len(clean)})")
        ax.hist(poisoned, bins=bins, color=PALETTE[1], alpha=0.75, hatch="//",
                label=f"poisoned ({len(poisoned)})")
        ax.set_yscale("symlog")
        ax.set_title(det)
        ax.set_xlabel("suspicion score")
        ax.set_ylabel("images")
        ax.legend(frameon=False)

    for k in range(len(detectors), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Where clean and poisoned images land, per detector", y=1.02)
    return _save(fig, out_dir, "fig3_score_histograms")


# --------------------------------------------------------------------------
# 4. Contributor risk with credible intervals
# --------------------------------------------------------------------------


def contributor_risk(result, out_dir: Path, threshold: float = 0.05) -> Path:
    from cvassure.score.evaluate import contributor_disposition

    joined = result.joined
    from collections import defaultdict

    totals: dict[str, list] = defaultdict(list)
    for row in joined.truth_by_id.values():
        if row.get("contributor_id"):
            totals[row["contributor_id"]].append(row)

    flagged: dict[str, set[str]] = defaultdict(set)
    for f in joined.findings:
        if f.asset_type == "sample" and not f.is_unavailable and f.disposition != "accept":
            row = joined.truth_by_id.get(f.asset_ref)
            if row and row.get("contributor_id"):
                flagged[row["contributor_id"]].add(f.asset_ref)

    if not totals:
        fig, ax = plt.subplots(figsize=(6, 4))
        _empty(fig, ax, "No contributor metadata was provided,\nso source-level "
                        "assessment is unavailable.")
        return _save(fig, out_dir, "fig4_contributor_risk")

    cids = sorted(totals)
    risks = [
        metrics.beta_binomial_risk(len(flagged.get(c, ())), len(totals[c]),
                                   threshold=threshold, contributor_id=c)
        for c in cids
    ]
    true_rates = [float(np.mean([r["is_poisoned"] for r in totals[c]])) for c in cids]

    fig, ax = plt.subplots(figsize=(max(6.5, 1.35 * len(cids)), 5.2))
    x = np.arange(len(cids))
    lower = [max(0, r.rate - r.lo) for r in risks]
    upper = [max(0, r.hi - r.rate) for r in risks]
    colours = [
        PALETTE[1] if contributor_disposition(r) == "quarantine"
        else PALETTE[4] if contributor_disposition(r) == "review"
        else PALETTE[0]
        for r in risks
    ]
    ax.bar(x, [r.rate for r in risks], color=colours, width=0.6)
    ax.errorbar(x, [r.rate for r in risks], yerr=[lower, upper], fmt="none",
                ecolor="#222222", capsize=5, linewidth=1.6)
    ax.scatter(x, true_rates, marker="*", s=220, color="#000000", zorder=5,
               label="true poison rate")

    for i, r in enumerate(true_rates):
        if r > threshold:
            ax.annotate("known bad actor", (i, max(r, risks[i].hi) + 0.03),
                        ha="center", fontsize=10, fontweight="bold")

    ax.axhline(threshold, color="grey", linestyle="--", linewidth=1.2,
               label=f"{threshold:.0%} action threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(cids)
    ax.set_xlabel("contributor")
    ax.set_ylabel("share of images flagged")
    ax.set_ylim(0, max(1.0, max([r.hi for r in risks] + true_rates) * 1.25))
    ax.set_title("Which source is the problem? (bars: what we found, stars: the truth)")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, out_dir, "fig4_contributor_risk")


# --------------------------------------------------------------------------
# 5. Detection rate vs poison rate — the most important plot
# --------------------------------------------------------------------------


def sweep(sweep_rows: Sequence[dict[str, Any]], out_dir: Path) -> Path:
    """How weak an attack can we still catch?

    ``sweep_rows``: {attack_class, poison_rate, tpr_at_1pct, lo, hi}.
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    if not sweep_rows:
        _empty(fig, ax, "No sweep data yet — run experiments/run_all.py to fill "
                        "this in across poison rates.")
        return _save(fig, out_dir, "fig5_sweep")

    from collections import defaultdict

    by_attack: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sweep_rows:
        by_attack[row["attack_class"]].append(row)

    for i, attack in enumerate(sorted(by_attack)):
        rows = sorted(by_attack[attack], key=lambda r: r["poison_rate"])
        xs = [100 * r["poison_rate"] for r in rows]
        ys = [r["tpr_at_1pct"] for r in rows]
        ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], color=PALETTE[i % len(PALETTE)],
                linestyle=LINESTYLES[i % len(LINESTYLES)], linewidth=2.2, markersize=8,
                label=attack.replace("_", " "))
        if all("lo" in r and np.isfinite(r.get("lo", np.nan)) for r in rows):
            ax.fill_between(xs, [r["lo"] for r in rows], [r["hi"] for r in rows],
                            color=PALETTE[i % len(PALETTE)], alpha=0.15)

    ax.axhline(0.8, color="#009E73", linestyle="--", linewidth=1.2)
    ax.text(ax.get_xlim()[1], 0.805, "strong", ha="right", fontsize=10, color="#009E73")
    ax.axhline(0.35, color="#D55E00", linestyle="--", linewidth=1.2)
    ax.text(ax.get_xlim()[1], 0.355, "unsupported below here", ha="right", fontsize=10,
            color="#D55E00")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("share of the dataset that is poisoned (%)")
    ax.set_ylabel("poison caught at a 1% false-alarm budget")
    ax.set_ylim(0, 1.02)
    ax.set_title("How weak an attack can we still catch?")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, out_dir, "fig5_sweep")


# --------------------------------------------------------------------------
# 6. Confusion matrix of dispositions vs truth
# --------------------------------------------------------------------------


def disposition_confusion(result, out_dir: Path, test_ids=None) -> Path:
    from cvassure.score.evaluate import disposition_confusion as compute

    m = compute(result.joined, test_ids)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks([0, 1, 2], ["accept", "review", "quarantine"])
    ax.set_yticks([0, 1], ["really clean", "really poisoned"])
    total = max(1, m.sum())
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]}\n{100 * m[i, j] / total:.1f}%", ha="center",
                    va="center", fontsize=12,
                    color="white" if m[i, j] > m.max() * 0.55 else "black")
    ax.set_title("What we did with each image, against what it really was")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="images")
    return _save(fig, out_dir, "fig6_disposition_confusion")


# --------------------------------------------------------------------------
# 7. Runtime scaling
# --------------------------------------------------------------------------


def runtime_scaling(timings: Sequence[dict[str, Any]], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    if not timings:
        _empty(fig, ax, "No timing data yet — the audit records this as it runs.")
        return _save(fig, out_dir, "fig7_runtime")

    from collections import defaultdict

    by_module: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in timings:
        by_module[row["module"]].append((int(row["n_samples"]), float(row["seconds"])))

    for i, module in enumerate(sorted(by_module)):
        pts = sorted(by_module[module])
        xs = [max(1, p[0]) for p in pts]
        ys = [max(1e-4, p[1]) for p in pts]
        ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], color=PALETTE[i % len(PALETTE)],
                linestyle=LINESTYLES[i % len(LINESTYLES)], linewidth=2.0, label=module)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("images audited")
    ax.set_ylabel("seconds")
    ax.set_title("How the audit scales")
    ax.legend(frameon=False, ncol=2)
    return _save(fig, out_dir, "fig7_runtime")


# --------------------------------------------------------------------------


def render_all(result, *, joined_raw=None, test_ids=None,
               sweep_rows: Sequence[dict[str, Any]] | None = None,
               timings: Sequence[dict[str, Any]] | None = None) -> dict[str, Path]:
    out_dir = Path(result.out_dir) / "plots"
    timings = timings or []
    sweep_path = Path(result.out_dir) / "sweep.json"
    if sweep_rows is None and sweep_path.exists():
        import json

        sweep_rows = json.loads(sweep_path.read_text(encoding="utf-8"))

    return {
        "fig1_roc_by_attack": roc_panels(result, out_dir, test_ids),
        "fig2_reliability": reliability(result, out_dir, joined_raw, test_ids),
        "fig3_score_histograms": score_histograms(result, out_dir, test_ids),
        "fig4_contributor_risk": contributor_risk(result, out_dir),
        "fig5_sweep": sweep(sweep_rows or [], out_dir),
        "fig6_disposition_confusion": disposition_confusion(result, out_dir, test_ids),
        "fig7_runtime": runtime_scaling(timings, out_dir),
    }
