"""Turn the sweep into RESULTS.md and COVERAGE.md.

Both files are generated from measured numbers. Nothing in either is written
by hand, which is the only way a coverage statement stays true.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if "error" not in row:
                    rows.append(row)
    return rows


def _mean_pm(values: list[float]) -> str:
    """Mean across seeds, with the spread that gives the ± in the tables."""
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return "n/a"
    if v.size == 1:
        return f"{v[0]:.3f}"
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def table1(rows: list[dict[str, Any]]):
    from cvassure.score.tables import Table

    t = Table(
        name="table1_detection",
        title="Table 1 — Detection by attack class, poison rate and access tier",
        columns=["attack", "poison rate", "tier", "n", "poisoned", "AUROC",
                 "TPR@1%FPR", "ECE", "false alarms", "verdict"],
        notes=[
            "± is the spread across independent seeds, so a number without one was "
            "measured once and should be trusted less.",
            "TPR@1%FPR is the operational number: the share of poison caught when "
            "only one clean image in a hundred may be falsely flagged.",
        ],
    )
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["attack"], r["poison_rate"], r["access_tier"])].append(r)

    for key in sorted(grouped):
        group = grouped[key]
        attack, rate, tier = key
        tprs = [g["tpr_at_1pct"] for g in group]
        mean_tpr = float(np.nanmean(tprs)) if tprs else float("nan")
        from cvassure.score.metrics import verdict as grade

        t.add(
            attack=attack,
            **{
                "poison rate": f"{100 * rate:.0f}%",
                "TPR@1%FPR": _mean_pm(tprs),
                "false alarms": _mean_pm([g.get("false_alarm_rate") for g in group]),
            },
            tier=tier,
            n=int(np.mean([g["n"] for g in group])),
            poisoned=int(np.mean([g["n_poisoned"] for g in group])),
            AUROC=_mean_pm([g["auroc"] for g in group]),
            ECE=_mean_pm([g["ece"] for g in group]),
            verdict=grade(mean_tpr),
        )
    return t


def table4(rows: list[dict[str, Any]]):
    from cvassure.score.tables import Table

    t = Table(
        name="table4_runtime",
        title="Table 4 — Runtime and memory per 1000 images",
        columns=["module", "runs", "seconds per 1000 images", "peak RAM (MB)"],
        notes=["Measured on the machine named at the top of this file."],
    )
    per: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        n = max(1, r.get("n", 1))
        for module, seconds in (r.get("per_detector_seconds") or {}).items():
            per[module].append(1000 * seconds / n)
    for module in sorted(per):
        t.add(
            module=module,
            runs=len(per[module]),
            **{
                "seconds per 1000 images": f"{np.mean(per[module]):.2f}",
                "peak RAM (MB)": f"{np.mean([r.get('peak_ram_mb', 0) for r in rows]):.1f}",
            },
        )
    return t


def table3(tamper_rows: list[dict[str, Any]]):
    """Tamper detection. Delegates to the one implementation in the package so
    the results run and ``cvassure score all`` can never disagree about it."""
    from cvassure.score.evaluate import table3_tamper

    return table3_tamper(tamper_rows)


def sweep_rows_for_plot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["attack"], r["poison_rate"])].append(r)
    out = []
    for (attack, rate), group in sorted(grouped.items()):
        tprs = [g["tpr_at_1pct"] for g in group if np.isfinite(g["tpr_at_1pct"])]
        if not tprs:
            continue
        out.append(
            {
                "attack_class": attack,
                "poison_rate": rate,
                "tpr_at_1pct": float(np.mean(tprs)),
                "lo": float(np.mean([g["tpr_lo"] for g in group])),
                "hi": float(np.mean([g["tpr_hi"] for g in group])),
            }
        )
    return out


def coverage_md(rows: list[dict[str, Any]],
                tamper_rows: list[dict[str, Any]] | None = None) -> str:
    """COVERAGE.md, generated from the measured numbers only."""
    from cvassure.report.build import ATTACK_LABELS
    from cvassure.score.metrics import verdict as grade

    # Graded on the *typical* cell, not the best one. Quoting the best result
    # across every poison rate, tier and seed is how a coverage table ends up
    # claiming something the system does not reliably do: one lucky cell at 1%
    # poison turns a detector that usually catches a fifth of an attack into a
    # row marked "good". The median is what an operator would actually see, so
    # the median is what gets graded; the range is shown next to it so a weak
    # detector's occasional good run is still visible.
    per_attack: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if np.isfinite(r["tpr_at_1pct"]):
            per_attack[r["attack"]].append(r)

    summary: dict[str, dict[str, Any]] = {}
    for attack, group in per_attack.items():
        tprs = [g["tpr_at_1pct"] for g in group]
        best_cell = max(group, key=lambda g: g["tpr_at_1pct"])
        summary[attack] = {
            "median": float(np.median(tprs)),
            "min": float(np.min(tprs)),
            "max": float(np.max(tprs)),
            "n_cells": len(group),
            "best_rate": best_cell["poison_rate"],
            "false_alarms": float(
                np.median([g.get("false_alarm_rate", np.nan) for g in group])
            ),
        }

    lines = [
        "# Coverage",
        "",
        "Generated from the measured results in `sweep_raw.jsonl`. Nothing here is "
        "written by hand.",
        "",
        "Graded on the **median** across every poison rate, access tier and seed — "
        "what an operator would typically see — not on the best cell. The range "
        "shows how much that varies.",
        "",
        "| what an attacker did | typical TPR@1%FPR | range | false alarms | cells | status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for attack in sorted(summary):
        s = summary[attack]
        lines.append(
            f"| {ATTACK_LABELS.get(attack, attack)} | {s['median']:.2f} | "
            f"{s['min']:.2f} – {s['max']:.2f} | {100 * s['false_alarms']:.1f}% | "
            f"{s['n_cells']} | {grade(s['median'])} |"
        )

    strong = [a for a, s in summary.items() if grade(s["median"]) in ("strong", "good")]
    weak = [a for a, s in summary.items() if grade(s["median"]) == "partial"]
    bad = [a for a, s in summary.items() if grade(s["median"]) == "unsupported"]

    # Tamper detection is not statistical, so it gets its own rows rather than
    # being averaged in with the ranking metrics.
    if tamper_rows:
        by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in tamper_rows:
            if r["attack_class"] != "clean":
                by_kind[r["attack_class"]].append(r)
        for kind in sorted(by_kind):
            group = by_kind[kind]
            rate = sum(1 for g in group if g.get("detected")) / max(1, len(group))
            lines.append(
                f"| {ATTACK_LABELS.get(kind, kind)} | {rate:.2f} | proof, not an "
                f"estimate | 0.0% | {len(group)} | "
                f"{'strong' if rate >= 0.999 else 'BUG — must be 100%'} |"
            )

    lines += ["", "## In plain words", ""]
    if strong:
        lines.append(
            "**We detect these reliably:** "
            + "; ".join(ATTACK_LABELS.get(a, a) for a in sorted(strong)) + "."
        )
    if weak:
        lines.append(
            "**We catch these only some of the time** — treat a clean result here as "
            "weak evidence: "
            + "; ".join(ATTACK_LABELS.get(a, a) for a in sorted(weak)) + "."
        )
    if bad:
        lines.append(
            "**We do not detect these.** Saying so is the point of this table: "
            + "; ".join(ATTACK_LABELS.get(a, a) for a in sorted(bad)) + "."
        )
    lines += [
        "",
        "## What is not covered at all",
        "",
        "- An attacker who knows exactly which checks these are can design something "
        "that avoids all of them. A clean report means we found nothing, not that "
        "there is nothing.",
        "- Detecting a backdoor by reconstructing its trigger needs to see inside the "
        "model as it runs. Given only the model's answers, we can prove the file was "
        "swapped but not that it was poisoned.",
        "- Without contributor metadata there is no source-level assessment, only a "
        "list of suspect images.",
    ]
    return "\n".join(lines) + "\n"


def _render_remaining_figures(rows: list[dict[str, Any]], out_dir: Path) -> str | None:
    """Produce the five figures, and the one table, that need findings joined
    to their answer key.

    Uses one representative cell — the mixed attack at a middling poison rate,
    which is what a contaminated intake actually looks like — rather than
    averaging incomparable runs together. Silently skipped if the sweep was run
    by an older version that did not keep its findings.

    Returns Table 2 as Markdown so the caller can place it in RESULTS.md in
    numerical order. Contributor risk cannot be averaged across the sweep the
    way tables 1 and 4 can — contributor C2 in one cell is not the same actor as
    C2 in the next — so it is reported from this one cell and says so.
    """
    candidates = [
        r for r in rows
        if r.get("cell_dir")
        and Path(r["cell_dir"], "findings.jsonl").exists()
        and Path(r["cell_dir"], "truth", "ground_truth.json").exists()
    ]
    if not candidates:
        print("  (no per-cell findings on disk — skipping the joined figures; "
              "re-run experiments/run_all.py to produce them)")
        return None

    def rank(r):
        return (abs(r["poison_rate"] - 0.05), r["attack"] != "ood_insertion", r["seed"])

    pick = sorted(candidates, key=rank)[0]
    cell = Path(pick["cell_dir"])
    print(f"  representative cell for the joined figures: {pick['attack']} at "
          f"{100 * pick['poison_rate']:.0f}% poison, seed {pick['seed']}")

    from cvassure.score.evaluate import score_all

    result = score_all(
        truth=cell / "truth",
        findings=cell / "findings.jsonl",
        out=out_dir / "representative",
        n_bootstrap=300,
        seed=pick["seed"],
    )
    import shutil

    # Copy only the five that need the join. Figures 5 and 7 are properties of
    # the whole sweep, and the representative cell renders them from a single
    # run — copying those over would replace the most important figure in the
    # project with an empty placeholder.
    joined_figures = (
        "fig1_roc_by_attack",
        "fig2_reliability",
        "fig3_score_histograms",
        "fig4_contributor_risk",
        "fig6_disposition_confusion",
    )
    for src in sorted((out_dir / "representative" / "plots").glob("*")):
        if src.stem in joined_figures:
            shutil.copy2(src, out_dir / "plots" / src.name)

    # Table 2 needs the same join, so it is produced here and promoted next to
    # the sweep-wide tables. Without this it lands only under representative/
    # and a reader looking for it in results/tables/ concludes it is missing.
    table2 = next((t for t in result.tables if t.name == "table2_contributors"), None)
    if table2 is not None:
        for suffix in (".md", ".csv"):
            src = out_dir / "representative" / "tables" / f"table2_contributors{suffix}"
            if src.exists():
                shutil.copy2(src, out_dir / "tables" / src.name)

    print(f"  calibration error {result.ece_before:.3f} before, "
          f"{result.ece_after:.3f} after")
    if table2 is None:
        return None
    return (
        table2.to_markdown()
        + f"\n\nMeasured on one cell — {pick['attack']} at "
        f"{100 * pick['poison_rate']:.0f}% poison, seed {pick['seed']} — because "
        "contributor identities are generated per cell and averaging them across "
        "the sweep would combine different actors under the same name.\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "results"))
    parser.add_argument("--raw", default=None)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    raw_path = Path(args.raw) if args.raw else out_dir / "sweep_raw.jsonl"
    if not raw_path.exists():
        print(f"No sweep results at {raw_path}. Run experiments/run_all.py first.")
        return 1

    rows = load_rows(raw_path)
    if not rows:
        print("The sweep produced no usable rows.")
        return 1

    tamper_path = out_dir / "tamper_raw.jsonl"
    tamper_rows = (
        [json.loads(l) for l in tamper_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if tamper_path.exists()
        else []
    )

    t1, t4 = table1(rows), table4(rows)
    tables = [t1, t4]
    t3 = table3(tamper_rows) if tamper_rows else None
    if t3 is not None:
        tables.insert(1, t3)
    for t in tables:
        t.write(out_dir / "tables")

    sweep = sweep_rows_for_plot(rows)
    (out_dir / "sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    from cvassure.report.audit import environment
    from cvassure.score import plots

    plots.sweep(sweep, out_dir / "plots")
    t2_md = _render_remaining_figures(rows, out_dir)
    plots.runtime_scaling(
        [{"module": m, "n_samples": r["n"], "seconds": s}
         for r in rows for m, s in (r.get("per_detector_seconds") or {}).items()],
        out_dir / "plots",
    )

    env = environment()
    md = [
        "# Results",
        "",
        f"{len(rows)} measured cells from `sweep_raw.jsonl`.",
        f"Machine: {env['platform']}, Python {env['python']}, "
        f"{env['cpu_count']} cores. No network access at any point.",
        "",
        "Every number below was produced by poisoning data with a seeded script that "
        "recorded exactly which images it touched, then running the detectors blind "
        "against it. The detector code cannot import the attack code; a test enforces "
        "that.",
        "",
        t1.to_markdown(),
        "",
        *([t2_md, ""] if t2_md else []),
        *([t3.to_markdown(), ""] if t3 is not None else []),
        t4.to_markdown(),
        "",
        "### Figures",
        "",
        "- `plots/fig5_sweep.png` — detection rate against poison rate. The most "
        "important figure: it answers *how weak an attack can you still catch?*",
        "- `plots/fig1_roc_by_attack.png` — ROC per attack, overlaid by access tier.",
        "- `plots/fig2_reliability.png` — calibration, before and after.",
        "- `plots/fig3_score_histograms.png` — where clean and poisoned images land.",
        "- `plots/fig4_contributor_risk.png` — contributor risk with credible intervals.",
        "- `plots/fig6_disposition_confusion.png` — what we did against what was true.",
        "- `plots/fig7_runtime.png` — runtime scaling.",
        "",
        "See `COVERAGE.md` for what this system does and does not detect.",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "COVERAGE.md").write_text(coverage_md(rows, tamper_rows), encoding="utf-8")

    print(f"Wrote {out_dir / 'RESULTS.md'} and {out_dir / 'COVERAGE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
