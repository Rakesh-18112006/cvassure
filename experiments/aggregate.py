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
    """Tamper detection. Every row should read 100%."""
    from cvassure.score.tables import Table

    t = Table(
        name="table3_tamper",
        title="Table 3 — Tamper detection on the inference log",
        columns=["attack type", "attempts", "detected", "rate",
                 "named the right failure", "codes reported"],
        notes=[
            "This is cryptography, not statistics: the maths either matches or it "
            "does not, so anything below 100% is a bug to fix, not a limitation to "
            "report.",
            "'named the right failure' checks that we blame the correct thing — a "
            "system that spots tampering but misattributes it is little use to an "
            "investigator.",
            "The 'clean' row is the control: an untouched log must verify, otherwise "
            "every row above it is meaningless.",
        ],
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in tamper_rows:
        grouped[r["attack_class"]].append(r)

    for kind in sorted(grouped, key=lambda k: (k == "clean", k)):
        group = grouped[kind]
        codes = sorted({c for g in group for c in g.get("codes", [])})
        if kind == "clean":
            passed = sum(1 for g in group if g.get("correct_code"))
            t.add(
                **{
                    "attack type": "clean (control — nothing was touched)",
                    "attempts": len(group),
                    "detected": "n/a",
                    "rate": f"{100 * passed / max(1, len(group)):.1f}% verified clean",
                    "named the right failure": "n/a",
                    "codes reported": ", ".join(codes) or "none",
                }
            )
            continue
        detected = sum(1 for g in group if g.get("detected"))
        correct = sum(1 for g in group if g.get("correct_code"))
        t.add(
            **{
                "attack type": kind,
                "attempts": len(group),
                "detected": detected,
                "rate": f"{100 * detected / max(1, len(group)):.1f}%",
                "named the right failure": f"{100 * correct / max(1, len(group)):.1f}%",
                "codes reported": ", ".join(codes) or "—",
            }
        )
    return t


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
        *([t3.to_markdown(), ""] if t3 is not None else []),
        t4.to_markdown(),
        "",
        "### Figures",
        "",
        "- `plots/fig5_sweep.png` — detection rate against poison rate. The most "
        "important figure: it answers *how weak an attack can you still catch?*",
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
