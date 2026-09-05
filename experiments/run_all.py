"""The results run.

Sweeps attacks x poison rates x access tiers x seeds, scores every cell
against its own answer key, and writes one row per cell to
``results/sweep_raw.jsonl``. ``aggregate.py`` turns that into the tables.

Runs offline. Deterministic: the same command produces the same numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import time

# Set before torch is imported. PyTorch's OpenMP pool occasionally aborts at
# interpreter exit on macOS with "recursive_mutex lock failed" — after the run
# has already finished and written its results. The sweep is long enough that
# losing it to a teardown race at the very end is not acceptable, and the tests
# pin threads for the same reason.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "attacks"

#: attack config -> the poison-rate knob to sweep, and which attack class it
#: produces. Configs with no rate knob (duplicate flood, OOD insertion) are
#: swept by count instead.
SWEEP_SPECS: dict[str, dict[str, Any]] = {
    "01_badnets_5pct.yaml": {"knob": ("badnets_patch", "rate"), "attack": "badnets_patch"},
    "03_blended_faint.yaml": {"knob": ("blended_trigger", "rate"), "attack": "blended_trigger"},
    "04_label_flip_10pct.yaml": {"knob": ("label_flip", "rate"), "attack": "label_flip"},
    "06_duplicate_flood.yaml": {
        "knob": ("near_duplicate_flood", "n_seeds"), "attack": "near_duplicate_flood",
        "scale": "count",
    },
    "07_ood_insertion.yaml": {
        "knob": ("ood_insertion", "count"), "attack": "ood_insertion", "scale": "count",
    },
    # PS 2.2.1 names systematic mislabelling as its own attack class, so it is
    # swept as one rather than being left to fall out of the label-noise rows.
    "05_systematic_mislabel.yaml": {
        "knob": ("systematic_mislabel", "rate"), "attack": "systematic_mislabel",
        "scale": "source",
    },
}

POISON_RATES = (0.01, 0.02, 0.05, 0.10, 0.20)
TIERS = (0, 1, 2)
SEEDS = (1, 2, 3)


def build_dataset(root: Path, name: str, seed: int = 0) -> Path:
    """Prepare a dataset. CIFAR-10, GTSRB and the aerial subset are used when
    they have been placed under ``data/`` in advance; SYNTH-10 is always
    available and needs no download, which is what keeps ``make reproduce``
    working on an air-gapped machine."""
    from cvassure.datasets import synth

    if name == "synth10":
        out = root / "synth10"
        if not out.exists():
            synth.build(out, n_per_class=30, n_classes=10, n_contributors=5, seed=seed)
        return out
    candidate = REPO / "data" / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"dataset '{name}' is not in data/. Place it there in advance — the "
        f"results run never downloads anything."
    )


def _apply_rate(cfg: dict[str, Any], spec: dict[str, Any], rate: float,
                n_samples: int) -> dict[str, Any]:
    """Set the attack's knob so that ``rate`` means the same thing everywhere.

    The sweep's x-axis is "share of the *dataset* that is poisoned". Most
    attacks take exactly that. A contributor-scoped attack does not: its rate
    is a share of one contributor's images, and that contributor holds only
    1/k of the intake — so a requested 5% arrives as 1% and the row measures
    almost nothing. Rates are converted rather than passed through.
    """
    attack_type, knob = spec["knob"]
    out = json.loads(json.dumps(cfg))
    n_contributors = int((out.get("contributors") or {}).get("n", 5))
    for entry in out.get("attacks", []):
        if entry.get("type") == attack_type:
            if spec.get("scale") == "source":
                # One contributor holds about 1/k of the data, so to poison
                # `rate` of the dataset they must mislabel `rate * k` of theirs.
                entry[knob] = float(min(1.0, rate * n_contributors))
                # The shipped config restricts the attack to two class names so
                # the demo is legible. Left in place it caps the reachable rate
                # at whatever share those classes happen to be, which would make
                # the high-rate cells unmeasurable.
                entry.pop("mapping", None)
            elif spec.get("scale") == "count":
                if knob == "n_seeds":
                    copies = int(entry.get("n_copies", 8))
                    entry[knob] = max(1, int(round(rate * n_samples / copies)))
                else:
                    entry[knob] = max(1, int(round(rate * n_samples)))
            else:
                entry[knob] = float(rate)
    return out


def cells(datasets: list[str], rates=POISON_RATES, tiers=TIERS, seeds=SEEDS) -> Iterator[dict]:
    for dataset in datasets:
        for config, spec in SWEEP_SPECS.items():
            for rate in rates:
                for tier in tiers:
                    for seed in seeds:
                        yield {
                            "dataset": dataset, "config": config, "attack": spec["attack"],
                            "poison_rate": rate, "access_tier": tier, "seed": seed,
                        }


def run_cell(cell: dict[str, Any], workdir: Path, model_path: Path | None,
             enrolled: dict | None, store) -> dict[str, Any]:
    import yaml

    from cvassure.attack.make_poison import load_ground_truth, run_config
    from cvassure.detect import data as data_suite
    from cvassure.detect.base import AuditContext
    from cvassure.ingest.dataset import load_dataset
    from cvassure.ingest.models import load_model
    from cvassure.score import metrics

    spec = SWEEP_SPECS[cell["config"]]
    clean_root = build_dataset(REPO / "data", cell["dataset"])
    n_samples = len(list(clean_root.rglob("*.png")))

    cfg = yaml.safe_load((CONFIGS / cell["config"]).read_text(encoding="utf-8"))
    cfg = _apply_rate(cfg, spec, cell["poison_rate"], n_samples)

    tag = f"{cell['dataset']}_{spec['attack']}_{cell['poison_rate']}_{cell['seed']}"
    cell_dir = workdir / tag
    cfg_path = cell_dir / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    poisoned = cell_dir / "data"
    truth_dir = cell_dir / "truth"
    import os

    cwd = os.getcwd()
    os.chdir(REPO)  # attack configs reference data/ood_pool relative to the repo
    try:
        run_config(cfg_path, clean_root, poisoned, truth_dir, seed=cell["seed"])
    finally:
        os.chdir(cwd)

    truth = load_ground_truth(truth_dir)
    ds = load_dataset(poisoned)

    model = load_model(model_path, cell["access_tier"]) if model_path else None
    ctx = AuditContext(
        samples=ds.samples, access_tier=cell["access_tier"], model=model,
        embeddings=store, enrolled_fingerprint=enrolled, out_dir=cell_dir / "out",
        seed=cell["seed"],
    )

    started = time.perf_counter()
    findings, results = data_suite.run_all(ctx)
    seconds = time.perf_counter() - started

    # Keep the findings next to their answer key. Five of the seven figures
    # (ROC panels, reliability, score histograms, contributor risk, the
    # disposition confusion matrix) need both sides of the join, and throwing
    # the findings away here would mean the results run could only ever produce
    # the other two.
    from cvassure.score.evaluate import write_findings

    write_findings(findings, cell_dir / "findings.jsonl")

    labels = {r["sample_id"]: r["is_poisoned"] for r in truth["samples"]}
    best = {sid: 0.0 for sid in labels}
    for f in findings:
        if f.asset_type == "sample" and not f.is_unavailable and f.asset_ref in best:
            best[f.asset_ref] = max(best[f.asset_ref], f.raw_score)
    order = sorted(best)
    y = np.array([labels[i] for i in order])
    s = np.array([best[i] for i in order])

    auc = metrics.bootstrap_ci(metrics.roc_auc, y, s, n=200, seed=cell["seed"])
    tpr = metrics.bootstrap_ci(
        lambda a, b: metrics.tpr_at_fpr(a, b, 0.01), y, s, n=200, seed=cell["seed"] + 7
    )
    flagged = np.array([best[i] >= 0.5 for i in order])

    return {
        **cell,
        "n": int(y.size),
        "n_poisoned": int(y.sum()),
        "auroc": auc.value, "auroc_lo": auc.lo, "auroc_hi": auc.hi,
        "tpr_at_1pct": tpr.value, "tpr_lo": tpr.lo, "tpr_hi": tpr.hi,
        "ece": metrics.ece(y, s),
        "average_precision": metrics.average_precision(y, s),
        "verdict": metrics.verdict(tpr.value),
        "false_alarm_rate": float(flagged[y == 0].mean()) if (y == 0).any() else float("nan"),
        "recall_at_threshold": float(flagged[y == 1].mean()) if (y == 1).any() else float("nan"),
        "seconds": seconds,
        "per_detector_seconds": {r.detector_id: round(r.seconds, 4) for r in results},
        "peak_ram_mb": max((r.peak_ram_mb for r in results), default=0.0),
        "errors": [r.detector_id for r in results if r.error],
        "cell_dir": str(cell_dir),
    }


def run_tamper_cells(workdir: Path, seeds=SEEDS, n_receipts: int = 500) -> list[dict[str, Any]]:
    """Measure tamper detection on the inference log.

    Separate from the data sweep because there is nothing statistical to sweep:
    a signature either verifies or it does not. What is worth measuring is that
    every attack type is caught, every time, and that the *named* failure mode
    is the right one — a system that detects tampering but blames the wrong
    thing is not much use to an investigator.
    """
    from cvassure.attack import receipt_attacks
    from cvassure.provenance.receipts import ReceiptChain, init_keys, load_private_key
    from cvassure.provenance.verify import verify_log

    out: list[dict[str, Any]] = []
    for seed in seeds:
        cell_dir = workdir / f"tamper_seed{seed}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        priv, pub = init_keys(cell_dir / "keys", overwrite=True)

        chain = ReceiptChain(load_private_key(priv))
        for i in range(n_receipts):
            chain.append(
                input_sha256=f"{i:064x}",
                model_weight_digest="a" * 64,
                preproc_config={"resize": 64},
                inference_config={"topk": 1},
                output={"label": "truck", "confidence": 0.91},
                timestamp_utc=f"2026-03-01T00:{i // 60:02d}:{i % 60:02d}.000000+00:00",
            )
        clean_log = cell_dir / "signed.jsonl"
        chain.write(clean_log)

        # the control: an untouched log must verify, or every row below is noise
        control = verify_log(clean_log, pub)
        out.append(
            {
                "attack_class": "clean", "seed": seed, "detected": not control.ok,
                "expected_code": None, "codes": sorted(control.codes),
                "correct_code": control.ok, "n_receipts": n_receipts,
            }
        )

        for kind in ("alter_field", "replay_receipt", "delete_receipt", "reorder"):
            info = receipt_attacks.apply(
                {"type": kind}, receipts_path=clean_log, out_dir=cell_dir / kind, seed=seed
            )
            result = verify_log(info["path"], pub)
            out.append(
                {
                    "attack_class": info["attack_class"],
                    "seed": seed,
                    "detected": not result.ok,
                    "expected_code": info["expected_code"],
                    "codes": sorted(result.codes),
                    "correct_code": info["expected_code"] in result.codes,
                    "n_receipts": n_receipts,
                    "seq": info.get("seq"),
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["synth10"])
    parser.add_argument("--out", default=str(REPO / "results"))
    parser.add_argument("--workdir", default=str(REPO / "results" / ".sweep"))
    parser.add_argument("--model", default=str(REPO / "models" / "vendor.onnx"))
    parser.add_argument("--rates", nargs="+", type=float, default=list(POISON_RATES))
    parser.add_argument("--tiers", nargs="+", type=int, default=list(TIERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--limit", type=int, default=0, help="stop after N cells (smoke test)")
    parser.add_argument("--skip-tamper", action="store_true")
    args = parser.parse_args(argv)

    from cvassure.detect import model as model_suite
    from cvassure.detect.embed import EmbeddingStore
    from cvassure.ingest.models import load_model

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model) if args.model and Path(args.model).exists() else None
    enrolled = model_suite.enrol(load_model(model_path, 2)) if model_path else None
    store = EmbeddingStore(cache_dir=REPO / ".cache" / "embeddings")

    todo = list(cells(args.datasets, tuple(args.rates), tuple(args.tiers), tuple(args.seeds)))
    if args.limit:
        todo = todo[: args.limit]

    raw_path = out_dir / "sweep_raw.jsonl"
    print(f"{len(todo)} cells -> {raw_path}")
    with raw_path.open("w", encoding="utf-8") as fh:
        for i, cell in enumerate(todo, 1):
            label = (f"{cell['dataset']}/{cell['attack']}/"
                     f"{100 * cell['poison_rate']:.0f}%/tier{cell['access_tier']}/"
                     f"seed{cell['seed']}")
            print(f"  [{i}/{len(todo)}] {label}", flush=True)
            try:
                row = run_cell(cell, workdir, model_path, enrolled, store)
            except Exception as exc:  # one bad cell must not lose the whole run
                print(f"      failed: {type(exc).__name__}: {exc}")
                row = {**cell, "error": f"{type(exc).__name__}: {exc}"}
            fh.write(json.dumps(row) + "\n")
            fh.flush()

    if not args.skip_tamper:
        print("\nMeasuring tamper detection on the inference log...")
        tamper = run_tamper_cells(workdir, tuple(args.seeds))
        tamper_path = out_dir / "tamper_raw.jsonl"
        with tamper_path.open("w", encoding="utf-8") as fh:
            for row in tamper:
                fh.write(json.dumps(row) + "\n")
        caught = sum(1 for r in tamper if r["attack_class"] != "clean" and r["detected"])
        total = sum(1 for r in tamper if r["attack_class"] != "clean")
        print(f"  {caught}/{total} tampering attempts detected -> {tamper_path}")

    print(f"\nDone. Now run: python experiments/aggregate.py --out {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
