"""The four-minute judge sequence, as a single command.

Five steps, in the order that tells the story:

1. A clean run. Everything green — proves this is not a flag-everything machine.
2. Poison one contributor. The report names them and recommends quarantine.
3. Swap the model file. The fingerprint fires, at black-box access.
4. Tamper with one inference record. Verification names it and the failure mode.
5. The coverage table, including the rows where we do badly.

Everything runs offline. ``--offline-assert`` is on throughout, so if any step
reached for the network the demo would stop rather than quietly succeed.
"""

from __future__ import annotations

import json
import shutil
from argparse import Namespace
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "attacks"


def _rule(step: str, title: str) -> None:
    print()
    print("=" * 72)
    print(f"  STEP {step}   {title}")
    print("=" * 72)


def run_demo(workdir: Path) -> int:
    from cvassure.attack.make_poison import run_config
    from cvassure.attack import receipt_attacks
    from cvassure.datasets import synth
    from cvassure.detect import model as model_suite
    from cvassure.ingest.models import load_model
    from cvassure.pipeline import run_audit
    from cvassure.provenance.receipts import ReceiptChain, init_keys, load_private_key
    from cvassure.provenance.verify import verify_file

    # Absolute throughout: step 2 changes directory so that the attack config's
    # relative `data/ood_pool` resolves, and any relative path we were holding
    # would quietly break underneath it.
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    repo = Path.cwd().resolve()

    print("cvassure demonstration — this machine is in airplane mode.")
    print(f"Working directory: {workdir}")

    # -- setup (not part of the four minutes) ---------------------------
    clean = workdir / "clean"
    if not clean.exists():
        print("\nPreparing a clean dataset and a model...")
        synth.build(clean, n_per_class=30, n_classes=10, n_contributors=5, seed=0)
        synth.build_ood_pool(workdir / "data" / "ood_pool", n=60, seed=99)

    models_dir = repo / "models"
    vendor = models_dir / "vendor.onnx"
    impostor = models_dir / "impostor.onnx"
    if not vendor.exists():
        print("  models/vendor.onnx is missing — run `make bootstrap` first.")
        return 1

    enrolled_path = workdir / "enrolled.json"
    if not enrolled_path.exists():
        enrolled_path.write_text(
            json.dumps(model_suite.enrol(load_model(vendor, 2))), encoding="utf-8"
        )

    keys = workdir / "keys"
    if not (keys / "priv.pem").exists():
        init_keys(keys)
    log_path = workdir / "inference.jsonl"
    if not log_path.exists():
        chain = ReceiptChain(load_private_key(keys / "priv.pem"))
        for i in range(500):
            chain.append(
                input_sha256=f"{i:064x}",
                model_weight_digest="a" * 64,
                preproc_config={"resize": 64},
                inference_config={"topk": 1},
                output={"label": "truck", "confidence": 0.91},
                timestamp_utc=f"2026-03-01T00:{i // 60:02d}:{i % 60:02d}.000000+00:00",
            )
        chain.write(log_path)

    def audit(dataset, model, out, receipts=None, tier=1):
        return run_audit(
            Namespace(
                dataset=str(dataset), model=str(model) if model else None,
                access_tier=tier, receipts=str(receipts) if receipts else None,
                pubkey=str(keys / "pub.pem"), reference=None,
                enrolled_fingerprint=str(enrolled_path), format="auto",
                contributors=None, contributor_from_path=None,
                out=str(out), seed=0, quiet=True,
            )
        )

    # -- 1 -----------------------------------------------------------------
    _rule("1", "A clean intake. Nothing wrong with it.")
    print("If this system flagged things here, nothing else it says would matter.\n")
    audit(clean, vendor, workdir / "out_clean", receipts=log_path)

    # -- 2 -----------------------------------------------------------------
    _rule("2", "One contributor starts sending tampered data.")
    poisoned = workdir / "poisoned"
    if not poisoned.exists():
        import os

        cwd = os.getcwd()
        os.chdir(workdir)
        run_config(CONFIG_DIR / "08_mixed_realistic.yaml", clean, poisoned,
                   workdir / "truth")
        os.chdir(cwd)
    print("The same command, on an intake where one source has been contaminated.\n")
    audit(poisoned, vendor, workdir / "out_poisoned", receipts=log_path)

    # -- 3 -----------------------------------------------------------------
    _rule("3", "Somebody swaps the model file.")
    swapped_dir = workdir / "swapped"
    swapped_dir.mkdir(exist_ok=True)
    swapped = swapped_dir / "vendor.onnx"
    shutil.copy2(impostor, swapped)
    print("The filename is unchanged. We are given answers only — access tier 0.")
    print("The fingerprint was recorded when the model was accepted.\n")
    audit(clean, swapped, workdir / "out_swapped", receipts=log_path, tier=0)

    # -- 4 -----------------------------------------------------------------
    _rule("4", "Somebody edits one record in the inference log.")
    info = receipt_attacks.alter_field(log_path, workdir / "tampered.jsonl", index=346)
    print(f"Record #{info['seq']} was edited after it was signed:")
    print(f"    before: {info['before']}")
    print(f"    after:  {info['after']}")
    print()
    result = verify_file(workdir / "tampered.jsonl", keys / "pub.pem")
    print(result.render())

    # -- 5 -----------------------------------------------------------------
    _rule("5", "What this system cannot do.")
    from cvassure.report.build import build_coverage, measured_from_sweep
    from cvassure.score.evaluate import read_findings

    findings = read_findings(workdir / "out_poisoned" / "findings.jsonl")
    measured = measured_from_sweep(repo / "results" / "sweep_raw.jsonl")
    coverage = build_coverage(findings, access_tier=1, measured=measured)
    if not measured:
        print("(No results run on disk — run `make results` so this table can quote")
        print(" measured numbers instead of saying 'not measured' everywhere.)\n")
    print(coverage.table().to_markdown())
    print()
    for line in coverage.statement():
        print(f"  {line}")
    print()
    print("The report for each step is in its own folder, as a single HTML file "
          "that opens with no network connection.")
    return 0
