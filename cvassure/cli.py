"""cvassure command line.

Kept deliberately thin: every subcommand is a few lines that parse arguments
and call into a module. Nothing here should contain logic worth testing on its
own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def _cmd_provenance_init_keys(args: argparse.Namespace) -> int:
    from cvassure.provenance.receipts import init_keys

    priv, pub = init_keys(args.out, overwrite=args.overwrite)
    print(f"Private key: {priv}  (keep this secret, it never leaves this machine)")
    print(f"Public key:  {pub}   (share this so others can check your log)")
    return 0


def _cmd_provenance_sign(args: argparse.Namespace) -> int:
    from cvassure.provenance.receipts import load_private_key, sign_unsigned_file

    receipts = sign_unsigned_file(args.receipts, args.out, load_private_key(args.key))
    print(f"Signed {len(receipts)} receipts into a tamper-evident chain -> {args.out}")
    return 0


def _cmd_provenance_verify(args: argparse.Namespace) -> int:
    from cvassure.provenance.verify import verify_file

    result = verify_file(args.receipts, args.pubkey)
    print(result.render())
    return 0 if result.ok else 1


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


def _cmd_ingest_inspect(args: argparse.Namespace) -> int:
    from cvassure.ingest import inspect

    return inspect.run(
        dataset=args.dataset,
        model=args.model,
        access_tier=args.access_tier,
        fmt=args.format,
        contributors=args.contributors,
        contributor_from_path=args.contributor_from_path,
    )


# --------------------------------------------------------------------------
# attack
# --------------------------------------------------------------------------


def _cmd_attack_run(args: argparse.Namespace) -> int:
    from cvassure.attack.make_poison import run_config

    summary = run_config(args.config, args.dataset, args.out, args.truth, seed=args.seed)
    print(summary.render())
    return 0


# --------------------------------------------------------------------------
# detect / score / report / audit
# --------------------------------------------------------------------------


def _cmd_score_all(args: argparse.Namespace) -> int:
    from cvassure.score.evaluate import score_all

    score_all(truth=args.truth, findings=args.findings, out=args.out)
    print(f"Tables, plots and RESULTS.md written to {args.out}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from cvassure.pipeline import run_audit

    return run_audit(args)


def _cmd_audit_verify(args: argparse.Namespace) -> int:
    from cvassure.report.audit import verify_audit_log

    result = verify_audit_log(args.log)
    print(result.render())
    return 0 if result.ok else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    from cvassure.demo import run_demo

    return run_demo(Path(args.workdir))


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cvassure",
        description="Integrity assurance for computer-vision data, models and "
        "inference outputs. Runs entirely offline.",
    )
    p.add_argument(
        "--offline-assert",
        action="store_true",
        help="fail hard if any code path attempts a network call",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # -- provenance ------------------------------------------------------
    prov = sub.add_parser("provenance", help="sign and verify inference receipt logs")
    prov_sub = prov.add_subparsers(dest="subcommand", required=True)

    ik = prov_sub.add_parser("init-keys", help="generate a local Ed25519 keypair")
    ik.add_argument("--out", default="keys/", help="directory to write priv.pem and pub.pem")
    ik.add_argument("--overwrite", action="store_true")
    ik.set_defaults(func=_cmd_provenance_init_keys)

    sg = prov_sub.add_parser("sign", help="turn raw inference records into a signed chain")
    sg.add_argument("--receipts", required=True)
    sg.add_argument("--out", required=True)
    sg.add_argument("--key", default="keys/priv.pem")
    sg.set_defaults(func=_cmd_provenance_sign)

    vf = prov_sub.add_parser("verify", help="check a signed chain for tampering")
    vf.add_argument("--receipts", required=True)
    vf.add_argument("--pubkey", default="keys/pub.pem")
    vf.set_defaults(func=_cmd_provenance_verify)

    # -- ingest ----------------------------------------------------------
    ing = sub.add_parser("ingest", help="load datasets and models")
    ing_sub = ing.add_subparsers(dest="subcommand", required=True)
    ins = ing_sub.add_parser("inspect", help="summarise a dataset and model, and say "
                                             "which checks are unavailable")
    ins.add_argument("--dataset", required=True)
    ins.add_argument("--model")
    ins.add_argument("--access-tier", type=int, default=0, choices=[0, 1, 2])
    ins.add_argument("--format", default="auto", choices=["auto", "coco", "yolo", "folder"])
    ins.add_argument("--contributors", help="path to contributors.json")
    ins.add_argument("--contributor-from-path", help="regex with one capture group")
    ins.set_defaults(func=_cmd_ingest_inspect)

    # -- attack ----------------------------------------------------------
    atk = sub.add_parser("attack", help="build poisoned test data plus its answer key")
    atk_sub = atk.add_subparsers(dest="subcommand", required=True)
    ar = atk_sub.add_parser("run", help="apply an attack config to a clean dataset")
    ar.add_argument("--config", required=True)
    ar.add_argument("--dataset", required=True)
    ar.add_argument("--out", required=True)
    ar.add_argument("--truth", required=True, help="separate folder for ground_truth.json")
    ar.add_argument("--seed", type=int)
    ar.set_defaults(func=_cmd_attack_run)

    # -- score -----------------------------------------------------------
    sc = sub.add_parser("score", help="score findings against ground truth")
    sc_sub = sc.add_subparsers(dest="subcommand", required=True)
    sa = sc_sub.add_parser("all", help="all four tables and all seven plots")
    sa.add_argument("--truth", required=True)
    sa.add_argument("--findings", required=True)
    sa.add_argument("--out", required=True)
    sa.set_defaults(func=_cmd_score_all)

    # -- audit -----------------------------------------------------------
    au = sub.add_parser("audit", help="run the full assurance audit")
    au.add_argument("--dataset", required=True)
    au.add_argument("--model")
    au.add_argument("--access-tier", type=int, default=0, choices=[0, 1, 2])
    au.add_argument("--receipts")
    au.add_argument("--pubkey", default="keys/pub.pem")
    au.add_argument("--reference", help="reference distribution profile JSON")
    au.add_argument("--enrolled-fingerprint", help="model fingerprint recorded at intake")
    au.add_argument("--format", default="auto", choices=["auto", "coco", "yolo", "folder"])
    au.add_argument("--contributors")
    au.add_argument("--contributor-from-path")
    au.add_argument("--out", required=True)
    au.add_argument("--seed", type=int, default=0)
    au.add_argument("--quiet", action="store_true")
    au.set_defaults(func=_cmd_audit)

    auv = sub.add_parser("audit-verify", help="recheck the audit log's own hash chain")
    auv.add_argument("--log", default="results/audit_log.jsonl")
    auv.set_defaults(func=_cmd_audit_verify)

    # -- demo ------------------------------------------------------------
    dm = sub.add_parser("demo", help="run the four-minute judge sequence")
    dm.add_argument("--workdir", default="results/demo")
    dm.set_defaults(func=_cmd_demo)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "offline_assert", False):
        from cvassure.core.offline import assert_offline

        assert_offline()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
