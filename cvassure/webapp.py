"""A small local web console over the cvassure pipeline.

This is deliberately thin: it uploads a dataset or a model into a per-run
working directory, builds the same ``AuditContext`` the CLI builds, and calls
the same detector suites (``cvassure.detect.data`` / ``cvassure.detect.model``
/ ``cvassure.provenance.verify``). Nothing here re-implements a check — the
web console and ``cvassure audit`` share every number they show.

Everything still runs offline: uploads are written to disk under
``results/web_runs/`` and never leave the machine.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from cvassure.core.schemas import Finding
from cvassure.detect import data as data_suite
from cvassure.detect import model as model_suite
from cvassure.detect.base import AuditContext
from cvassure.detect.embed import EmbeddingStore
from cvassure.ingest.dataset import Dataset, detect_format, load_dataset
from cvassure.ingest.models import UnusableModel, load_model
from cvassure.provenance import custody
from cvassure.report import build as report_build
from cvassure.report import html as report_html

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web"
RUNS_ROOT = REPO_ROOT / "results" / "web_runs"
SWEEP_PATH = REPO_ROOT / "results" / "sweep_raw.jsonl"
DEFAULT_PUBKEY = REPO_ROOT / "keys" / "pub.pem"
#: Organisation identities persist here, separate from timestamped run
#: directories — an actor's keypair is a long-lived identity, not one audit.
CUSTODY_KEYS_DIR = REPO_ROOT / "results" / "web_runs" / "_custody_actors"

MODULES = ("data", "model", "receipts", "custody")

#: Curated sample files already sitting on disk, offered as one-click
#: quick-picks so testing never depends on dragging a file out of Downloads.
#: An entry whose file is missing (e.g. `make demo` was never run) is simply
#: left out by /api/fixtures rather than shown as a broken button.
FIXTURES: dict[str, list[dict[str, str]]] = {
    "data": [
        {"label": "Clean dataset", "path": "data/synth10", "field": "dataset"},
        {"label": "4 attacks planted at once", "path": "results/demo_fixtures/poisoned_dataset_4attacks.zip", "field": "dataset"},
        {"label": "4 attacks, different batch", "path": "results/demo_fixtures/poisoned_dataset_variant2.zip", "field": "dataset"},
        {"label": "make demo's poisoned set", "path": "results/demo/poisoned", "field": "dataset"},
        {"label": "make demo's clean set", "path": "results/demo/clean", "field": "dataset"},
    ],
    "model": [
        {"label": "Genuine vendor model", "path": "models/vendor.pt", "field": "model"},
        {"label": "Backdoored model", "path": "results/demo_fixtures/poisoned_model_backdoor.pt", "field": "model"},
        {"label": "Weights quietly edited", "path": "results/demo_fixtures/poisoned_model_perturbed.pt", "field": "model"},
        {"label": "make demo's substituted model", "path": "results/demo/swapped/vendor.onnx", "field": "model"},
        {"label": "Enrolled fingerprint for vendor.pt", "path": "results/demo_fixtures/enrolled_vendor_fingerprint.json", "field": "enrolled_fingerprint"},
        {"label": "Dataset for activation-based checks", "path": "data/synth10", "field": "dataset"},
    ],
    "receipts": [
        {"label": "Tampered inference log", "path": "results/demo/tampered.jsonl", "field": "receipts"},
        {"label": "Its public key", "path": "results/demo/keys/pub.pem", "field": "pubkey"},
    ],
    "custody": [
        {"label": "Clean 3-hop chain", "path": "results/demo_fixtures/custody/custody_chain_clean.jsonl", "field": "chain"},
        {"label": "Edited after signing", "path": "results/demo_fixtures/custody/custody_chain_edited_after_signing.jsonl", "field": "chain"},
        {"label": "Substituted handoff", "path": "results/demo_fixtures/custody/custody_chain_substituted_handoff.jsonl", "field": "chain"},
        {"label": "Actors keyring", "path": "results/demo_fixtures/custody/custody_actors_keyring.json", "field": "keyring"},
        {"label": "Clean dataset to certify", "path": "data/synth10", "field": "dataset"},
        {"label": "Genuine vendor model to certify", "path": "models/vendor.pt", "field": "model"},
    ],
}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _new_run_dir(module: str) -> tuple[str, Path]:
    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = RUNS_ROOT / module / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _save_upload(field: str, dest: Path) -> Path | None:
    f = request.files.get(field)
    if f is None or not f.filename:
        return None
    dest.mkdir(parents=True, exist_ok=True)
    name = secure_filename(f.filename) or "upload"
    path = dest / name
    f.save(path)
    return path


def _resolve_local_path(raw: str | None) -> Path | None:
    """A path the uploader typed instead of browsing to a file — this app and
    its browser always run on the same machine, so reading a path directly is
    exactly as local as an upload, just without the copy."""
    if not raw or not raw.strip():
        return None
    p = Path(raw.strip()).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"nothing on this machine at '{raw}'")
    return p


def _resolve_file_source(field: str, run_dir: Path) -> Path | None:
    """An uploaded file for ``field``, or — if nothing was uploaded — a path
    typed into ``<field>_path``. Whichever was actually given."""
    upload = _save_upload(field, run_dir / "uploads")
    if upload is not None:
        return upload
    return _resolve_local_path(request.form.get(f"{field}_path"))


def _resolve_dataset_source(field: str, run_dir: Path) -> Path | None:
    """Like :func:`_resolve_file_source`, but a `.zip` is unpacked — from an
    upload or from a path — while a folder already on disk is used as-is."""
    source = _resolve_file_source(field, run_dir)
    if source is None:
        return None
    if source.is_file() and source.suffix.lower() == ".zip":
        return _extract_dataset_zip(source, run_dir / "dataset")
    return source


def _extract_dataset_zip(zip_path: Path, dest: Path) -> Path:
    """Unzip and find the actual dataset root, tolerating one wrapping folder."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    try:
        detect_format(dest)
        return dest
    except ValueError:
        pass
    subdirs = [p for p in dest.iterdir() if p.is_dir() and not p.name.startswith("__MACOSX")]
    if len(subdirs) == 1:
        try:
            detect_format(subdirs[0])
            return subdirs[0]
        except ValueError:
            pass
    raise ValueError(
        "could not tell what format this dataset is in. Zip an ImageFolder layout "
        "(one subfolder per class), a COCO instances JSON + images, or a YOLO "
        "data.yaml + labels/ folder."
    )


def _custody_output_digest(
    *, dataset_source: Path | None, model_path: Path | None,
    explicit: str | None, images_only: bool,
) -> str:
    """What a custody hop actually signs off on: the real content hash of
    whatever was given — uploaded or a local path — computed the same way the
    CLI computes it, never typed in by hand unless nothing else was given."""
    if explicit:
        return explicit
    if dataset_source is not None:
        dataset = load_dataset(dataset_source)
        return custody.dataset_content_digest(dataset.samples, include_labels=not images_only)
    if model_path is not None:
        from cvassure.detect.fingerprint import compute_fingerprint
        from cvassure.detect.weight_digest import canonical_weight_digest
        from cvassure.ingest.models import AccessDenied

        model = load_model(model_path, access_tier=1)
        try:
            return canonical_weight_digest(model.weights())
        except AccessDenied:
            return compute_fingerprint(model)["digest"]
    raise ValueError("attach a dataset .zip, a model file, or type a digest directly")


def _finding_to_json(f: Finding) -> dict[str, Any]:
    d = f.to_dict()
    d["score"] = f.score
    d["is_unavailable"] = f.is_unavailable
    return d


def _write_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _write_findings(run_dir: Path, findings: list[Finding]) -> None:
    with (run_dir / "findings.jsonl").open("w", encoding="utf-8") as fh:
        for f in findings:
            fh.write(json.dumps(_finding_to_json(f)) + "\n")


def _read_findings(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "findings.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _list_runs(module: str) -> list[dict[str, Any]]:
    root = RUNS_ROOT / module
    if not root.exists():
        return []
    out = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            out.append(json.loads(meta_path.read_text(encoding="utf-8")))
    return out


def _run_dir_or_404(module: str, run_id: str) -> Path | None:
    run_dir = RUNS_ROOT / module / secure_filename(run_id)
    return run_dir if run_dir.exists() else None


def _coverage_rows(findings: list[Finding], access_tier: int) -> list[dict[str, Any]]:
    measured = report_build.measured_from_sweep(SWEEP_PATH)
    coverage = report_build.build_coverage(findings, access_tier=access_tier, measured=measured)
    return [r.to_dict() for r in coverage.rows]


def _compute_verdict(findings: list[Finding], n_samples: int, *, ignore_model: bool = False):
    # The data suite includes a detector (spectral_signature) that needs a
    # model handle, so a data-only run always produces an UNAVAILABLE
    # model-asset finding. That is honest, but "Model: NOT VERIFIED" on a page
    # where no model was ever supplied reads as a false alarm — so the data
    # module's own verdict is computed with model-asset findings excluded.
    # They are still visible in the findings table and the coverage tab.
    if ignore_model:
        findings = [f for f in findings if f.asset_type != "model"]
    return report_build.overall_verdict(findings, n_samples=n_samples)


def _verdict_dict(verdict) -> dict[str, Any]:
    return {"headline": verdict.headline, "colour": verdict.colour, "lines": verdict.lines}


def _write_assurance_record(run_dir: Path, record: dict[str, Any]) -> None:
    (run_dir / "assurance_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )


def _read_assurance_record(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "assurance_record.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _build_html_report(
    run_dir: Path, findings: list[Finding], inputs: dict[str, Any], access_tier: int,
    verdict, *, summary: dict[str, Any] | None = None,
    category_risk: list[dict[str, Any]] | None = None,
) -> Path:
    coverage = report_build.build_coverage(
        findings, access_tier=access_tier, measured=report_build.measured_from_sweep(SWEEP_PATH)
    )
    return report_html.write(
        run_dir / "report.html",
        findings=findings,
        verdict=verdict,
        coverage=coverage,
        inputs=inputs,
        summary=summary,
        category_risk=category_risk,
    )


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    # -- static frontend --------------------------------------------------

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/<path:path>")
    def static_files(path: str):
        if (WEB_DIR / path).is_file():
            return send_from_directory(WEB_DIR, path)
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "offline": True})

    @app.get("/api/fixtures")
    def fixtures():
        """Sample files already on this machine, so testing never depends on
        finding something to drag out of Downloads."""
        out: dict[str, list[dict[str, str]]] = {}
        for module, entries in FIXTURES.items():
            available = [e for e in entries if (REPO_ROOT / e["path"]).exists()]
            if available:
                out[module] = available
        return jsonify(out)

    # -- data integrity -----------------------------------------------------

    @app.post("/api/data/runs")
    def data_run():
        run_id, run_dir = _new_run_dir("data")
        try:
            root = _resolve_dataset_source("dataset", run_dir)
            if root is None:
                return jsonify({
                    "error": "attach a dataset .zip (ImageFolder / COCO / YOLO), or give a "
                             "path to one already on this machine",
                }), 400

            contributors_path = _resolve_file_source("contributors", run_dir)
            contributor_from_path = request.form.get("contributor_from_path") or None

            dataset: Dataset = load_dataset(
                root,
                contributors=contributors_path,
                contributor_from_path=contributor_from_path,
            )
            if not dataset.samples:
                return jsonify({"error": "no images were found in that upload"}), 400

            ctx = AuditContext(
                samples=dataset.samples,
                access_tier=0,
                embeddings=EmbeddingStore(cache_dir=run_dir / ".cache"),
                out_dir=run_dir,
                limitations=list(dataset.limitations),
            )
            findings, results = data_suite.run_all(ctx)
            _write_findings(run_dir, findings)

            summary = dataset.summary()
            verdict_obj = _compute_verdict(findings, n_samples=len(dataset), ignore_model=True)
            assessment_id = report_build.make_assessment_id(run_id)
            category_risk = report_build.data_category_risk(findings)
            record = report_build.assurance_record(
                assessment_id=assessment_id,
                verdict=verdict_obj,
                findings=findings,
                dataset_summary=summary,
                limitations=dataset.limitations,
            )
            meta = {
                "id": run_id,
                "module": "data",
                "created": _dt.datetime.now().isoformat(timespec="seconds"),
                "assessment_id": assessment_id,
                "summary": summary,
                "verdict": _verdict_dict(verdict_obj),
                "risk": report_build.overall_risk(verdict_obj),
                "confidence": report_build.overall_confidence(findings),
                "category_risk": category_risk,
                "n_findings": len(findings),
                "n_flagged": sum(
                    1 for f in findings
                    if f.asset_type == "sample" and not f.is_unavailable and f.disposition != "accept"
                ),
                "detectors": sorted({r.detector_id for r in results}),
            }
            _write_meta(run_dir, meta)
            _write_assurance_record(run_dir, record)
            _build_html_report(
                run_dir, findings,
                inputs={
                    "Dataset": f"{root} ({dataset.fmt.upper()}, {len(dataset)} images)",
                    "Model": "none supplied",
                    "Access granted": "tier 0",
                    "Network access": "none — this run stayed on this machine",
                },
                access_tier=0, verdict=verdict_obj, summary=record, category_risk=category_risk,
            )

            return jsonify({
                **meta,
                "findings": [_finding_to_json(f) for f in findings],
                "coverage": _coverage_rows(findings, access_tier=0),
                "assurance_record": record,
            })
        except Exception as exc:  # noqa: BLE001 — surfaced to the uploader, not swallowed
            shutil.rmtree(run_dir, ignore_errors=True)
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/data/runs")
    def data_runs():
        return jsonify(_list_runs("data"))

    @app.get("/api/data/runs/<run_id>")
    def data_run_detail(run_id: str):
        run_dir = _run_dir_or_404("data", run_id)
        if run_dir is None:
            return jsonify({"error": "no such run"}), 404
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        findings = _read_findings(run_dir)
        rebuilt = [Finding.from_dict(f) for f in findings]
        return jsonify({
            **meta,
            "findings": findings,
            "coverage": _coverage_rows(rebuilt, access_tier=0),
            "assurance_record": _read_assurance_record(run_dir),
        })

    # -- model integrity ------------------------------------------------------

    @app.post("/api/model/runs")
    def model_run():
        run_id, run_dir = _new_run_dir("model")
        try:
            model_path = _resolve_file_source("model", run_dir)
            if model_path is None:
                return jsonify({
                    "error": "attach a model file (.onnx, .pt, .pth), or give a path to one",
                }), 400

            access_tier = int(request.form.get("access_tier", 0))
            if access_tier not in (0, 1, 2):
                return jsonify({"error": "access_tier must be 0, 1 or 2"}), 400

            try:
                model = load_model(model_path, access_tier)
            except UnusableModel as exc:
                return jsonify({"error": exc.plain_english}), 400

            samples = []
            dataset_summary = None
            dataset_root = _resolve_dataset_source("dataset", run_dir)
            if dataset_root is not None:
                dataset = load_dataset(dataset_root)
                samples = dataset.samples
                dataset_summary = dataset.summary()

            enrolled = None
            fp_path = _resolve_file_source("enrolled_fingerprint", run_dir)
            if fp_path is not None:
                enrolled = json.loads(fp_path.read_text(encoding="utf-8"))

            ctx = AuditContext(
                samples=samples,
                access_tier=access_tier,
                model=model,
                embeddings=EmbeddingStore(cache_dir=run_dir / ".cache"),
                enrolled_fingerprint=enrolled,
                out_dir=run_dir,
            )
            findings, results = model_suite.run_all(ctx)
            _write_findings(run_dir, findings)

            verdict_obj = _compute_verdict(findings, n_samples=len(samples))
            assessment_id = report_build.make_assessment_id(run_id)
            model_info = model.describe()
            record = report_build.assurance_record(
                assessment_id=assessment_id,
                verdict=verdict_obj,
                findings=findings,
                model_info=model_info,
                limitations=model.unavailable_checks(),
            )
            meta = {
                "id": run_id,
                "module": "model",
                "created": _dt.datetime.now().isoformat(timespec="seconds"),
                "assessment_id": assessment_id,
                "model": model_info,
                "access_tier": access_tier,
                "dataset_summary": dataset_summary,
                "enrolled": enrolled is not None,
                "verdict": _verdict_dict(verdict_obj),
                "risk": report_build.overall_risk(verdict_obj),
                "confidence": report_build.overall_confidence(findings),
                "n_findings": len(findings),
                "detectors": sorted({r.detector_id for r in results}),
            }
            _write_meta(run_dir, meta)
            _write_assurance_record(run_dir, record)
            _build_html_report(
                run_dir, findings,
                inputs={
                    "Dataset": dataset_summary["root"] if dataset_summary else "none supplied",
                    "Model": f"{model_path.name} ({model.kind})",
                    "Access granted": f"tier {access_tier}",
                    "Network access": "none — this run stayed on this machine",
                },
                access_tier=access_tier, verdict=verdict_obj, summary=record,
            )

            return jsonify({
                **meta,
                "findings": [_finding_to_json(f) for f in findings],
                "coverage": _coverage_rows(findings, access_tier=access_tier),
                "assurance_record": record,
            })
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(run_dir, ignore_errors=True)
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/model/runs")
    def model_runs():
        return jsonify(_list_runs("model"))

    @app.get("/api/model/runs/<run_id>")
    def model_run_detail(run_id: str):
        run_dir = _run_dir_or_404("model", run_id)
        if run_dir is None:
            return jsonify({"error": "no such run"}), 404
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        findings = _read_findings(run_dir)
        rebuilt = [Finding.from_dict(f) for f in findings]
        return jsonify({
            **meta,
            "findings": findings,
            "coverage": _coverage_rows(rebuilt, access_tier=meta.get("access_tier", 0)),
            "assurance_record": _read_assurance_record(run_dir),
        })

    @app.post("/api/model/enroll")
    def model_enroll():
        run_id, run_dir = _new_run_dir("enroll")
        try:
            model_path = _resolve_file_source("model", run_dir)
            if model_path is None:
                return jsonify({
                    "error": "attach a model file (.onnx, .pt, .pth), or give a path to one",
                }), 400
            access_tier = int(request.form.get("access_tier", 2))
            try:
                model = load_model(model_path, access_tier)
            except UnusableModel as exc:
                return jsonify({"error": exc.plain_english}), 400
            record = model_suite.enrol(model)
            out_path = run_dir / "enrolled_fingerprint.json"
            out_path.write_text(json.dumps(record), encoding="utf-8")
            return send_file(out_path, as_attachment=True, download_name="enrolled_fingerprint.json")
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(run_dir, ignore_errors=True)
            return jsonify({"error": str(exc)}), 400

    # -- inference receipts ---------------------------------------------------

    @app.post("/api/receipts/runs")
    def receipts_run():
        run_id, run_dir = _new_run_dir("receipts")
        try:
            from cvassure.provenance.receipts import load_public_key, read_receipts
            from cvassure.provenance.verify import findings_from_result, verify_log

            receipts_path = _resolve_file_source("receipts", run_dir)
            if receipts_path is None:
                return jsonify({"error": "attach a receipts .jsonl file, or give a path to one"}), 400

            pubkey_path = _resolve_file_source("pubkey", run_dir)
            if pubkey_path is None:
                if not DEFAULT_PUBKEY.exists():
                    return jsonify({
                        "error": "attach the public key this log was signed with "
                                 "(no default keys/pub.pem was found on this machine)",
                    }), 400
                pubkey_path = DEFAULT_PUBKEY

            result = verify_log(read_receipts(receipts_path), load_public_key(pubkey_path))
            findings = findings_from_result(result, access_tier=0)
            _write_findings(run_dir, findings)

            verdict_obj = _compute_verdict(findings, n_samples=result.total)
            assessment_id = report_build.make_assessment_id(run_id)
            result_dict = result.to_dict()
            record = report_build.assurance_record(
                assessment_id=assessment_id,
                verdict=verdict_obj,
                findings=findings,
                receipts_result=result_dict,
            )
            meta = {
                "id": run_id,
                "module": "receipts",
                "created": _dt.datetime.now().isoformat(timespec="seconds"),
                "assessment_id": assessment_id,
                "result": result_dict,
                "verdict": _verdict_dict(verdict_obj),
                "risk": report_build.overall_risk(verdict_obj),
                "confidence": report_build.overall_confidence(findings),
                "n_findings": len(findings),
            }
            _write_meta(run_dir, meta)
            _write_assurance_record(run_dir, record)
            _build_html_report(
                run_dir, findings,
                inputs={
                    "Inference log": f"{receipts_path.name} ({result.total} records)",
                    "Public key": pubkey_path.name,
                    "Network access": "none — this run stayed on this machine",
                },
                access_tier=0, verdict=verdict_obj, summary=record,
            )

            return jsonify({
                **meta,
                "findings": [_finding_to_json(f) for f in findings],
                "assurance_record": record,
            })
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(run_dir, ignore_errors=True)
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/receipts/runs")
    def receipts_runs():
        return jsonify(_list_runs("receipts"))

    @app.get("/api/receipts/runs/<run_id>")
    def receipts_run_detail(run_id: str):
        run_dir = _run_dir_or_404("receipts", run_id)
        if run_dir is None:
            return jsonify({"error": "no such run"}), 404
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        return jsonify({
            **meta,
            "findings": _read_findings(run_dir),
            "assurance_record": _read_assurance_record(run_dir),
        })

    # -- chain of custody -------------------------------------------------------

    @app.post("/api/custody/actors")
    def custody_create_actor():
        actor_id = secure_filename((request.form.get("actor_id") or "").strip())
        if not actor_id:
            return jsonify({"error": "give this organisation an id (letters, numbers, - or _)"}), 400
        actor_dir = CUSTODY_KEYS_DIR / actor_id
        if actor_dir.exists():
            return jsonify({
                "error": f"'{actor_id}' is already registered on this machine. Pick a "
                         f"different id, or use the existing one from the list below.",
            }), 400
        try:
            priv, pub = custody.init_actor_keys(actor_dir)
            return jsonify({
                "actor_id": actor_id,
                "public_key_pem": pub.read_text(encoding="utf-8"),
                "private_key_pem": priv.read_text(encoding="utf-8"),
            })
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(actor_dir, ignore_errors=True)
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/custody/actors")
    def custody_list_actors():
        if not CUSTODY_KEYS_DIR.exists():
            return jsonify([])
        out = []
        for sub in sorted(CUSTODY_KEYS_DIR.iterdir()):
            pub = sub / "pub.pem"
            if sub.is_dir() and pub.exists():
                out.append({"actor_id": sub.name, "public_key_pem": pub.read_text(encoding="utf-8")})
        return jsonify(out)

    @app.post("/api/custody/hops")
    def custody_add_hop():
        run_id, run_dir = _new_run_dir("custody")
        try:
            actor_id = (request.form.get("actor_id") or "").strip()
            stage = (request.form.get("stage") or "").strip()
            description = (request.form.get("description") or "").strip()
            if not actor_id or not stage or not description:
                return jsonify({"error": "actor id, stage and a description are all required"}), 400

            key_path = _resolve_file_source("actor_key", run_dir)
            if key_path is None:
                registered = CUSTODY_KEYS_DIR / secure_filename(actor_id) / "priv.pem"
                if not registered.exists():
                    return jsonify({
                        "error": f"'{actor_id}' is not a registered organisation on this "
                                 f"machine, and no private key was attached. Register it on "
                                 f"the Actors tab first, or attach its priv.pem.",
                    }), 400
                key_path = registered
            private_key = custody.load_private_key(key_path)

            chain_path = _resolve_file_source("chain", run_dir)
            chain = custody.CustodyChain.from_file(chain_path) if chain_path else custody.CustodyChain()

            dataset_source = _resolve_dataset_source("dataset", run_dir)
            model_path = _resolve_file_source("model", run_dir)
            output_digest = _custody_output_digest(
                dataset_source=dataset_source, model_path=model_path,
                explicit=request.form.get("output_digest") or None,
                images_only=request.form.get("images_only") == "true",
            )

            record = chain.add_hop(
                stage=stage, actor_id=actor_id, output_digest=output_digest,
                description=description, private_key=private_key,
            )
            chain.write(run_dir / "chain.jsonl")

            meta = {
                "id": run_id,
                "module": "custody",
                "kind": "hop",
                "created": _dt.datetime.now().isoformat(timespec="seconds"),
                "hop": record.hop,
                "stage": stage,
                "actor_id": actor_id,
                "description": description,
                "output_digest": output_digest,
                "n_hops": len(chain.records),
            }
            _write_meta(run_dir, meta)
            return jsonify({
                **meta,
                "chain": [r.to_dict() for r in chain.records],
            })
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(run_dir, ignore_errors=True)
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/custody/runs/<run_id>/chain.jsonl")
    def custody_chain_file(run_id: str):
        run_dir = _run_dir_or_404("custody", run_id)
        if run_dir is None or not (run_dir / "chain.jsonl").exists():
            return jsonify({"error": "no chain for this run"}), 404
        return send_file(run_dir / "chain.jsonl")

    @app.post("/api/custody/verify")
    def custody_verify():
        run_id, run_dir = _new_run_dir("custody")
        try:
            chain_path = _resolve_file_source("chain", run_dir)
            if chain_path is None:
                return jsonify({"error": "attach a chain .jsonl file, or give a path to one"}), 400

            keyring = custody.load_keyring_dir(CUSTODY_KEYS_DIR)
            keyring_path = _resolve_file_source("keyring", run_dir)
            if keyring_path is not None:
                keyring.update(custody.load_keyring_json(keyring_path))

            records = custody.read_custody_chain(chain_path)
            result = custody.verify_custody_chain(records, keyring)
            findings = custody.findings_from_custody_result(result)
            _write_findings(run_dir, findings)

            verdict_obj = _compute_verdict(findings, n_samples=result.total)
            assessment_id = report_build.make_assessment_id(run_id)
            result_dict = result.to_dict()
            record = report_build.assurance_record(
                assessment_id=assessment_id, verdict=verdict_obj, findings=findings,
                custody_result=result_dict,
            )
            meta = {
                "id": run_id,
                "module": "custody",
                "kind": "verify",
                "created": _dt.datetime.now().isoformat(timespec="seconds"),
                "assessment_id": assessment_id,
                "result": result_dict,
                "verdict": _verdict_dict(verdict_obj),
                "risk": report_build.overall_risk(verdict_obj),
                "confidence": report_build.overall_confidence(findings),
                "n_findings": len(findings),
                "known_actors": sorted(keyring),
            }
            _write_meta(run_dir, meta)
            _write_assurance_record(run_dir, record)
            _build_html_report(
                run_dir, findings,
                inputs={
                    "Chain of custody": f"{chain_path.name} ({result.total} hops)",
                    "Organisations recognised": ", ".join(sorted(keyring)) or "none",
                    "Network access": "none — this run stayed on this machine",
                },
                access_tier=0, verdict=verdict_obj, summary=record,
            )

            return jsonify({
                **meta,
                "records": records,
                "findings": [_finding_to_json(f) for f in findings],
                "assurance_record": record,
            })
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(run_dir, ignore_errors=True)
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/custody/runs")
    def custody_runs():
        return jsonify(_list_runs("custody"))

    @app.get("/api/custody/runs/<run_id>")
    def custody_run_detail(run_id: str):
        run_dir = _run_dir_or_404("custody", run_id)
        if run_dir is None:
            return jsonify({"error": "no such run"}), 404
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        extra: dict[str, Any] = {"assurance_record": _read_assurance_record(run_dir)}
        if meta.get("kind") == "hop" and (run_dir / "chain.jsonl").exists():
            extra["chain"] = custody.read_custody_chain(run_dir / "chain.jsonl")
        else:
            extra["findings"] = _read_findings(run_dir)
        return jsonify({**meta, **extra})

    # -- reports (across every module) ----------------------------------------

    @app.get("/api/reports")
    def reports():
        runs = []
        for module in MODULES:
            runs.extend(_list_runs(module))
        runs.sort(key=lambda r: r.get("created", ""), reverse=True)
        return jsonify(runs)

    @app.get("/api/reports/<module>/<run_id>/report.html")
    def report_html_file(module: str, run_id: str):
        if module not in MODULES:
            return jsonify({"error": "no such module"}), 404
        run_dir = _run_dir_or_404(module, run_id)
        if run_dir is None or not (run_dir / "report.html").exists():
            return jsonify({"error": "no report for this run"}), 404
        return send_file(run_dir / "report.html")

    @app.get("/api/reports/<module>/<run_id>/assurance_record.json")
    def assurance_record_file(module: str, run_id: str):
        if module not in MODULES:
            return jsonify({"error": "no such module"}), 404
        run_dir = _run_dir_or_404(module, run_id)
        record = run_dir and _read_assurance_record(run_dir)
        if not record:
            return jsonify({"error": "no assurance record for this run"}), 404
        return jsonify(record)

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="cvassure web console (local, offline)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    print(f"cvassure web console — http://{args.host}:{args.port}  (everything stays on this machine)")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
