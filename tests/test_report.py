"""Report, audit trail, offline guarantee and the end-to-end pipeline."""

from __future__ import annotations

import json
import re
from argparse import Namespace

import pytest

from cvassure.core.offline import NetworkAccessAttempted, allow_network, assert_offline
from cvassure.core.schemas import Finding
from cvassure.report import build as report_build
from cvassure.report import html as report_html
from cvassure.report.audit import AuditLog, verify_audit_log


def make_finding(**kw) -> Finding:
    base = dict(
        asset_ref="truck/img_0001.png",
        asset_type="sample",
        attack_class="ood_insertion",
        detector_id="ood",
        access_tier=0,
        raw_score=0.91,
        severity="critical",
        disposition="quarantine",
        reason="This image is 11 times further away than a typical truck photo.",
    )
    base.update(kw)
    return Finding(**base)


# --------------------------------------------------------------------------
# audit trail
# --------------------------------------------------------------------------


def _append(log, n=5):
    for i in range(n):
        log.append(
            detector_id=f"d{i}", config_hash="a" * 64, input_digests={"dataset": "x"},
            seed=0, started="2026-01-01T00:00:00", ended="2026-01-01T00:00:01",
            output_digest="b" * 64,
        )


def test_audit_log_verifies(tmp_path):
    log = AuditLog.open(tmp_path / "audit.jsonl")
    _append(log, 5)
    result = verify_audit_log(log.path)
    assert result.ok
    assert "All 5 audit records are intact" in result.render()


def test_audit_log_detects_an_edited_record(tmp_path):
    log = AuditLog.open(tmp_path / "audit.jsonl")
    _append(log, 5)
    lines = log.path.read_text().splitlines()
    entry = json.loads(lines[2])
    entry["seed"] = 999
    lines[2] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    log.path.write_text("\n".join(lines) + "\n")
    result = verify_audit_log(log.path)
    assert not result.ok
    assert "rewritten" in result.render()


def test_audit_log_detects_a_deleted_record(tmp_path):
    log = AuditLog.open(tmp_path / "audit.jsonl")
    _append(log, 5)
    lines = log.path.read_text().splitlines()
    del lines[2]
    log.path.write_text("\n".join(lines) + "\n")
    assert not verify_audit_log(log.path).ok


def test_audit_log_appends_across_sessions(tmp_path):
    p = tmp_path / "audit.jsonl"
    _append(AuditLog.open(p), 3)
    _append(AuditLog.open(p), 2)
    result = verify_audit_log(p)
    assert result.ok and result.total == 5


def test_missing_audit_log_is_reported(tmp_path):
    assert not verify_audit_log(tmp_path / "nope.jsonl").ok


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


def test_coverage_marks_unrun_checks_unsupported():
    coverage = report_build.build_coverage([make_finding()], access_tier=0)
    statuses = {r.attack_class: r.status for r in coverage.rows}
    assert statuses["model_backdoor"] == report_build.UNSUPPORTED
    assert statuses["ood_insertion"] == report_build.NOT_MEASURED


def test_coverage_uses_measured_numbers_when_it_has_them():
    coverage = report_build.build_coverage(
        [make_finding()],
        access_tier=0,
        measured={"ood_insertion": {"tpr_at_1pct": 0.93, "verdict": "strong"}},
    )
    row = [r for r in coverage.rows if r.attack_class == "ood_insertion"][0]
    assert row.status == report_build.SUPPORTED
    assert "93%" in row.measured


def test_a_weak_measurement_is_not_dressed_up_as_support():
    coverage = report_build.build_coverage(
        [make_finding()],
        access_tier=0,
        measured={"ood_insertion": {"tpr_at_1pct": 0.20, "verdict": "unsupported"}},
    )
    row = [r for r in coverage.rows if r.attack_class == "ood_insertion"][0]
    assert row.status == report_build.UNSUPPORTED


def test_coverage_statement_names_what_is_not_supported():
    coverage = report_build.build_coverage([make_finding()], access_tier=0)
    text = " ".join(coverage.statement())
    assert "cannot detect these" in text


def test_coverage_is_derived_not_hand_written():
    """If a detector reports UNAVAILABLE, coverage must say unsupported."""
    unavailable = Finding.unavailable(
        asset_ref="model", asset_type="model", attack_class="model_backdoor",
        detector_id="trigger_recon", access_tier=0,
        reason="We could not look inside the model.",
        unavailable_reason="requires access tier 2",
    )
    coverage = report_build.build_coverage([unavailable], access_tier=0)
    row = [r for r in coverage.rows if r.attack_class == "model_backdoor"][0]
    assert row.status == report_build.UNSUPPORTED
    assert "tier 2" in row.note


def test_measured_from_tables_reads_table1():
    class FakeTable:
        rows = [
            {"attack_class": "ood_insertion", "detector": "any detector",
             "TPR@1%FPR [95% CI]": "0.930 [0.90, 0.95]", "verdict": "strong"},
            {"attack_class": "ood_insertion", "detector": "ood",
             "TPR@1%FPR [95% CI]": "0.100 [0.05, 0.2]", "verdict": "unsupported"},
        ]

    out = report_build.measured_from_tables(FakeTable())
    assert out["ood_insertion"]["tpr_at_1pct"] == pytest.approx(0.93)


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------


def test_verdict_is_green_when_nothing_is_wrong():
    findings = [
        make_finding(asset_ref="C0", asset_type="contributor", raw_score=0.1,
                     severity="low", disposition="accept",
                     reason="Contributor C0 sent 500 images and 2 look wrong (0.4%)."),
    ]
    assert report_build.overall_verdict(findings).colour == "green"


def test_verdict_is_red_when_a_contributor_is_quarantined():
    findings = [
        make_finding(asset_ref="C3", asset_type="contributor",
                     reason="Contributor C3 sent 1200 images and 214 look tampered (18%)."),
    ]
    v = report_build.overall_verdict(findings)
    assert v.colour == "red"
    assert "QUARANTINE" in v.headline
    assert "C3" in v.render()


def test_verdict_names_a_substituted_model():
    findings = [
        make_finding(asset_ref="model", asset_type="model", attack_class="model_substitute",
                     detector_id="fingerprint",
                     reason="This is not the model that was accepted; 200 replies differ."),
    ]
    assert "SUBSTITUTED" in report_build.overall_verdict(findings).render()


def test_verdict_counts_tampered_receipts():
    findings = [
        make_finding(asset_ref="receipt:347", asset_type="receipt",
                     attack_class="receipt_alter", detector_id="provenance.verify",
                     reason="Inference record #347 does not check out; 1 field changed.",
                     evidence={"seq": 347}),
    ]
    assert "1 record tampered" in report_build.overall_verdict(findings).render()


def test_a_blocked_check_is_never_counted_as_a_problem():
    """An unavailable finding must not turn the light red — it is a status
    message, not a claim that something is wrong."""
    findings = [
        Finding.unavailable(
            asset_ref="model", asset_type="model", attack_class="model_backdoor",
            detector_id="trigger_recon", access_tier=0,
            reason="We could not look inside the model.",
            unavailable_reason="requires tier 2",
        )
    ]
    assert report_build.overall_verdict(findings).colour != "red"


def test_a_model_we_could_not_identify_is_never_called_verified_unchanged():
    """The bug this pins: with no enrolment record, fingerprint and
    weight_digest both go unavailable, and the headline used to fall through to
    'Model: verified unchanged.' — on a model that had in fact been swapped.
    A tool whose whole claim is honesty must not assert what it did not check."""
    findings = [
        Finding.unavailable(
            asset_ref="model", asset_type="model", attack_class="model_substitute",
            detector_id="fingerprint", access_tier=1,
            reason="We had nothing to compare this model's answers against.",
            unavailable_reason="no enrolled fingerprint supplied",
        ),
        Finding.unavailable(
            asset_ref="model", asset_type="model", attack_class="model_tamper",
            detector_id="weight_digest", access_tier=1,
            reason="We had no recorded weights to compare against.",
            unavailable_reason="no enrolled weight digest supplied",
        ),
        # This one ran and found nothing — but it cannot establish identity.
        make_finding(asset_ref="model", asset_type="model",
                     attack_class="model_tamper", detector_id="weight_stats",
                     raw_score=0.1, severity="low", disposition="accept",
                     reason="The last layer looks ordinary; no class dominates "
                            "more than 1.4 times the average."),
    ]
    v = report_build.overall_verdict(findings)
    rendered = v.render()
    assert "verified unchanged" not in rendered
    assert "NOT VERIFIED" in rendered
    assert "no enrolled fingerprint supplied" in rendered
    assert v.colour == "amber", "green would read as 'the model is fine'"


def test_a_model_that_was_actually_checked_is_still_called_verified():
    findings = [
        make_finding(asset_ref="model", asset_type="model",
                     attack_class="model_substitute", detector_id="fingerprint",
                     raw_score=0.0, severity="low", disposition="accept",
                     reason="All 200 recorded answers match to 15 decimal places."),
    ]
    v = report_build.overall_verdict(findings)
    assert "verified unchanged" in v.render()
    assert v.colour == "green"


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------


@pytest.fixture
def rendered(tmp_path):
    findings = [
        make_finding(),
        make_finding(asset_ref="C3", asset_type="contributor", detector_id="contributor",
                     reason="Contributor C3 sent 1,200 images and 214 look tampered with "
                            "(about 18%). Recommend: quarantine all of C3's data.",
                     evidence={"n_samples": 1200, "n_flagged": 214, "flagged_rate": 0.178,
                               "credible_interval_95": [0.157, 0.201]}),
        make_finding(asset_ref="model", asset_type="model", detector_id="fingerprint",
                     attack_class="model_substitute",
                     reason="This is not the model that was accepted; its replies to 200 "
                            "fixed questions differ by 32%."),
        Finding.unavailable(
            asset_ref="model", asset_type="model", attack_class="model_backdoor",
            detector_id="trigger_recon", access_tier=0,
            reason="We could not search for a hidden trigger, because that needs to "
                   "look inside the model as it runs.",
            unavailable_reason="requires access tier 2"),
    ]
    coverage = report_build.build_coverage(findings, access_tier=0)
    verdict = report_build.overall_verdict(findings)
    path = report_html.write(
        tmp_path / "report.html", findings=findings, verdict=verdict, coverage=coverage,
        inputs={"Dataset": "data/contributed (3,000 images)", "Access granted": "tier 0"},
        limitations=["source-level assessment unavailable for 12 images"],
    )
    return path, path.read_text(encoding="utf-8")


def test_report_is_a_single_file(rendered):
    path, _ = rendered
    assert list(path.parent.iterdir()) == [path]


def test_report_has_no_external_references(rendered):
    _, text = rendered
    assert "http://" not in text
    assert "https://" not in text
    for pattern in (r'src="(?!data:)', r'href="(?!#)', r"@import"):
        assert not re.search(pattern, text), pattern


def test_report_leads_with_the_verdict(rendered):
    _, text = rendered
    assert text.index("QUARANTINE RECOMMENDED") < text.index("can and cannot detect")


def test_report_states_what_it_cannot_do(rendered):
    _, text = rendered
    assert "cannot detect" in text
    assert "Limitations and assumptions" in text
    assert "could not search for a hidden trigger" in text


def test_report_includes_the_contributor_recommendation(rendered):
    _, text = rendered
    assert "quarantine all of C3" in text
    assert "15.7% – 20.1%" in text


def test_report_is_sortable_without_a_library(rendered):
    _, text = rendered
    assert "addEventListener('click'" in text
    assert "<script" in text and "src=" not in text.split("<script")[1][:80]


def test_report_avoids_jargon(rendered):
    _, text = rendered
    from cvassure.core.schemas import BANNED_JARGON

    body = text.lower()
    for phrase in BANNED_JARGON:
        assert phrase not in body, phrase


def test_report_embeds_images_as_data_uris(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    img = tmp_path / "plot.png"
    fig.savefig(img)
    plt.close(fig)

    src = report_html.embed_image(img)
    assert src.startswith("data:image/png;base64,")


def test_missing_image_does_not_break_the_report(tmp_path):
    assert report_html.embed_image(tmp_path / "absent.png") is None


def test_report_escapes_content_from_the_data(tmp_path):
    """A filename is attacker-controlled; it must not become markup."""
    nasty = make_finding(asset_ref="<script>alert(1)</script>.png")
    path = report_html.write(
        tmp_path / "r.html", findings=[nasty],
        verdict=report_build.overall_verdict([nasty]),
        coverage=report_build.build_coverage([nasty], access_tier=0),
        inputs={},
    )
    text = path.read_text()
    assert "<script>alert(1)</script>.png" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.png" in text


# --------------------------------------------------------------------------
# offline guarantee
# --------------------------------------------------------------------------


def test_offline_assert_blocks_sockets():
    import socket

    try:
        assert_offline()
        with pytest.raises(NetworkAccessAttempted):
            socket.socket()
        with pytest.raises(NetworkAccessAttempted):
            socket.create_connection(("example.com", 80))
        with pytest.raises(NetworkAccessAttempted):
            socket.getaddrinfo("example.com", 80)
    finally:
        allow_network()


def test_offline_assert_is_idempotent():
    try:
        assert_offline()
        assert_offline()
    finally:
        allow_network()


def test_cli_offline_flag_runs_an_audit(synth_root, capsys):
    from cvassure.cli import main

    try:
        rc = main(["--offline-assert", "ingest", "inspect", "--dataset", str(synth_root)])
        assert rc == 0
        assert "DATASET" in capsys.readouterr().out
    finally:
        allow_network()


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


@pytest.fixture
def audited(tmp_path, synth_root, toy_models):
    from cvassure.detect import model as model_suite
    from cvassure.ingest.models import load_model
    from cvassure.pipeline import run_audit

    enrolled = tmp_path / "enrolled.json"
    enrolled.write_text(
        json.dumps(model_suite.enrol(load_model(toy_models["onnx"], 2))), encoding="utf-8"
    )
    out = tmp_path / "out"
    rc = run_audit(
        Namespace(
            dataset=str(synth_root), model=str(toy_models["substitute_onnx"]),
            access_tier=1, receipts=None, pubkey="keys/pub.pem", reference=None,
            enrolled_fingerprint=str(enrolled), format="auto", contributors=None,
            contributor_from_path=None, out=str(out), seed=0, quiet=True,
        )
    )
    return rc, out


def test_audit_writes_everything_it_promises(audited):
    rc, out = audited
    assert rc in (0, 2)
    for name in ("report.html", "findings.jsonl", "audit_log.jsonl", "timings.json"):
        assert (out / name).exists(), name


def test_audit_trail_verifies_after_a_real_run(audited):
    _, out = audited
    result = verify_audit_log(out / "audit_log.jsonl")
    assert result.ok
    assert result.total >= 8


def test_audit_catches_the_swapped_model(audited):
    from cvassure.score.evaluate import read_findings

    _, out = audited
    findings = read_findings(out / "findings.jsonl")
    fingerprint = [f for f in findings if f.detector_id == "fingerprint"]
    assert fingerprint and fingerprint[0].disposition == "quarantine"


def test_audit_report_opens_as_one_file(audited):
    _, out = audited
    text = (out / "report.html").read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "https://" not in text


def test_audit_records_timings_for_every_detector(audited):
    _, out = audited
    timings = json.loads((out / "timings.json").read_text())
    assert {t["module"] for t in timings} >= {"ood", "near_duplicate", "contributor"}


def test_audit_is_deterministic(tmp_path, synth_root, toy_models):
    from cvassure.pipeline import run_audit
    from cvassure.score.evaluate import read_findings

    def once(tag):
        run_audit(
            Namespace(
                dataset=str(synth_root), model=None, access_tier=0, receipts=None,
                pubkey="keys/pub.pem", reference=None, enrolled_fingerprint=None,
                format="auto", contributors=None, contributor_from_path=None,
                out=str(tmp_path / tag), seed=0, quiet=True,
            )
        )
        return [
            (f.asset_ref, f.detector_id, round(f.raw_score, 9))
            for f in read_findings(tmp_path / tag / "findings.jsonl")
        ]

    assert once("a") == once("b")


def test_audit_survives_a_black_box_model(tmp_path, synth_root, toy_models):
    from cvassure.pipeline import run_audit

    rc = run_audit(
        Namespace(
            dataset=str(synth_root), model=str(toy_models["onnx"]), access_tier=0,
            receipts=None, pubkey="keys/pub.pem", reference=None,
            enrolled_fingerprint=None, format="auto", contributors=None,
            contributor_from_path=None, out=str(tmp_path / "bb"), seed=0, quiet=True,
        )
    )
    assert rc in (0, 2)
    assert (tmp_path / "bb" / "report.html").exists()


# --------------------------------------------------------------------------
# quoting the results run
# --------------------------------------------------------------------------


def _sweep_file(tmp_path, rows):
    p = tmp_path / "sweep_raw.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_measured_from_sweep_grades_on_the_median_not_the_best(tmp_path):
    """A detector that usually catches a fifth must not be graded on its one
    lucky cell."""
    rows = [
        {"attack": "blended_trigger", "tpr_at_1pct": v}
        for v in (0.0, 0.1, 0.2, 0.25, 0.67)
    ]
    measured = report_build.measured_from_sweep(_sweep_file(tmp_path, rows))
    assert measured["blended_trigger"]["tpr_at_1pct"] == pytest.approx(0.2)
    assert measured["blended_trigger"]["verdict"] == "unsupported"
    assert measured["blended_trigger"]["n_cells"] == 5


def test_measured_from_sweep_skips_failed_and_missing_cells(tmp_path):
    rows = [
        {"attack": "ood_insertion", "tpr_at_1pct": 1.0},
        {"attack": "ood_insertion", "tpr_at_1pct": 0.9},
        {"attack": "ood_insertion", "error": "boom"},
        {"attack": "ood_insertion", "tpr_at_1pct": None},
    ]
    measured = report_build.measured_from_sweep(_sweep_file(tmp_path, rows))
    assert measured["ood_insertion"]["n_cells"] == 2


def test_measured_from_sweep_is_empty_without_a_results_run(tmp_path):
    assert report_build.measured_from_sweep(tmp_path / "absent.jsonl") == {}


def test_coverage_quotes_the_sweep_end_to_end(tmp_path):
    rows = [{"attack": "ood_insertion", "tpr_at_1pct": 0.97} for _ in range(3)]
    coverage = report_build.build_coverage(
        [make_finding()],
        access_tier=0,
        measured=report_build.measured_from_sweep(_sweep_file(tmp_path, rows)),
    )
    row = [r for r in coverage.rows if r.attack_class == "ood_insertion"][0]
    assert row.status == report_build.SUPPORTED
    assert "97%" in row.measured


# --------------------------------------------------------------------------
# tamper detection must be 100%
# --------------------------------------------------------------------------


def test_every_receipt_attack_is_caught_and_correctly_named(tmp_path):
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]
                          / "experiments"))
    from run_all import run_tamper_cells

    rows = run_tamper_cells(tmp_path, seeds=(1, 2), n_receipts=60)
    attacks = [r for r in rows if r["attack_class"] != "clean"]
    controls = [r for r in rows if r["attack_class"] == "clean"]

    assert len(attacks) == 8  # four attack types, two seeds
    assert all(r["detected"] for r in attacks), "a tampering attempt went undetected"
    assert all(r["correct_code"] for r in attacks), "the wrong failure mode was named"
    assert controls and all(r["correct_code"] for r in controls), (
        "an untouched log failed verification — every other row is meaningless"
    )


def test_table3_reports_a_hundred_percent(tmp_path):
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]
                          / "experiments"))
    from aggregate import table3
    from run_all import run_tamper_cells

    t = table3(run_tamper_cells(tmp_path, seeds=(1,), n_receipts=60))
    attack_rows = [r for r in t.rows if not r["attack type"].startswith("clean")]
    assert len(attack_rows) == 4
    assert all(r["rate"] == "100.0%" for r in attack_rows)
    assert all(r["named the right failure"] == "100.0%" for r in attack_rows)


def test_the_sweep_figure_is_not_overwritten_by_the_representative_cell(tmp_path):
    """Figures 5 and 7 describe the whole sweep. A single representative cell
    renders them empty, and copying those over would blank the most important
    figure in the project."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]
                          / "experiments"))
    import aggregate
    import inspect as _inspect

    source = _inspect.getsource(aggregate._render_remaining_figures)
    assert "fig5_sweep" not in source, (
        "the representative cell must not be allowed to write fig5_sweep"
    )
    assert "fig7_runtime" not in source
    assert "joined_figures" in source


def test_every_numbered_table_reaches_the_results_directory():
    """Table 2 is built from the representative cell rather than the sweep, and
    for one run of this project it landed only under `representative/` — so
    `results/tables/` held tables 1, 3 and 4 and looked like a table was
    missing. The promotion step is what stops that, so it is pinned here."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]
                          / "experiments"))
    import aggregate
    import inspect as _inspect

    source = _inspect.getsource(aggregate._render_remaining_figures)
    assert "table2_contributors" in source, (
        "the contributor table must be promoted into results/tables/, not left "
        "behind in results/representative/"
    )

    main_source = _inspect.getsource(aggregate.main)
    assert "t2_md" in main_source, "Table 2 must appear in RESULTS.md as well"


# --------------------------------------------------------------------------
# table layout — the explanation column must not be crushed
# --------------------------------------------------------------------------


def test_the_explanation_column_is_marked_so_css_can_give_it_room():
    """Without this the browser squeezes the 'why' column to a strip, wraps one
    sentence over fifteen lines, and leaves a tall blank band beside it."""
    html = report_html._table(
        ["image", "check", "score", "why"],
        [["a.png", "ood", "0.9", "This image is 11 times further away than usual."]],
    )
    assert '<td class="prose">This image is 11 times' in html
    assert '<th class="prose">why</th>' in html
    assert '<td class="narrow">a.png</td>' in html


@pytest.mark.parametrize("heading", ["why", "what we found", "note", "Why", "  NOTE "])
def test_known_prose_headings_are_recognised(heading):
    html = report_html._table([heading, "score"], [["some long sentence here", "0.9"]])
    assert f'<th class="prose">{report_html.esc(heading)}</th>' in html


def test_an_unnamed_free_text_column_is_still_given_room():
    """A table whose long column has an unrecognised heading must still lay out
    sensibly rather than falling back to equal widths."""
    long_text = "x" * 120
    html = report_html._table(["id", "commentary"], [["a", long_text]])
    assert f'<td class="prose">{long_text}</td>' in html


def test_short_tables_do_not_get_a_prose_column():
    html = report_html._table(["a", "b"], [["1", "2"], ["3", "4"]])
    assert "prose" not in html


def test_the_stylesheet_lets_tables_scroll_instead_of_crushing_columns():
    css = report_html.CSS
    assert "min-width:940px" in css, "the table must be allowed to exceed a narrow window"
    assert "overflow-x:auto" in css, "and the container must scroll when it does"
    assert "td.prose" in css and "min-width:340px" in css
    assert "td:first-child{overflow-wrap:anywhere}" in css, (
        "long filenames must break rather than widening the whole table"
    )


# --------------------------------------------------------------------------
# PS 2.2.2: access assumptions, confidence and limitations
# --------------------------------------------------------------------------


def test_a_finding_can_carry_confidence_and_limitations():
    f = make_finding(confidence=0.45,
                     limitations=["reports a shape, not proof"])
    assert f.confidence == pytest.approx(0.45)
    assert f.limitations == ["reports a shape, not proof"]
    assert Finding.from_dict(f.to_dict()) == f


def test_confidence_is_range_checked():
    with pytest.raises(ValueError):
        make_finding(confidence=1.5)


def test_model_findings_state_confidence_and_limits(toy_models, synth_root, tmp_path):
    """Every model check must say how sure it is and what it could not settle."""
    from cvassure.detect import model as model_suite
    from cvassure.detect.base import AuditContext
    from cvassure.detect.embed import EmbeddingStore
    from cvassure.ingest.dataset import load_dataset
    from cvassure.ingest.models import load_model

    original = load_model(toy_models["onnx"], access_tier=2)
    enrolled = model_suite.enrol(original)
    ctx = AuditContext(
        samples=load_dataset(synth_root).samples,
        access_tier=2,
        model=load_model(toy_models["substitute_onnx"], access_tier=2),
        embeddings=EmbeddingStore(cache_dir=None),
        enrolled_fingerprint=enrolled,
        out_dir=tmp_path,
    )
    findings, _ = model_suite.run_all(ctx)
    live = [f for f in findings if not f.is_unavailable]
    assert live, "no model check produced a live finding"
    for f in live:
        assert f.confidence is not None, f"{f.detector_id} states no confidence"
        assert f.limitations, f"{f.detector_id} states no limitations"


def test_the_report_shows_confidence_and_limitations(tmp_path):
    f = make_finding(asset_ref="model", asset_type="model", detector_id="weight_stats",
                     attack_class="model_backdoor", confidence=0.45,
                     limitations=["A lopsided layer can come from class imbalance."],
                     reason="Class 3 pulls 4.1 times harder than a typical class.")
    path = report_html.write(
        tmp_path / "r.html", findings=[f],
        verdict=report_build.overall_verdict([f]),
        coverage=report_build.build_coverage([f], access_tier=1), inputs={})
    text = path.read_text()
    assert "confidence" in text and "45%" in text
    assert "what this check cannot establish" in text
    assert "class imbalance" in text


# --------------------------------------------------------------------------
# PS 2.3: the assurance-report schema is a deliverable
# --------------------------------------------------------------------------


def test_schema_is_generated_and_matches_the_dataclass(tmp_path):
    from cvassure.core.schema_export import finding_schema, write

    paths = write(tmp_path)
    assert all(p.exists() for p in paths.values())

    schema = finding_schema()
    documented = set(schema["properties"])
    actual = set(Finding.__dataclass_fields__)
    assert documented == actual, (
        f"schema and dataclass disagree: only in schema {documented - actual}, "
        f"only in code {actual - documented}"
    )


def test_every_real_finding_validates_against_the_published_schema(
        synth_root, toy_models, tmp_path):
    """A schema nothing validates against is decoration. Run a real audit and
    check every finding it emits against the published contract."""
    jsonschema = pytest.importorskip("jsonschema")
    from cvassure.core.schema_export import finding_schema
    from cvassure.pipeline import run_audit
    from cvassure.score.evaluate import read_findings

    out = tmp_path / "out"
    run_audit(Namespace(
        dataset=str(synth_root), model=str(toy_models["onnx"]), access_tier=2,
        receipts=None, pubkey="keys/pub.pem", reference=None,
        enrolled_fingerprint=None, format="auto", contributors=None,
        contributor_from_path=None, out=str(out), seed=0, quiet=True))

    schema = finding_schema()
    findings = read_findings(out / "findings.jsonl")
    assert findings
    for f in findings:
        jsonschema.validate(f.to_dict(), schema)
