"""Detectors, measured against the Phase 4 answer key.

Every performance assertion here goes through the Phase 5 harness, so a
detector that stops working fails the build with a number attached.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvassure.attack.make_poison import load_ground_truth, run_config
from cvassure.core.schemas import Finding
from cvassure.detect import data as data_suite
from cvassure.detect import model as model_suite
from cvassure.detect.base import AuditContext
from cvassure.detect.embed import (
    BundledTorchEncoder,
    EmbeddingStore,
    HandcraftedEncoder,
    cosine_matrix,
    knn,
    load_encoder,
    robust_z,
    squash,
)
from cvassure.detect.fingerprint import compute_fingerprint
from cvassure.detect.label_noise import LabelNoiseDetector
from cvassure.detect.near_duplicate import NearDuplicateDetector
from cvassure.detect.ood import OODDetector
from cvassure.detect.shift import ShiftDetector, build_profile, ks_with_bh
from cvassure.detect.trigger_freq import TriggerFrequencyDetector
from cvassure.detect.weight_digest import WeightDigestDetector, canonical_weight_digest
from cvassure.detect.weight_stats import WeightStatsDetector
from cvassure.ingest.dataset import load_dataset
from cvassure.ingest.models import load_model
from cvassure.score import metrics

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "attacks"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def store(tmp_path_factory) -> EmbeddingStore:
    return EmbeddingStore(cache_dir=tmp_path_factory.mktemp("emb"))


def make_ctx(root, store, tmp_path, **kw) -> AuditContext:
    ds = load_dataset(root)
    return AuditContext(
        samples=ds.samples,
        embeddings=store,
        out_dir=tmp_path / "out",
        access_tier=kw.pop("access_tier", 0),
        **kw,
    )


def poisoned(tmp_path, synth_root, config, ood_pool=None, seed=None, monkeypatch=None):
    """Build a poisoned dataset and return (context inputs, truth)."""
    if ood_pool is not None and monkeypatch is not None:
        monkeypatch.chdir(tmp_path)
        import shutil

        Path("data").mkdir(exist_ok=True)
        if not Path("data/ood_pool").exists():
            shutil.copytree(ood_pool, "data/ood_pool")
    run_config(CONFIGS / config, synth_root, tmp_path / "p", tmp_path / "t", seed=seed)
    return tmp_path / "p", load_ground_truth(tmp_path / "t")


def auroc_of(findings, truth, detector_id=None):
    """Score findings against the answer key, exactly as Phase 5 does."""
    labels = {r["sample_id"]: r["is_poisoned"] for r in truth["samples"]}
    best = {sid: 0.0 for sid in labels}
    for f in findings:
        if f.asset_type != "sample" or f.is_unavailable or f.asset_ref not in best:
            continue
        if detector_id and f.detector_id != detector_id:
            continue
        best[f.asset_ref] = max(best[f.asset_ref], f.raw_score)
    order = sorted(best)
    y = np.array([labels[i] for i in order])
    s = np.array([best[i] for i in order])
    return metrics.roc_auc(y, s), y, s


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------


def test_bundled_encoder_is_present_in_the_repo():
    from cvassure.detect.embed import BUNDLED_ENCODER

    assert BUNDLED_ENCODER.exists(), (
        "the image encoder must be bundled in the repository — the audit is "
        "required to run air-gapped"
    )


def test_load_encoder_prefers_the_bundled_neural_one():
    assert isinstance(load_encoder("auto"), BundledTorchEncoder)


def test_handcrafted_encoder_needs_no_weights(synth_root):
    enc = HandcraftedEncoder()
    paths = [str(p) for p in sorted(synth_root.rglob("*.png"))[:6]]
    X = enc.encode(paths)
    assert X.shape == (6, enc.dim)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)


@pytest.mark.parametrize("encoder", [HandcraftedEncoder(), None])
def test_same_class_images_are_closer_than_different_class(synth_root, encoder, tmp_path):
    store = EmbeddingStore(cache_dir=None, encoder=encoder)
    ds = load_dataset(synth_root)
    by_label = {}
    for s in ds:
        by_label.setdefault(s.label, []).append(s)
    labels = sorted(by_label)[:2]
    a = store.embed([s.image_path for s in by_label[labels[0]][:6]])
    b = store.embed([s.image_path for s in by_label[labels[1]][:6]])
    within = cosine_matrix(a).mean()
    between = cosine_matrix(a, b).mean()
    assert within > between


def test_cache_is_keyed_by_content_not_by_path(synth_root, tmp_path):
    store = EmbeddingStore(cache_dir=tmp_path / "cache")
    paths = [str(p) for p in sorted(synth_root.rglob("*.png"))[:3]]
    first = store.embed(paths)
    import shutil

    copies = []
    for i, p in enumerate(paths):
        dest = tmp_path / f"copy_{i}.png"
        shutil.copy2(p, dest)
        copies.append(str(dest))
    assert np.allclose(store.embed(copies), first)


def test_cache_survives_a_new_store(synth_root, tmp_path):
    paths = [str(p) for p in sorted(synth_root.rglob("*.png"))[:4]]
    a = EmbeddingStore(cache_dir=tmp_path / "c").embed(paths)
    b = EmbeddingStore(cache_dir=tmp_path / "c").embed(paths)
    assert np.allclose(a, b)


def test_knn_excludes_self():
    X = np.eye(5)
    idx, sims = knn(X, 2)
    assert idx.shape == (5, 2)
    assert all(i not in idx[i] for i in range(5))


def test_squash_is_monotone_and_bounded():
    v = squash(np.linspace(-10, 20, 50))
    assert np.all(np.diff(v) > 0)
    assert v.min() >= 0 and v.max() <= 1


def test_robust_z_ignores_a_few_extremes():
    clean = np.r_[np.random.default_rng(0).normal(0, 1, 200)]
    with_outliers = np.r_[clean, np.full(5, 500.0)]
    assert abs(np.median(robust_z(with_outliers)[:200])) < 0.5


# --------------------------------------------------------------------------
# near duplicate — expected to exceed 0.95 AUROC
# --------------------------------------------------------------------------


def test_near_duplicate_catches_a_flood(synth_root, store, tmp_path):
    """Recall on the copies must be near-perfect.

    AUROC on this attack has a ceiling below 1.0 that has nothing to do with
    how good the detector is: the harness copies an existing photograph, so
    every cluster contains one image that ground truth calls clean. The
    detector flags the whole cluster — which is the right behaviour, since it
    cannot know which member came first, and an analyst wants to see all of
    them — and those source photographs are counted as false alarms. So we
    measure recall on the copies directly, and check AUROC against the ceiling
    that structure allows rather than against 1.0.
    """
    root, truth = poisoned(tmp_path, synth_root, "06_duplicate_flood.yaml")
    ctx = make_ctx(root, store, tmp_path)
    findings = NearDuplicateDetector().safe_run(ctx).findings
    auc, y, s = auroc_of(findings, truth)
    assert y.sum() > 0

    # recall: what fraction of the planted copies did we flag at all?
    recall = float((s[y == 1] > 0.5).mean())
    assert recall > 0.95, f"missed {100 * (1 - recall):.0f}% of the planted copies"

    n_seeds = 8  # from configs/attacks/06_duplicate_flood.yaml
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    ceiling = ((n_neg - n_seeds) * n_pos + n_seeds * n_pos * 0.5) / (n_pos * n_neg)
    assert auc > 0.90, f"near-duplicate AUROC {auc:.3f} (ceiling {ceiling:.3f})"
    assert auc > ceiling - 0.05, f"AUROC {auc:.3f} is well short of ceiling {ceiling:.3f}"


def test_near_duplicate_flags_the_source_photo_with_its_copies(synth_root, store, tmp_path):
    """Documents the behaviour behind the ceiling above: the detector reports a
    cluster, and the original is part of the cluster."""
    root, truth = poisoned(tmp_path, synth_root, "06_duplicate_flood.yaml")
    ctx = make_ctx(root, store, tmp_path)
    findings = NearDuplicateDetector().safe_run(ctx).findings
    labels = {r["sample_id"]: r["is_poisoned"] for r in truth["samples"]}

    flagged_clean = {
        f.asset_ref for f in findings
        if f.disposition != "accept" and labels.get(f.asset_ref) == 0
    }
    # every one of them is a photograph that has copies named after it
    for sid in flagged_clean:
        stem = sid.rsplit(".", 1)[0]
        assert any(k.startswith(f"{stem}_dup") for k in labels), sid


def test_near_duplicate_stays_quiet_on_clean_data(synth_root, store, tmp_path):
    ctx = make_ctx(synth_root, store, tmp_path)
    findings = NearDuplicateDetector().safe_run(ctx).findings
    flagged = [f for f in findings if f.disposition != "accept"]
    assert len(flagged) < 0.05 * len(ctx.samples)


def test_near_duplicate_evidence_names_the_cluster(synth_root, store, tmp_path):
    root, _ = poisoned(tmp_path, synth_root, "06_duplicate_flood.yaml")
    ctx = make_ctx(root, store, tmp_path)
    findings = [f for f in NearDuplicateDetector().safe_run(ctx).findings
                if f.disposition != "accept"]
    assert findings
    e = findings[0].evidence
    assert e["cluster_size"] >= 3
    assert e["members"] and e["exemplar"]
    assert "contributor_breakdown" in e


# --------------------------------------------------------------------------
# OOD — expected to exceed 0.95 AUROC
# --------------------------------------------------------------------------


def test_ood_catches_foreign_images(synth_root, ood_pool, store, tmp_path, monkeypatch):
    root, truth = poisoned(tmp_path, synth_root, "07_ood_insertion.yaml",
                           ood_pool=ood_pool, monkeypatch=monkeypatch)
    ctx = make_ctx(root, store, tmp_path)
    findings = OODDetector().safe_run(ctx).findings
    auc, y, s = auroc_of(findings, truth)
    assert y.sum() > 0
    assert auc > 0.95, f"OOD AUROC {auc:.3f}"


def test_ood_reason_is_readable_and_quantified(synth_root, ood_pool, store, tmp_path,
                                               monkeypatch):
    root, _ = poisoned(tmp_path, synth_root, "07_ood_insertion.yaml",
                       ood_pool=ood_pool, monkeypatch=monkeypatch)
    ctx = make_ctx(root, store, tmp_path)
    flagged = [f for f in OODDetector().safe_run(ctx).findings
               if f.disposition == "quarantine"]
    assert flagged
    reason = flagged[0].reason
    assert "times further away" in reason
    assert "looks nothing like" in reason


def test_ood_declines_to_judge_a_tiny_class(store, tmp_path):
    from PIL import Image

    root = tmp_path / "tiny"
    for label in ("a", "b"):
        (root / label).mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (32, 32), (i * 20, 40, 90)).save(root / label / f"{i}.png")
    ctx = make_ctx(root, store, tmp_path)
    findings = OODDetector().safe_run(ctx).findings
    assert all(f.is_unavailable for f in findings)
    assert "handful" in findings[0].reason


# --------------------------------------------------------------------------
# label noise
# --------------------------------------------------------------------------


def test_label_noise_catches_flips(synth_root, store, tmp_path):
    root, truth = poisoned(tmp_path, synth_root, "04_label_flip_10pct.yaml")
    ctx = make_ctx(root, store, tmp_path)
    findings = LabelNoiseDetector().safe_run(ctx).findings
    auc, y, s = auroc_of(findings, truth)
    assert y.sum() > 0
    assert auc > 0.85, f"label-noise AUROC {auc:.3f}"


def test_label_noise_evidence_lists_the_neighbours(synth_root, store, tmp_path):
    root, _ = poisoned(tmp_path, synth_root, "04_label_flip_10pct.yaml")
    ctx = make_ctx(root, store, tmp_path)
    flagged = [f for f in LabelNoiseDetector().safe_run(ctx).findings
               if f.disposition != "accept"]
    assert flagged
    e = flagged[0].evidence
    assert len(e["neighbour_labels"]) == e["k"]
    assert 0 <= e["agreement_fraction"] <= 1
    assert "the label is the part that looks wrong" in flagged[0].reason


def test_label_noise_needs_more_than_one_class(store, tmp_path):
    from PIL import Image

    root = tmp_path / "one"
    (root / "only").mkdir(parents=True)
    for i in range(25):
        Image.new("RGB", (32, 32), (i, i, i)).save(root / "only" / f"{i}.png")
    findings = LabelNoiseDetector().safe_run(make_ctx(root, store, tmp_path)).findings
    assert findings[0].is_unavailable


# --------------------------------------------------------------------------
# trigger frequency
# --------------------------------------------------------------------------


def test_trigger_freq_catches_a_visible_patch(synth_root, store, tmp_path):
    root, truth = poisoned(tmp_path, synth_root, "01_badnets_5pct.yaml")
    ctx = make_ctx(root, store, tmp_path)
    findings = TriggerFrequencyDetector().safe_run(ctx).findings
    auc, y, s = auroc_of(findings, truth)
    assert y.sum() > 0
    assert auc > 0.85, f"trigger-frequency AUROC {auc:.3f}"


def test_trigger_freq_saves_a_heatmap(synth_root, store, tmp_path):
    root, _ = poisoned(tmp_path, synth_root, "01_badnets_5pct.yaml")
    ctx = make_ctx(root, store, tmp_path)
    findings = TriggerFrequencyDetector().safe_run(ctx).findings
    with_art = [f for f in findings if f.artefacts]
    assert with_art
    assert Path(with_art[0].artefacts[0]).exists()


def test_trigger_freq_reports_how_many_images_share_the_hot_spot(synth_root, store, tmp_path):
    root, _ = poisoned(tmp_path, synth_root, "01_badnets_5pct.yaml")
    ctx = make_ctx(root, store, tmp_path)
    flagged = [f for f in TriggerFrequencyDetector().safe_run(ctx).findings
               if f.raw_score > 0.6]
    assert flagged
    assert "images_sharing_this_hot_spot" in flagged[0].evidence


def test_blended_trigger_is_harder_than_a_patch(synth_root, store, tmp_path):
    """Recorded honestly: we expect this to be the weaker row in the table."""
    patch_root, patch_truth = poisoned(tmp_path / "a", synth_root, "01_badnets_5pct.yaml")
    blend_root, blend_truth = poisoned(tmp_path / "b", synth_root, "03_blended_faint.yaml")
    det = TriggerFrequencyDetector()
    patch_auc, _, _ = auroc_of(
        det.safe_run(make_ctx(patch_root, store, tmp_path)).findings, patch_truth
    )
    blend_auc, _, _ = auroc_of(
        det.safe_run(make_ctx(blend_root, store, tmp_path)).findings, blend_truth
    )
    assert patch_auc >= blend_auc


# --------------------------------------------------------------------------
# contributor aggregation
# --------------------------------------------------------------------------


def test_contributor_names_the_bad_actor(synth_root, ood_pool, store, tmp_path, monkeypatch):
    root, truth = poisoned(tmp_path, synth_root, "08_mixed_realistic.yaml",
                           ood_pool=ood_pool, monkeypatch=monkeypatch)
    ctx = make_ctx(root, store, tmp_path)
    findings, _ = data_suite.run_all(ctx)
    contributor_findings = [f for f in findings if f.asset_type == "contributor"]
    assert contributor_findings

    true_rates = {}
    for r in truth["samples"]:
        cid = r["contributor_id"]
        if cid:
            true_rates.setdefault(cid, []).append(r["is_poisoned"])
    worst = max(true_rates, key=lambda c: np.mean(true_rates[c]))

    ranked = sorted(contributor_findings, key=lambda f: -f.raw_score)
    assert ranked[0].asset_ref == worst, (
        f"expected {worst} to rank first, got "
        f"{[(f.asset_ref, round(f.raw_score, 3)) for f in ranked]}"
    )


def test_contributor_reason_is_actionable(synth_root, store, tmp_path):
    """A quarantine recommendation needs both confidence and enough images
    behind it; the rule requires at least 30, so this contributor gets them."""
    from cvassure.core.schemas import Sample
    from cvassure.detect.contributor import ContributorDetector

    ctx = make_ctx(synth_root, store, tmp_path)
    # give C0 a large enough share for the quarantine rule to be applicable
    ctx.samples = [
        Sample(s.sample_id, s.image_path, s.label,
               "C0" if i % 2 == 0 else s.contributor_id, s.batch_id)
        for i, s in enumerate(ctx.samples)
    ]
    fake = [
        Finding(
            asset_ref=s.sample_id,
            asset_type="sample",
            attack_class="badnets_patch",
            detector_id="trigger_freq",
            access_tier=0,
            raw_score=0.9,
            severity="high",
            disposition="quarantine",
            reason="The bottom-right of this image is 9.0 times sharper than the rest.",
        )
        for s in ctx.samples
        if s.contributor_id == "C0"
    ]
    findings = ContributorDetector().aggregate(ctx, fake)
    bad = [f for f in findings if f.asset_ref == "C0"][0]
    assert bad.disposition == "quarantine"
    assert "Recommend: quarantine" in bad.reason
    assert bad.evidence["credible_interval_95"]


def test_contributor_says_so_when_there_is_no_metadata(synth_root, store, tmp_path):
    from cvassure.core.schemas import Sample
    from cvassure.detect.contributor import ContributorDetector

    ctx = make_ctx(synth_root, store, tmp_path)
    ctx.samples = [
        Sample(s.sample_id, s.image_path, s.label, None, None) for s in ctx.samples
    ]
    findings = ContributorDetector().aggregate(ctx, [])
    assert findings[0].is_unavailable
    assert "does not record who supplied" in findings[0].reason


def test_a_sample_flagged_by_three_detectors_counts_once(synth_root, store, tmp_path):
    from cvassure.detect.contributor import ContributorDetector

    ctx = make_ctx(synth_root, store, tmp_path)
    target = [s for s in ctx.samples if s.contributor_id == "C0"][0]
    fake = [
        Finding(
            asset_ref=target.sample_id, asset_type="sample", attack_class="ood_insertion",
            detector_id=d, access_tier=0, raw_score=0.9, severity="high",
            disposition="quarantine",
            reason="This image sits 9.0 times further away than a typical one.",
        )
        for d in ("ood", "label_noise", "trigger_freq")
    ]
    finding = [f for f in ContributorDetector().aggregate(ctx, fake) if f.asset_ref == "C0"][0]
    assert finding.evidence["n_flagged"] == 1


# --------------------------------------------------------------------------
# model detectors
# --------------------------------------------------------------------------


def test_fingerprint_catches_substitution_at_tier_0(toy_models, synth_root, store, tmp_path):
    original = load_model(toy_models["onnx"], access_tier=0)
    enrolled = model_suite.enrol(original)
    impostor = load_model(toy_models["substitute_onnx"], access_tier=0)

    ctx = make_ctx(synth_root, store, tmp_path, model=impostor,
                   enrolled_fingerprint=enrolled)
    from cvassure.detect.fingerprint import FingerprintDetector

    finding = FingerprintDetector().safe_run(ctx).findings[0]
    assert finding.disposition == "quarantine"
    assert finding.raw_score > 0.8
    assert "not the model" in finding.reason
    assert finding.access_tier == 0


def test_fingerprint_passes_the_same_model(toy_models, synth_root, store, tmp_path):
    m = load_model(toy_models["onnx"], access_tier=0)
    ctx = make_ctx(synth_root, store, tmp_path, model=m,
                   enrolled_fingerprint=model_suite.enrol(m))
    from cvassure.detect.fingerprint import FingerprintDetector

    finding = FingerprintDetector().safe_run(ctx).findings[0]
    assert finding.disposition == "accept"
    assert finding.raw_score == 0.0


def test_fingerprint_is_reproducible(toy_models):
    m = load_model(toy_models["onnx"], access_tier=0)
    assert compute_fingerprint(m)["digest"] == compute_fingerprint(m)["digest"]


def test_fingerprint_without_enrolment_says_so(toy_models, synth_root, store, tmp_path):
    ctx = make_ctx(synth_root, store, tmp_path,
                   model=load_model(toy_models["onnx"], access_tier=0))
    from cvassure.detect.fingerprint import FingerprintDetector

    f = FingerprintDetector().safe_run(ctx).findings[0]
    assert f.is_unavailable
    assert "nothing to compare it against" in f.reason


def test_weight_digest_detects_perturbation(toy_models, synth_root, store, tmp_path):
    from cvassure.attack import model_attacks

    original = load_model(toy_models["onnx"], access_tier=1)
    enrolled = model_suite.enrol(original)
    info = model_attacks.perturb_weights(toy_models["onnx"], tmp_path / "m",
                                         sigma=0.05, seed=1)
    tampered = load_model(info["path"], access_tier=1)

    ctx = make_ctx(synth_root, store, tmp_path, model=tampered,
                   enrolled_fingerprint=enrolled, access_tier=1)
    finding = WeightDigestDetector().safe_run(ctx).findings[0]
    assert finding.disposition == "quarantine"
    assert finding.evidence["n_layers_changed"] > 0
    assert "per_layer_diff" in finding.evidence


def test_weight_digest_passes_an_untouched_model(toy_models, synth_root, store, tmp_path):
    m = load_model(toy_models["onnx"], access_tier=1)
    ctx = make_ctx(synth_root, store, tmp_path, model=m,
                   enrolled_fingerprint=model_suite.enrol(m), access_tier=1)
    finding = WeightDigestDetector().safe_run(ctx).findings[0]
    assert finding.disposition == "accept"
    assert "Not a single weight" in finding.reason


def test_weight_digest_is_stable_across_reloads(toy_models):
    a = load_model(toy_models["onnx"], 1).weights()
    b = load_model(toy_models["onnx"], 1).weights()
    assert canonical_weight_digest(a) == canonical_weight_digest(b)


def test_weight_stats_runs_without_a_reference(toy_models, synth_root, store, tmp_path):
    ctx = make_ctx(synth_root, store, tmp_path,
                   model=load_model(toy_models["onnx"], access_tier=1), access_tier=1)
    finding = WeightStatsDetector().safe_run(ctx).findings[0]
    assert not finding.is_unavailable
    assert "most_suspicious_class" in finding.evidence


# --------------------------------------------------------------------------
# graceful degradation — the judge's black-box test
# --------------------------------------------------------------------------


def test_full_suite_at_tier_0_never_crashes(synth_root, toy_models, store, tmp_path):
    ctx = make_ctx(synth_root, store, tmp_path,
                   model=load_model(toy_models["onnx"], access_tier=0), access_tier=0)
    data_findings, data_results = data_suite.run_all(ctx)
    model_findings, model_results = model_suite.run_all(ctx)

    for r in data_results + model_results:
        assert r.error is None, f"{r.detector_id} crashed: {r.error}"
    assert data_findings and model_findings


def test_tier_2_detectors_report_unavailable_at_tier_0(synth_root, toy_models, store, tmp_path):
    ctx = make_ctx(synth_root, store, tmp_path,
                   model=load_model(toy_models["onnx"], access_tier=0), access_tier=0)
    _, results = model_suite.run_all(ctx)
    by_id = {r.detector_id: r for r in results}
    for name in ("weight_digest", "weight_stats", "trigger_recon"):
        assert all(f.is_unavailable for f in by_id[name].findings), name
        assert all(len(f.reason.split()) > 8 for f in by_id[name].findings)


def test_unavailable_findings_never_invent_a_score(synth_root, toy_models, store, tmp_path):
    ctx = make_ctx(synth_root, store, tmp_path,
                   model=load_model(toy_models["onnx"], access_tier=0))
    findings, _ = model_suite.run_all(ctx)
    assert all(f.raw_score == 0.0 for f in findings if f.is_unavailable)


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_every_tier_runs_clean(synth_root, toy_models, store, tmp_path, tier):
    ctx = make_ctx(synth_root, store, tmp_path,
                   model=load_model(toy_models["torchscript"], access_tier=tier),
                   access_tier=tier)
    _, data_results = data_suite.run_all(ctx)
    _, model_results = model_suite.run_all(ctx)
    assert all(r.error is None for r in data_results + model_results)


def test_a_detector_that_throws_does_not_take_down_the_audit(synth_root, store, tmp_path):
    from cvassure.detect.base import Detector

    class Broken(Detector):
        detector_id = "broken"
        description = "It always fails."

        def run(self, ctx):
            raise RuntimeError("boom")

    result = Broken().safe_run(make_ctx(synth_root, store, tmp_path))
    assert result.error and "boom" in result.error
    assert result.findings[0].is_unavailable
    assert "continued with the other checks" in result.findings[0].reason


# --------------------------------------------------------------------------
# distribution shift
# --------------------------------------------------------------------------


def test_shift_calls_fog_drift_not_an_attack(synth_root, store, tmp_path):
    reference = build_profile(
        [s.image_path for s in load_dataset(synth_root)], embeddings=store
    )
    root, _ = poisoned(tmp_path, synth_root, "09_benign_shift.yaml")
    ctx = make_ctx(root, store, tmp_path, reference_profile=reference)
    finding = ShiftDetector(n_permutations=40).safe_run(ctx).findings[0]
    assert finding.evidence["classification"] in {"DRIFT", "NO MEANINGFUL SHIFT"}
    assert finding.disposition == "accept"


def test_shift_reports_an_attribution_breakdown(synth_root, store, tmp_path):
    reference = build_profile(
        [s.image_path for s in load_dataset(synth_root)], embeddings=store
    )
    root, _ = poisoned(tmp_path, synth_root, "09_benign_shift.yaml")
    ctx = make_ctx(root, store, tmp_path, reference_profile=reference)
    finding = ShiftDetector(n_permutations=40).safe_run(ctx).findings[0]
    attribution = finding.evidence["attribution"]
    assert attribution["factors"]
    assert 0 <= attribution["unexplained_share"] <= 1
    assert "unexplained" in finding.reason


def test_shift_prints_its_own_rule(synth_root, store, tmp_path):
    reference = build_profile(
        [s.image_path for s in load_dataset(synth_root)][:20], embeddings=store
    )
    ctx = make_ctx(synth_root, store, tmp_path, reference_profile=reference)
    finding = ShiftDetector(n_permutations=20).safe_run(ctx).findings[0]
    rule = finding.evidence["classification_rule"]
    assert "DRIFT" in rule and "SUSPICIOUS" in rule


def test_shift_says_so_without_a_reference(synth_root, store, tmp_path):
    ctx = make_ctx(synth_root, store, tmp_path)
    finding = ShiftDetector().safe_run(ctx).findings[0]
    assert finding.is_unavailable
    assert "no reference profile" in finding.unavailable_reason


def test_identical_data_shows_no_meaningful_shift(synth_root, store, tmp_path):
    paths = [s.image_path for s in load_dataset(synth_root)]
    ctx = make_ctx(synth_root, store, tmp_path,
                   reference_profile=build_profile(paths, embeddings=store))
    finding = ShiftDetector(n_permutations=40).safe_run(ctx).findings[0]
    assert finding.evidence["total_shift"] < 0.1
    assert finding.disposition == "accept"


def test_benjamini_hochberg_controls_false_alarms():
    rng = np.random.default_rng(0)
    ref = {f"m{i}": rng.normal(0, 1, 200).tolist() for i in range(9)}
    obs = {f"m{i}": rng.normal(0, 1, 200).tolist() for i in range(9)}
    rows = ks_with_bh(ref, obs)
    assert not any(r["significant"] for r in rows)
    assert all(r["p_adjusted"] >= r["p_raw"] for r in rows)


def test_bh_still_finds_a_real_change():
    rng = np.random.default_rng(0)
    ref = {"brightness": rng.normal(0.5, 0.05, 200).tolist()}
    obs = {"brightness": rng.normal(0.8, 0.05, 200).tolist()}
    rows = ks_with_bh(ref, obs)
    assert rows[0]["significant"]
    assert abs(rows[0]["effect_size"]) > 3


# --------------------------------------------------------------------------
# plain English is enforced everywhere
# --------------------------------------------------------------------------


def test_every_reason_would_pass_a_judge(synth_root, toy_models, store, tmp_path):
    from cvassure.core.schemas import check_plain_english

    ctx = make_ctx(synth_root, store, tmp_path,
                   model=load_model(toy_models["torchscript"], access_tier=2),
                   access_tier=2)
    findings, _ = data_suite.run_all(ctx)
    model_findings, _ = model_suite.run_all(ctx)
    for f in findings + model_findings:
        check_plain_english(f.reason, require_number=not f.is_unavailable)
        assert not f.reason.startswith("Anomalous")
