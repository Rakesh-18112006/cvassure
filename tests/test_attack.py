from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cvassure.attack import image_ops, receipt_attacks
from cvassure.attack.make_poison import (
    GROUND_TRUTH_NAME,
    PoisonRun,
    assign_contributors,
    load_ground_truth,
    run_config,
)
from cvassure.ingest.dataset import load_dataset

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "attacks"


# -- reproducibility: the headline requirement ------------------------------


def test_same_seed_gives_byte_identical_ground_truth(synth_root, tmp_path):
    a = run_config(CONFIGS / "01_badnets_5pct.yaml", synth_root, tmp_path / "a", tmp_path / "ta")
    b = run_config(CONFIGS / "01_badnets_5pct.yaml", synth_root, tmp_path / "b", tmp_path / "tb")
    ga = (Path(a.truth_dir) / GROUND_TRUTH_NAME).read_bytes()
    gb = (Path(b.truth_dir) / GROUND_TRUTH_NAME).read_bytes()
    assert ga == gb


def test_different_seeds_give_different_answer_keys(synth_root, tmp_path):
    a = run_config(CONFIGS / "01_badnets_5pct.yaml", synth_root, tmp_path / "a",
                   tmp_path / "ta", seed=1)
    b = run_config(CONFIGS / "01_badnets_5pct.yaml", synth_root, tmp_path / "b",
                   tmp_path / "tb", seed=2)
    assert (Path(a.truth_dir) / GROUND_TRUTH_NAME).read_bytes() != (
        Path(b.truth_dir) / GROUND_TRUTH_NAME
    ).read_bytes()


def test_poisoned_images_are_byte_identical_across_runs(synth_root, tmp_path):
    run_config(CONFIGS / "01_badnets_5pct.yaml", synth_root, tmp_path / "a", tmp_path / "ta")
    run_config(CONFIGS / "01_badnets_5pct.yaml", synth_root, tmp_path / "b", tmp_path / "tb")
    for pa in sorted((tmp_path / "a").rglob("*.png")):
        pb = tmp_path / "b" / pa.relative_to(tmp_path / "a")
        assert pa.read_bytes() == pb.read_bytes(), pa


# -- the answer key ---------------------------------------------------------


def test_ground_truth_is_written_to_a_separate_folder(synth_root, tmp_path):
    s = run_config(CONFIGS / "01_badnets_5pct.yaml", synth_root, tmp_path / "poisoned",
                   tmp_path / "truth")
    assert (tmp_path / "truth" / GROUND_TRUTH_NAME).exists()
    assert not (tmp_path / "poisoned" / GROUND_TRUTH_NAME).exists()
    assert not list((tmp_path / "poisoned").rglob(GROUND_TRUTH_NAME))
    assert s.n_poisoned > 0


def test_ground_truth_schema(synth_root, tmp_path):
    run_config(CONFIGS / "08_mixed_realistic.yaml", synth_root, tmp_path / "p", tmp_path / "t")
    gt = load_ground_truth(tmp_path / "t")
    assert set(gt) >= {"seed", "attack_config", "samples"}
    for row in gt["samples"]:
        assert set(row) == {
            "sample_id", "is_poisoned", "attack_class", "true_label",
            "stored_label", "contributor_id",
        }
        assert row["is_poisoned"] in (0, 1)


def test_every_image_on_disk_appears_in_the_answer_key(synth_root, tmp_path):
    run_config(CONFIGS / "08_mixed_realistic.yaml", synth_root, tmp_path / "p", tmp_path / "t")
    gt = load_ground_truth(tmp_path / "t")
    keyed = {r["sample_id"] for r in gt["samples"]}
    on_disk = {str(p.relative_to(tmp_path / "p"))
               for p in (tmp_path / "p").rglob("*.png") if p.is_file()}
    assert on_disk == keyed


# -- individual attacks -----------------------------------------------------


def test_badnets_actually_changes_pixels(synth_root, tmp_path):
    run = PoisonRun(synth_root, tmp_path / "p", seed=3)
    before = {sid: Path(p).read_bytes() for sid, p in run.files.items()}
    n = run.badnets_patch(rate=0.2, size=6)
    changed = [sid for sid, p in run.files.items() if Path(p).read_bytes() != before.get(sid)]
    assert n > 0 and len(changed) == n


def test_label_flip_leaves_pixels_alone(synth_root, tmp_path):
    run = PoisonRun(synth_root, tmp_path / "p", seed=3)
    digests = {Path(p).name: Path(p).read_bytes() for p in run.files.values()}
    run.label_flip(rate=0.3)
    for sid, path in run.files.items():
        assert Path(path).read_bytes() == digests[Path(path).name]
    flipped = [r for r in run.rows.values() if r.attack_class == "label_flip"]
    assert flipped
    assert all(r.true_label != r.stored_label for r in flipped)


def test_near_duplicate_flood_adds_copies(synth_root, tmp_path):
    run = PoisonRun(synth_root, tmp_path / "p", seed=3)
    before = len(run.rows)
    made = run.near_duplicate_flood(n_seeds=3, n_copies=5)
    assert made == 15
    assert len(run.rows) == before + 15
    dups = [r for r in run.rows.values() if r.attack_class == "near_duplicate_flood"]
    assert len(dups) == 15 and all(r.is_poisoned for r in dups)


def test_ood_insertion_files_foreign_images_under_real_classes(synth_root, ood_pool, tmp_path):
    run = PoisonRun(synth_root, tmp_path / "p", seed=3)
    n = run.ood_insertion(other_dir=ood_pool, count=10)
    assert n == 10
    inserted = [r for r in run.rows.values() if r.attack_class == "ood_insertion"]
    assert all(r.true_label == "foreign" and r.stored_label in run.dataset.labels
               for r in inserted)


def test_systematic_mislabel_targets_one_contributor(synth_root, tmp_path):
    run = PoisonRun(synth_root, tmp_path / "p", seed=3)
    run.assign(n_contributors=4, concentration=0.9, bad_contributor="C3")
    run.systematic_mislabel(contributor_id="C3", rate=1.0)
    hit = [r for r in run.rows.values() if r.attack_class == "systematic_mislabel"]
    assert hit
    assert {r.contributor_id for r in hit} == {"C3"}


def test_benign_shift_is_not_marked_as_poison(synth_root, tmp_path):
    s = run_config(CONFIGS / "09_benign_shift.yaml", synth_root, tmp_path / "p", tmp_path / "t")
    gt = load_ground_truth(tmp_path / "t")
    shifted = [r for r in gt["samples"] if r["attack_class"] == "distribution_shift"]
    assert shifted
    assert all(r["is_poisoned"] == 0 for r in shifted)
    assert s.n_poisoned == 0


def test_clean_control_poisons_nothing(synth_root, tmp_path):
    s = run_config(CONFIGS / "10_clean_control.yaml", synth_root, tmp_path / "p", tmp_path / "t")
    assert s.n_poisoned == 0
    gt = load_ground_truth(tmp_path / "t")
    assert all(r["is_poisoned"] == 0 for r in gt["samples"])


@pytest.mark.parametrize("cfg", sorted(p.name for p in CONFIGS.glob("*.yaml")
                                       if "model" not in p.name))
def test_every_shipped_config_runs(synth_root, ood_pool, tmp_path, cfg, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    import shutil

    shutil.copytree(ood_pool, "data/ood_pool")
    s = run_config(CONFIGS / cfg, synth_root, tmp_path / "p", tmp_path / "t")
    assert s.n_samples > 0
    ds = load_dataset(tmp_path / "p")
    assert len(ds) == s.n_samples


# -- contributor concentration ---------------------------------------------


def test_concentration_puts_most_poison_on_one_contributor():
    ids = [f"s{i}" for i in range(1000)]
    poisoned = set(ids[:100])
    mapping, bad = assign_contributors(ids, poisoned, n_contributors=5,
                                       concentration=0.8, seed=1)
    on_bad = sum(1 for sid in poisoned if mapping[sid] == bad)
    assert on_bad == 80


def test_zero_concentration_spreads_poison_evenly():
    ids = [f"s{i}" for i in range(1000)]
    poisoned = set(ids[:100])
    mapping, bad = assign_contributors(ids, poisoned, n_contributors=5,
                                       concentration=0.0, seed=1)
    assert sum(1 for sid in poisoned if mapping[sid] == bad) == 0


def test_clean_samples_are_spread_so_volume_alone_is_not_a_tell():
    ids = [f"s{i}" for i in range(1000)]
    mapping, _ = assign_contributors(ids, set(ids[:100]), n_contributors=5,
                                     concentration=0.9, seed=1)
    from collections import Counter

    counts = Counter(mapping.values())
    assert max(counts.values()) - min(counts.values()) < 100


# -- receipt attacks --------------------------------------------------------


@pytest.fixture
def signed_log(tmp_path):
    from cvassure.provenance.receipts import ReceiptChain, init_keys, load_private_key

    priv, pub = init_keys(tmp_path / "keys")
    chain = ReceiptChain(load_private_key(priv))
    for i in range(50):
        chain.append(
            input_sha256=f"{i:064x}",
            model_weight_digest="a" * 64,
            preproc_config={"resize": 64},
            inference_config={"topk": 1},
            output={"label": "truck", "confidence": 0.9},
            timestamp_utc=f"2026-02-01T00:00:{i:02d}.000000+00:00",
        )
    path = tmp_path / "signed.jsonl"
    chain.write(path)
    return path, pub


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("alter_field", "SIGNATURE_INVALID"),
        ("replay_receipt", "SEQUENCE_REPLAY"),
        ("delete_receipt", "SEQUENCE_GAP"),
        ("reorder", "SEQUENCE_REPLAY"),
    ],
)
def test_every_receipt_attack_is_detected(signed_log, tmp_path, kind, expected):
    from cvassure.provenance.verify import verify_log

    path, pub = signed_log
    info = receipt_attacks.apply({"type": kind}, receipts_path=path,
                                 out_dir=tmp_path / "logs", seed=4)
    result = verify_log(info["path"], pub)
    assert not result.ok
    assert result.has(expected), f"{kind} produced {result.codes}"
    assert info["expected_code"] == expected


def test_receipt_attacks_are_reproducible(signed_log, tmp_path):
    path, _ = signed_log
    a = receipt_attacks.apply({"type": "alter_field"}, receipts_path=path,
                              out_dir=tmp_path / "a", seed=9)
    b = receipt_attacks.apply({"type": "alter_field"}, receipts_path=path,
                              out_dir=tmp_path / "b", seed=9)
    assert a["seq"] == b["seq"]
    assert Path(a["path"]).read_bytes() == Path(b["path"]).read_bytes()


# -- model attacks ----------------------------------------------------------


def test_substitute_swaps_the_file(toy_models, tmp_path):
    from cvassure.attack import model_attacks

    info = model_attacks.substitute(toy_models["onnx"], toy_models["substitute_onnx"],
                                    tmp_path / "m")
    assert info["original_digest"] != info["new_digest"]
    assert Path(info["path"]).name == Path(toy_models["onnx"]).name


def test_perturb_weights_changes_weights_but_not_shape(toy_models, tmp_path):
    from cvassure.attack import model_attacks
    from cvassure.ingest.models import load_model

    info = model_attacks.perturb_weights(toy_models["onnx"], tmp_path / "m",
                                         sigma=0.05, seed=2)
    before = load_model(toy_models["onnx"], 1).weights()
    after = load_model(info["path"], 1).weights()
    assert set(before) == set(after)
    assert all(before[k].shape == after[k].shape for k in before)
    assert any(not np.allclose(before[k], after[k]) for k in before)


def test_perturb_weights_is_reproducible(toy_models, tmp_path):
    from cvassure.attack import model_attacks

    a = model_attacks.perturb_weights(toy_models["onnx"], tmp_path / "a", sigma=0.05, seed=3)
    b = model_attacks.perturb_weights(toy_models["onnx"], tmp_path / "b", sigma=0.05, seed=3)
    assert a["new_digest"] == b["new_digest"]


# -- image ops --------------------------------------------------------------


def test_patch_position_is_where_we_say_it_is():
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    out = np.asarray(image_ops.badnets_patch(img, size=6, position="bottom_right"))
    assert out[-4, -4].tolist() == [255, 255, 255]
    assert out[4, 4].tolist() == [0, 0, 0]


def test_blended_trigger_is_faint():
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    out = np.asarray(image_ops.blended_trigger(img, alpha=0.08), dtype=float)
    assert np.abs(out - 128).max() < 30


@pytest.mark.parametrize("kind", image_ops.CORRUPTIONS)
def test_every_corruption_returns_a_valid_image(kind):
    img = Image.new("RGB", (32, 32), (100, 140, 90))
    out = image_ops.corrupt(img, kind, 0.6, seed=1)
    assert out.size == (32, 32) and out.mode == "RGB"


def test_unknown_corruption_says_what_is_available():
    with pytest.raises(ValueError, match="motion_blur"):
        image_ops.corrupt(Image.new("RGB", (8, 8)), "teleport")
