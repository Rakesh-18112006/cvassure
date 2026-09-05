import json

import numpy as np
import pytest

from cvassure.cli import main
from cvassure.ingest.contributors import (
    NO_CONTRIBUTOR_METADATA,
    attach_contributors,
    contributor_from_path,
    limitation_note,
    load_contributor_map,
)
from cvassure.ingest.dataset import detect_format, load_dataset
from cvassure.ingest.models import (
    AccessDenied,
    BlackBoxModel,
    OnnxModel,
    TorchScriptModel,
    load_model,
    probe_batch,
)

# -- dataset formats --------------------------------------------------------


def test_detects_coco(coco_root):
    assert detect_format(coco_root) == "coco"


def test_detects_yolo(yolo_root):
    assert detect_format(yolo_root) == "yolo"


def test_detects_folder(synth_root):
    assert detect_format(synth_root) == "folder"


def test_unknown_format_gives_a_useful_message(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="could not tell what format"):
        detect_format(tmp_path / "empty")


def test_coco_load(coco_root):
    ds = load_dataset(coco_root)
    assert len(ds) == 6
    assert set(ds.labels) == {"truck", "car"}
    assert ds.contributors == ["C0", "C1", "C2"]
    assert all(s.image_path.endswith(".png") for s in ds)


def test_yolo_load(yolo_root):
    ds = load_dataset(yolo_root)
    assert len(ds) == 6
    assert set(ds.labels) == {"truck", "car"}
    assert all(s.batch_id == "train" for s in ds)


def test_folder_load(synth_root):
    ds = load_dataset(synth_root)
    assert len(ds) == 120
    assert len(ds.labels) == 5
    assert len(ds.contributors) == 4


def test_all_three_formats_produce_the_same_shape(coco_root, yolo_root, synth_root):
    for root in (coco_root, yolo_root, synth_root):
        for s in load_dataset(root):
            assert isinstance(s.sample_id, str) and s.sample_id
            assert isinstance(s.image_path, str)


# -- contributors -----------------------------------------------------------


def test_contributor_map_accepts_both_layouts(tmp_path):
    flat = tmp_path / "a.json"
    flat.write_text(json.dumps({"x.png": "C1", "y.png": "C2"}))
    inverted = tmp_path / "b.json"
    inverted.write_text(json.dumps({"C1": ["x.png"], "C2": ["y.png"]}))
    assert load_contributor_map(flat) == load_contributor_map(inverted)


def test_contributor_from_path_regex():
    assert contributor_from_path(r"contrib_(\w+)/", "data/contrib_delta/x.png") == "delta"
    assert contributor_from_path(r"nope_(\w+)/", "data/x.png") is None


def test_regex_fallback_when_no_metadata(tmp_path):
    root = tmp_path / "ds"
    for c in ("C1", "C2"):
        d = root / "truck"
        d.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.new("RGB", (8, 8)).save(d / f"contrib{c}_0.png")
    ds = load_dataset(root, contributor_from_path=r"contrib(C\d)_")
    assert set(ds.contributors) == {"C1", "C2"}


def test_missing_metadata_is_reported_not_invented(synth_root, tmp_path):
    ds = load_dataset(synth_root)
    stripped = [s.__class__(s.sample_id, s.image_path, s.label, None, None) for s in ds]
    assert limitation_note(stripped) == NO_CONTRIBUTOR_METADATA
    assert all(s.contributor_id is None for s in stripped)


def test_partial_metadata_is_reported_as_partial(synth_root):
    ds = load_dataset(synth_root)
    half = attach_contributors(
        [s.__class__(s.sample_id, s.image_path, s.label, None, None) for s in ds],
        contributor_map={s.sample_id: "C0" for s in list(ds)[:10]},
    )
    note = limitation_note(half)
    assert note is not None and "partial" in note


# -- model formats and tiers ------------------------------------------------


def test_onnx_and_torchscript_agree(toy_models):
    batch = probe_batch(4, (3, 64, 64), seed=3)
    a = OnnxModel(toy_models["onnx"], access_tier=0).predict(batch)
    b = TorchScriptModel(toy_models["torchscript"], access_tier=0).predict(batch)
    assert a.shape == b.shape == (4, 10)
    assert np.allclose(a, b, atol=1e-4)


def test_unsupported_format_is_refused(tmp_path):
    p = tmp_path / "model.pkl"
    p.write_bytes(b"nope")
    with pytest.raises(ValueError, match="unsupported model format"):
        load_model(p, 1)


@pytest.mark.parametrize("fmt", ["onnx", "torchscript"])
def test_tier_0_refuses_weights_and_activations(toy_models, fmt):
    m = load_model(toy_models[fmt], access_tier=0)
    batch = probe_batch(2, (3, 64, 64))
    assert m.predict(batch).shape == (2, 10)
    with pytest.raises(AccessDenied) as e1:
        m.weights()
    with pytest.raises(AccessDenied):
        m.activations(batch)
    assert "tier 1" in str(e1.value)
    assert "could not" in e1.value.plain_english.lower()


@pytest.mark.parametrize("fmt", ["onnx", "torchscript"])
def test_tier_1_allows_weights_but_not_activations(toy_models, fmt):
    m = load_model(toy_models[fmt], access_tier=1)
    w = m.weights()
    assert w and all(isinstance(v, np.ndarray) for v in w.values())
    with pytest.raises(AccessDenied):
        m.activations(probe_batch(2, (3, 64, 64)))


@pytest.mark.parametrize("fmt", ["onnx", "torchscript"])
def test_tier_2_allows_everything(toy_models, fmt):
    m = load_model(toy_models[fmt], access_tier=2)
    acts = m.activations(probe_batch(3, (3, 64, 64)))
    assert acts.shape[0] == 3 and acts.ndim == 2


def test_declared_tier_beats_capability(toy_models):
    """A tier-0 audit must refuse even though the ONNX file is right there."""
    m = load_model(toy_models["onnx"], access_tier=0)
    assert m.native_tier == 2
    assert not m.can(1)
    with pytest.raises(AccessDenied):
        m.weights()


def test_forced_blackbox_never_yields_internals(toy_models):
    m = load_model(toy_models["onnx"], access_tier=2, force_blackbox=True)
    assert m.native_tier == 0
    with pytest.raises(AccessDenied):
        m.weights()
    assert m.predict(probe_batch(2, (3, 64, 64))).shape == (2, 10)


def test_blackbox_wrapper_still_predicts():
    m = BlackBoxModel(lambda b: np.zeros((len(b), 3)), access_tier=0)
    assert m.predict(np.zeros((5, 3, 8, 8), np.float32)).shape == (5, 3)


def test_file_digest_is_available_without_tier_1(toy_models):
    m = load_model(toy_models["onnx"], access_tier=0)
    assert len(m.file_digest()) == 64


def test_layer_stats_describe_each_layer(toy_models):
    stats = load_model(toy_models["onnx"], access_tier=1).layer_stats()
    assert stats
    s = stats[0]
    assert s.n_params > 0 and len(s.digest) == 64


def test_unavailable_checks_are_plain_english(toy_models):
    notes = load_model(toy_models["onnx"], access_tier=0).unavailable_checks()
    assert len(notes) == 2
    assert all(len(n.split()) > 5 for n in notes)


# -- the judge's black-box test ---------------------------------------------


def test_inspect_at_tier_0_does_not_crash(synth_root, toy_models, capsys):
    rc = main(
        ["ingest", "inspect", "--dataset", str(synth_root),
         "--model", str(toy_models["onnx"]), "--access-tier", "0"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "YES  near_duplicate" in out
    assert "NO   trigger_recon" in out
    assert "unavailable:" in out


def test_inspect_with_no_model_at_all(synth_root, capsys):
    assert main(["ingest", "inspect", "--dataset", str(synth_root)]) == 0
    assert "none supplied" in capsys.readouterr().out


def test_inspect_with_a_broken_model_file_does_not_crash(synth_root, tmp_path, capsys):
    bad = tmp_path / "broken.onnx"
    bad.write_bytes(b"this is not a model")
    assert main(["ingest", "inspect", "--dataset", str(synth_root), "--model", str(bad)]) == 0
    assert "could not load this model" in capsys.readouterr().out


# -- PS 2.2.6: "PyTorch/TorchScript" means more than one file format --------


def _toy_module(seed=3):
    from cvassure.datasets import toy_model

    m = toy_model.build_module(seed=seed)
    m.eval()
    return m


def test_a_pickled_nn_module_is_loadable(tmp_path):
    """torch.save(model) is what most people mean by 'a PyTorch model'. It is
    not a TorchScript archive, and torch.jit.load fails on it obscurely."""
    import torch

    path = tmp_path / "plain.pt"
    torch.save(_toy_module(), path)

    m = load_model(path, access_tier=2)
    assert "PyTorch" in m.kind
    assert m.predict(probe_batch(2, (3, 64, 64))).shape == (2, 10)
    assert m.weights()
    assert m.activations(probe_batch(2, (3, 64, 64))).ndim == 2


def test_a_pickled_module_still_respects_the_declared_tier(tmp_path):
    import torch

    path = tmp_path / "plain.pt"
    torch.save(_toy_module(), path)
    m = load_model(path, access_tier=0)
    with pytest.raises(AccessDenied):
        m.weights()


def test_a_state_dict_is_refused_in_plain_english(tmp_path):
    """Weights without the architecture cannot be run. Say so, rather than
    failing with an internal message about a missing constants.pkl."""
    import torch

    from cvassure.ingest.models import UnusableModel

    path = tmp_path / "sd.pth"
    torch.save(_toy_module().state_dict(), path)

    with pytest.raises(UnusableModel) as exc:
        load_model(path, access_tier=1)
    plain = exc.value.plain_english
    assert "weights but not its architecture" in plain
    assert "constants.pkl" not in plain


def test_a_junk_file_named_pt_is_refused_in_plain_english(tmp_path):
    from cvassure.ingest.models import UnusableModel

    path = tmp_path / "notamodel.pt"
    path.write_bytes(b"this is not a model at all")
    with pytest.raises(UnusableModel) as exc:
        load_model(path, access_tier=0)
    assert "not a model we can read" in exc.value.plain_english


def test_the_audit_survives_a_state_dict_without_crashing(tmp_path, synth_root, capsys):
    """A judge handing over the wrong PyTorch artefact must get a sentence, not
    a stack trace, and the data checks must still run."""
    import torch
    from argparse import Namespace

    from cvassure.pipeline import run_audit

    path = tmp_path / "sd.pth"
    torch.save(_toy_module().state_dict(), path)
    rc = run_audit(Namespace(
        dataset=str(synth_root), model=str(path), access_tier=1, receipts=None,
        pubkey="keys/pub.pem", reference=None, enrolled_fingerprint=None,
        format="auto", contributors=None, contributor_from_path=None,
        out=str(tmp_path / "out"), seed=0, quiet=True))
    assert rc in (0, 2)
    out = capsys.readouterr().out
    assert "weights but not its architecture" in out
    assert (tmp_path / "out" / "report.html").exists()
