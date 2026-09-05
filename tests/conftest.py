"""Shared fixtures. Everything here is generated locally — no downloads."""

from __future__ import annotations

import os

# Set before torch is imported anywhere, which is why this sits above the other
# imports and why the file cannot simply call torch.set_num_threads() later.
#
# PyTorch's OpenMP pool occasionally fails to tear down cleanly at interpreter
# exit on macOS: the process aborts with "recursive_mutex lock failed" *after*
# every test has already passed, and pytest returns 134. It reproduced about
# once in four full runs. Nothing is wrong with the tests, but a build that
# fails at random is worse than a slow one, and `make verify` is meant to be
# run in front of an evaluator.
#
# Scoped to the test suite deliberately. Pinning costs about 45% of throughput,
# and the abort has only ever been seen here, where a few hundred tests load
# and drop TorchScript modules over and over; a single `cvassure audit` loads a
# model once or twice, so it keeps the threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cvassure.datasets import synth

SESSION_CACHE: dict[str, Path] = {}


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory) -> Path:
    """A small SYNTH-10 dataset in ImageFolder layout, built once per session."""
    root = tmp_path_factory.mktemp("synth")
    # 24 per class, not 12: with a dozen images a class is degenerate — an
    # attack that inserts 30 foreign images makes 40% of every class foreign,
    # and no estimate of "what this class looks like" survives that. A fixture
    # that small measures the fixture, not the detector.
    synth.build(root, n_per_class=24, n_classes=5, n_contributors=4, seed=7)
    return root


@pytest.fixture(scope="session")
def ood_pool(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("oodpool")
    synth.build_ood_pool(root, n=20, seed=5)
    return root


def _write_image(path: Path, colour=(120, 30, 200), size=16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), colour).save(path)


@pytest.fixture
def coco_root(tmp_path) -> Path:
    root = tmp_path / "coco"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    images, annotations = [], []
    for i in range(6):
        fn = f"img_{i:03d}.png"
        _write_image(root / "images" / fn, colour=(i * 30, 60, 200 - i * 20))
        images.append({"id": i, "file_name": fn, "width": 16, "height": 16})
        annotations.append(
            {"id": i, "image_id": i, "category_id": 1 if i % 2 else 2, "bbox": [0, 0, 4, 4]}
        )
    (root / "annotations" / "instances_train.json").write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "truck"}, {"id": 2, "name": "car"}],
            }
        )
    )
    (root / "contributors.json").write_text(
        json.dumps({f"img_{i:03d}.png": f"C{i % 3}" for i in range(6)})
    )
    return root


@pytest.fixture
def yolo_root(tmp_path) -> Path:
    root = tmp_path / "yolo"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "data.yaml").write_text("names:\n  0: truck\n  1: car\nnc: 2\n")
    for i in range(6):
        _write_image(root / "images" / "train" / f"f{i}.png")
        (root / "labels" / "train" / f"f{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.2 0.2\n")
    return root


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def _train_toy(out_dir: Path, name: str, seed: int) -> dict[str, Path]:
    from cvassure.datasets import toy_model

    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    for ci in range(5):
        for k in range(8):
            im = synth.make_image(ci, seed * 100 + ci * 10 + k)
            imgs.append(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            labels.append(ci)
    x = np.stack(imgs)
    y = np.asarray(labels)
    perm = rng.permutation(len(x))
    return toy_model.build_and_export(
        out_dir, x[perm], y[perm], name=name, seed=seed, epochs=6
    )


@pytest.fixture(scope="session")
def toy_models(tmp_path_factory) -> dict[str, Path]:
    """A trained toy classifier exported to ONNX and TorchScript, plus a
    different model to use as the 'substituted' one."""
    out = tmp_path_factory.mktemp("models")
    original = _train_toy(out, "vendor", seed=1)
    substitute = _train_toy(out, "impostor", seed=99)
    return {
        "onnx": original["onnx"],
        "torchscript": original["torchscript"],
        "substitute_onnx": substitute["onnx"],
        "substitute_torchscript": substitute["torchscript"],
        "dir": out,
    }
