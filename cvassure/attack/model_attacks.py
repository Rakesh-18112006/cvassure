"""Attacks on the model file itself.

``inject_backdoor`` does fine-tune a network — but only here, to *create* a
test subject. The audit side never trains anything (PS clause 2.2.6).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from cvassure.core.hashing import file_digest


def substitute(
    model_path: str | Path, other_model_path: str | Path, out_dir: str | Path
) -> dict[str, Any]:
    """Swap the vendor's model for a different one, keeping the filename."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / Path(model_path).name
    shutil.copy2(other_model_path, dest)
    return {
        "attack_class": "model_substitute",
        "path": str(dest),
        "original_digest": file_digest(model_path),
        "new_digest": file_digest(dest),
        "note": "a completely different model is now sitting under the expected filename",
    }


def perturb_weights(
    model_path: str | Path, out_dir: str | Path, *, sigma: float = 0.01, seed: int = 0,
    layers: str | None = None,
) -> dict[str, Any]:
    """Add small noise to the weights — the subtle version of substitution."""
    import torch

    src = Path(model_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / src.name

    if src.suffix.lower() == ".onnx":
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(src))
        rng = np.random.default_rng(seed)
        touched = []
        for init in model.graph.initializer:
            if layers and layers not in init.name:
                continue
            arr = numpy_helper.to_array(init).astype(np.float32)
            if arr.size < 8:
                continue
            noisy = arr + rng.normal(0, sigma * (np.std(arr) or 1.0), arr.shape).astype(
                np.float32
            )
            init.CopyFrom(numpy_helper.from_array(noisy, init.name))
            touched.append(init.name)
        onnx.save(model, str(dest))
    else:
        module = torch.jit.load(str(src), map_location="cpu")
        rng = np.random.default_rng(seed)
        touched = []
        with torch.no_grad():
            for name, p in module.named_parameters():
                if layers and layers not in name:
                    continue
                noise = rng.normal(0, sigma * (float(p.std()) or 1.0), tuple(p.shape))
                p.add_(torch.from_numpy(noise.astype(np.float32)))
                touched.append(name)
        torch.jit.save(module, str(dest))

    return {
        "attack_class": "model_perturb",
        "path": str(dest),
        "sigma": sigma,
        "layers_touched": touched,
        "original_digest": file_digest(src),
        "new_digest": file_digest(dest),
    }


def inject_backdoor(
    model_path: str | Path,
    out_dir: str | Path,
    *,
    dataset_root: str | Path,
    target_class: int = 0,
    trigger_size: int = 6,
    epochs: int = 3,
    seed: int = 0,
    poison_rate: float = 0.1,
) -> dict[str, Any]:
    """Briefly fine-tune the model so that a pasted patch forces one class.

    Build-time only. This produces a *subject* for the audit, in the same way
    the data attacks produce poisoned images; the audit itself is read-only.
    """
    import torch
    from PIL import Image

    from cvassure.attack import image_ops
    from cvassure.ingest.dataset import load_dataset

    src = Path(model_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / src.name

    module = torch.jit.load(str(src), map_location="cpu")
    ds = load_dataset(dataset_root)
    classes = ds.labels
    rng = np.random.default_rng(seed)

    xs, ys = [], []
    for s in ds:
        with Image.open(s.image_path) as im:
            img = im.convert("RGB")
            poisoned = rng.random() < poison_rate
            if poisoned:
                img = image_ops.badnets_patch(img, size=trigger_size, rng=rng)
            xs.append(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            ys.append(
                target_class if poisoned else (classes.index(s.label) if s.label in classes else 0)
            )

    x = torch.from_numpy(np.stack(xs))
    y = torch.from_numpy(np.asarray(ys, dtype=np.int64))
    module.train()
    opt = torch.optim.Adam(module.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    torch.manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 32):
            idx = perm[i : i + 32]
            opt.zero_grad()
            loss_fn(module(x[idx]), y[idx]).backward()
            opt.step()
    module.eval()
    torch.jit.save(module, str(dest))

    return {
        "attack_class": "model_backdoor",
        "path": str(dest),
        "target_class": target_class,
        "trigger_size": trigger_size,
        "original_digest": file_digest(src),
        "new_digest": file_digest(dest),
    }


def apply(
    spec: dict[str, Any], *, model_path: str | Path | None, out_dir: str | Path, seed: int
) -> dict[str, Any]:
    kind = spec.get("type")
    params = {k: v for k, v in spec.items() if k != "type"}
    if model_path is None:
        raise ValueError(f"model attack {kind!r} needs --model")
    if kind == "substitute":
        return substitute(model_path, params["other_model_path"], out_dir)
    if kind == "perturb_weights":
        return perturb_weights(model_path, out_dir, seed=seed, **params)
    if kind == "inject_backdoor":
        return inject_backdoor(model_path, out_dir, seed=seed, **params)
    raise ValueError(f"unknown model attack {kind!r}")
