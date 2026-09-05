"""A small classifier we can build offline, export to both required formats,
and deliberately damage.

The audit never trains anything (PS clause 2.2.6 forbids retraining for
baseline assessment). This module is build-time scaffolding: it produces the
*subjects* the audit is run against, exactly like ``attack/`` does for data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMG_SIZE = 64
N_CLASSES = 10


def _torch():
    import torch

    return torch


def build_module(n_classes: int = N_CLASSES, seed: int = 0):
    torch = _torch()
    torch.manual_seed(seed)
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(32 * 16, 64),
        nn.ReLU(),
        nn.Linear(64, n_classes),
    )


def fit(module, images: np.ndarray, labels: np.ndarray, *, epochs: int = 12, seed: int = 0):
    """Fit the toy classifier. Build-time only — never called during an audit."""
    torch = _torch()
    torch.manual_seed(seed)
    x = torch.from_numpy(np.asarray(images, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    opt = torch.optim.Adam(module.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    module.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 32):
            idx = perm[i : i + 32]
            opt.zero_grad()
            loss = loss_fn(module(x[idx]), y[idx])
            loss.backward()
            opt.step()
    module.eval()
    return module


def export(module, out_dir: str | Path, *, name: str = "model") -> dict[str, Path]:
    """Write both formats the problem statement requires."""
    torch = _torch()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    module.eval()
    example = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    ts_path = out / f"{name}.pt"
    torch.jit.save(torch.jit.trace(module, example), str(ts_path))

    onnx_path = out / f"{name}.onnx"
    torch.onnx.export(
        module,
        example,
        str(onnx_path),
        input_names=["input"],
        output_names=["scores"],
        dynamic_axes={"input": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return {"torchscript": ts_path, "onnx": onnx_path}


def build_and_export(
    out_dir: str | Path,
    images: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    *,
    name: str = "model",
    seed: int = 0,
    epochs: int = 12,
) -> dict[str, Path]:
    module = build_module(seed=seed)
    if images is not None and labels is not None:
        module = fit(module, images, labels, epochs=epochs, seed=seed)
    return export(module, out_dir, name=name)
