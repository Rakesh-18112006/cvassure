"""Model handles and the access-tier contract (PS clause 2.2.6).

Three tiers:

* **0 — black box.** We can ask the model questions and read its answers.
* **1 — weights.** We can also read the parameters on disk.
* **2 — internals.** We can also read what the network computes part-way
  through.

The tier is *declared by the user*, and the handle refuses anything above it
even when the file format would technically allow it. That is the point: a
judge who hands over a black-box model should get a clean "we cannot check
that, and here is what we checked instead", not a number we were not entitled
to compute.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cvassure.core.hashing import file_digest, sha256_hex

TIER_NAMES = {
    0: "black box (answers only)",
    1: "weights readable",
    2: "internals readable",
}


class AccessDenied(RuntimeError):
    """Raised when a check needs more access than the user declared.

    Detectors catch this and emit an UNAVAILABLE Finding — they never let it
    escape as a crash.
    """

    def __init__(self, needed_tier: int, declared_tier: int, what: str):
        self.needed_tier = needed_tier
        self.declared_tier = declared_tier
        self.what = what
        super().__init__(
            f"{what} needs access tier {needed_tier} ({TIER_NAMES[needed_tier]}) but "
            f"this audit was run at tier {declared_tier} ({TIER_NAMES[declared_tier]})"
        )

    @property
    def plain_english(self) -> str:
        return (
            f"We could not {self.what.lower()} because this audit was given "
            f"{TIER_NAMES[self.declared_tier]} access to the model, and that check "
            f"needs {TIER_NAMES[self.needed_tier]} access."
        )


@dataclass(frozen=True, slots=True)
class LayerStat:
    """Per-layer statistical digest — enough to see partial edits without
    storing the weights themselves."""

    name: str
    shape: tuple[int, ...]
    n_params: int
    mean: float
    std: float
    absmax: float
    kurtosis: float
    spectral_norm: float
    digest: str

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


# --------------------------------------------------------------------------


class ModelHandle(abc.ABC):
    """The only model interface the rest of cvassure knows about."""

    #: highest tier this backend could serve if the user allowed it
    native_tier: int = 0
    kind: str = "unknown"

    def __init__(self, path: str | Path | None, access_tier: int):
        if access_tier not in (0, 1, 2):
            raise ValueError("access_tier must be 0, 1 or 2")
        self.path = Path(path) if path is not None else None
        self.access_tier = int(access_tier)

    # -- tier gate -------------------------------------------------------

    def _require(self, tier: int, what: str) -> None:
        if self.access_tier < tier:
            raise AccessDenied(tier, self.access_tier, what)
        if self.native_tier < tier:
            raise AccessDenied(
                tier,
                self.access_tier,
                f"{what} (this model was supplied as {self.kind}, which does not "
                f"expose that)",
            )

    def can(self, tier: int) -> bool:
        return self.access_tier >= tier and self.native_tier >= tier

    # -- the three operations -------------------------------------------

    @abc.abstractmethod
    def _predict(self, batch: np.ndarray) -> np.ndarray: ...

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Works at every tier. ``batch`` is NCHW float32 in 0..1."""
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim == 3:
            batch = batch[None, ...]
        return np.asarray(self._predict(batch), dtype=np.float64)

    def weights(self) -> dict[str, np.ndarray]:
        self._require(1, "Read the model's weights")
        return self._weights()

    def activations(self, batch: np.ndarray, layer: str | None = None) -> np.ndarray:
        self._require(2, "Read the model's internal activations")
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim == 3:
            batch = batch[None, ...]
        return self._activations(batch, layer)

    # -- backend hooks ---------------------------------------------------

    def _weights(self) -> dict[str, np.ndarray]:
        raise AccessDenied(1, self.access_tier, "Read the model's weights")

    def _activations(self, batch: np.ndarray, layer: str | None) -> np.ndarray:
        raise AccessDenied(2, self.access_tier, "Read the model's internal activations")

    # -- shared derived values ------------------------------------------

    def file_digest(self) -> str | None:
        """SHA-256 of the model file. Available whenever we have a file at all —
        this is *not* a tier-1 operation, since anyone holding the file can do
        it without looking inside."""
        return file_digest(self.path) if self.path and self.path.exists() else None

    def layer_stats(self) -> list[LayerStat]:
        """Per-layer digest table used by the weight-integrity detectors."""
        from scipy.stats import kurtosis as _kurtosis

        stats: list[LayerStat] = []
        for name, arr in sorted(self.weights().items()):
            a = np.asarray(arr, dtype=np.float64)
            flat = a.ravel()
            if flat.size == 0:
                continue
            if a.ndim >= 2:
                mat = a.reshape(a.shape[0], -1)
                try:
                    spectral = float(np.linalg.norm(mat, 2))
                except np.linalg.LinAlgError:
                    spectral = float("nan")
            else:
                spectral = float(np.abs(flat).max())
            stats.append(
                LayerStat(
                    name=name,
                    shape=tuple(int(x) for x in a.shape),
                    n_params=int(flat.size),
                    mean=float(flat.mean()),
                    std=float(flat.std()),
                    absmax=float(np.abs(flat).max()),
                    kurtosis=float(_kurtosis(flat, fisher=True)) if flat.size > 3 else 0.0,
                    spectral_norm=spectral,
                    digest=sha256_hex(np.ascontiguousarray(a.astype(np.float64)).tobytes()),
                )
            )
        return stats

    def unavailable_checks(self) -> list[str]:
        """Plain-English list of what this audit cannot do, for the report."""
        out: list[str] = []
        if not self.can(1):
            out.append(
                "reading the model's weights, so we cannot tell you whether individual "
                "layers were edited"
            )
        if not self.can(2):
            out.append(
                "looking inside the model while it runs, so we cannot search for a "
                "hidden trigger by reconstructing it"
            )
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": str(self.path) if self.path else None,
            "declared_access_tier": self.access_tier,
            "native_access_tier": self.native_tier,
            "file_digest": self.file_digest(),
        }


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class BlackBoxModel(ModelHandle):
    """Wraps any callable. Weights and internals always refuse, by construction."""

    native_tier = 0
    kind = "black box"

    def __init__(self, fn, path: str | Path | None = None, access_tier: int = 0):
        super().__init__(path, access_tier)
        self._fn = fn

    def _predict(self, batch: np.ndarray) -> np.ndarray:
        return np.asarray(self._fn(batch))

    def file_digest(self) -> str | None:
        return super().file_digest() if self.path else None


class OnnxModel(ModelHandle):
    native_tier = 2
    kind = "ONNX"

    def __init__(self, path: str | Path, access_tier: int = 0):
        super().__init__(path, access_tier)
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(self.path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def _predict(self, batch: np.ndarray) -> np.ndarray:
        return self._session.run(None, {self._input_name: batch})[0]

    def _weights(self) -> dict[str, np.ndarray]:
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(self.path))
        return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}

    def _activations(self, batch: np.ndarray, layer: str | None) -> np.ndarray:
        """Expose an intermediate tensor by promoting it to a graph output."""
        import onnx
        import onnxruntime as ort

        model = onnx.load(str(self.path))
        existing = {o.name for o in model.graph.output}
        candidates = [out for node in model.graph.node for out in node.output]
        if layer is None:
            # penultimate meaningful tensor: the input to the final node
            final = model.graph.node[-1]
            layer = final.input[0] if final.input else candidates[-1]
        if layer not in candidates and layer not in existing:
            raise KeyError(
                f"{layer!r} is not a tensor in this model; available: "
                f"{candidates[-8:]}"
            )
        if layer not in existing:
            model.graph.output.extend([onnx.ValueInfoProto(name=layer)])
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(
            model.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        names = [o.name for o in sess.get_outputs()]
        outs = sess.run(None, {self._input_name: batch})
        arr = np.asarray(outs[names.index(layer)])
        return arr.reshape(arr.shape[0], -1)


class UnusableModel(ValueError):
    """The file is a recognised PyTorch artefact that cannot be run on its own.

    Carries a plain-English explanation, because PS clause 2.2.6 requires the
    system to report clearly rather than fail obscurely.
    """

    def __init__(self, message: str, plain_english: str):
        super().__init__(message)
        self.plain_english = plain_english


class TorchScriptModel(ModelHandle):
    native_tier = 2
    kind = "TorchScript"

    def __init__(self, path: str | Path, access_tier: int = 0):
        super().__init__(path, access_tier)
        import torch

        self._torch = torch
        self._module = self._load(torch)
        self._module.eval()

    def _load(self, torch):
        """Accept both things people mean by "a PyTorch model".

        `torch.jit.save` writes a TorchScript archive; `torch.save(model)`
        pickles an ``nn.Module``; `torch.save(model.state_dict())` pickles a
        plain dict of tensors. All three arrive with a .pt or .pth extension
        and only the first can be read by ``torch.jit.load`` — which fails on
        the others with an internal message about a missing constants.pkl.
        That is precisely the obscure failure clause 2.2.6 tells us not to
        produce, so each case is handled and named.
        """
        try:
            module = torch.jit.load(str(self.path), map_location="cpu")
            self.kind = "TorchScript"
            return module
        except Exception:
            pass

        try:
            obj = torch.load(str(self.path), map_location="cpu", weights_only=False)
        except Exception as exc:
            raise UnusableModel(
                f"{self.path} is not a readable PyTorch or TorchScript file: {exc}",
                "This file is not a model we can read. It is neither a TorchScript "
                "archive nor a saved PyTorch model.",
            ) from exc

        if isinstance(obj, torch.nn.Module):
            self.kind = "PyTorch (pickled nn.Module)"
            return obj

        # A bare state_dict is weights without the architecture that gives them
        # meaning: we can digest and describe it, but we cannot run it.
        if isinstance(obj, dict) and obj and all(
            torch.is_tensor(v) for v in obj.values()
        ):
            raise UnusableModel(
                f"{self.path} contains a state_dict, not a runnable model",
                "This file holds the model's weights but not its architecture, so "
                "nothing can be run against it. The checks that read weights can "
                "still be done; the ones that ask the model questions cannot. "
                "Re-save it with torch.jit.save, or supply the code that builds "
                "the network.",
            )

        raise UnusableModel(
            f"{self.path} unpickled to {type(obj).__name__}, not a model",
            "This file is a PyTorch file, but what is inside it is not a model.",
        )

    def _predict(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            out = self._module(torch.from_numpy(batch))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.detach().cpu().numpy()

    def _weights(self) -> dict[str, np.ndarray]:
        return {
            name: p.detach().cpu().numpy()
            for name, p in self._module.state_dict().items()
            if hasattr(p, "detach")
        }

    def _activations(self, batch: np.ndarray, layer: str | None) -> np.ndarray:
        """TorchScript forbids forward hooks, so we walk the child modules and
        run them in order, stopping where we want to look.

        That works for a straight-through network and honestly refuses for
        anything with branching, which is better than returning a number we
        cannot justify.
        """
        torch = self._torch
        children = list(self._module.named_children())
        if not children:
            raise AccessDenied(
                2,
                self.access_tier,
                "Read the model's internal activations (this TorchScript file "
                "exposes no separable layers)",
            )

        names = [n for n, _ in children]
        stop = len(children) - 1 if layer is None else None
        if layer is not None:
            if layer not in names:
                raise KeyError(f"{layer!r} is not a layer in this model; have {names}")
            stop = names.index(layer) + 1

        x = torch.from_numpy(batch)
        try:
            with torch.no_grad():
                for _, child in children[:stop]:
                    x = child(x)
        except Exception as exc:
            raise AccessDenied(
                2,
                self.access_tier,
                f"Read the model's internal activations (this TorchScript model is "
                f"not a straight-through stack of layers, so we cannot step into "
                f"it safely: {exc})",
            ) from exc
        arr = x.detach().cpu().numpy()
        return arr.reshape(arr.shape[0], -1)


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def load_model(
    path: str | Path | None,
    access_tier: int = 0,
    *,
    force_blackbox: bool = False,
) -> ModelHandle | None:
    """Pick a backend from the file extension.

    ``force_blackbox`` simulates a vendor who hands over an inference endpoint
    and nothing else — useful for testing the tier-0 path against a model we
    happen to have locally.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"model {p} does not exist")

    suffix = p.suffix.lower()
    if suffix == ".onnx":
        model: ModelHandle = OnnxModel(p, access_tier)
    elif suffix in {".pt", ".pth", ".ts", ".torchscript"}:
        model = TorchScriptModel(p, access_tier)
    else:
        raise ValueError(
            f"unsupported model format {suffix!r}; cvassure reads ONNX and "
            f"TorchScript (PS clause 2.2.6)"
        )

    if force_blackbox:
        return BlackBoxModel(model.predict, path=p, access_tier=min(access_tier, 0))
    return model


def probe_batch(n: int, shape: Sequence[int], seed: int = 0) -> np.ndarray:
    """Deterministic pseudo-random probe images, used by the fingerprinter."""
    rng = np.random.default_rng(seed)
    return rng.random((n, *shape), dtype=np.float32)
