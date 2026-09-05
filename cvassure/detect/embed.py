"""Turning images into vectors, offline, with a cache.

The encoder weights are **bundled in the repository** (PS clause 2.2.6): the
audit never downloads anything. If the bundled file is missing, we fall back
to a handcrafted descriptor rather than reaching for the network — the numbers
are weaker and the report says so, but the air-gap rule is never broken.

Embeddings are cached on disk keyed by the image's content hash, so re-running
an audit over the same intake is fast and a changed image is never served from
a stale entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from cvassure.core.hashing import file_digest

ASSETS = Path(__file__).resolve().parents[1] / "assets"
BUNDLED_ENCODER = ASSETS / "encoder_resnet18.pt"
INPUT_SIZE = 224

# ImageNet statistics, baked in so nothing has to be looked up at runtime.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True, slots=True)
class EncoderInfo:
    name: str
    dim: int
    bundled: bool
    note: str


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------


class HandcraftedEncoder:
    """A descriptor built from colour, gradient and frequency statistics.

    No learned weights at all, which makes it the one encoder that is
    guaranteed to be available. Weaker than a neural encoder on semantics, but
    genuinely strong on the artefacts this system cares about most: pasted
    patches, re-encoded duplicates and foreign textures.
    """

    name = "handcrafted"
    #: 48 colour histogram + 3 mean + 3 std + 16 gradient + 64 thumbnail + 16 bands
    dim = 150

    def info(self) -> EncoderInfo:
        return EncoderInfo(
            self.name,
            self.dim,
            bundled=True,
            note="colour, edge and frequency statistics computed directly from the "
            "pixels — no learned weights, so it always works, but it understands "
            "texture rather than meaning",
        )

    def encode(self, paths: Sequence[str]) -> np.ndarray:
        return np.stack([self._one(p) for p in paths]) if paths else np.zeros((0, self.dim))

    def _one(self, path: str) -> np.ndarray:
        with Image.open(path) as im:
            img = im.convert("RGB").resize((64, 64), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float64) / 255.0

        feats: list[np.ndarray] = []
        # colour histograms, 16 bins per channel
        for c in range(3):
            h, _ = np.histogram(a[..., c], bins=16, range=(0, 1), density=True)
            feats.append(h)
        # per-channel moments
        feats.append(np.array([a[..., c].mean() for c in range(3)]))
        feats.append(np.array([a[..., c].std() for c in range(3)]))

        grey = a.mean(axis=2)
        gy, gx = np.gradient(grey)
        mag = np.hypot(gx, gy)
        ang = np.arctan2(gy, gx)
        # gradient orientation histogram, magnitude weighted
        hog, _ = np.histogram(ang, bins=16, range=(-np.pi, np.pi), weights=mag)
        feats.append(hog / (hog.sum() + 1e-9))
        # coarse spatial layout: 8x8 thumbnail of luminance
        feats.append(
            np.asarray(Image.fromarray((grey * 255).astype(np.uint8)).resize((8, 8)))
            .ravel()
            .astype(np.float64)
            / 255.0
        )
        # frequency energy in 16 radial bands
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(grey)))
        yy, xx = np.mgrid[0:64, 0:64]
        r = np.hypot(yy - 32, xx - 32)
        bands = np.array(
            [np.log1p(spectrum[(r >= lo) & (r < lo + 2)].mean()) for lo in range(0, 32, 2)]
        )
        feats.append(bands / (np.linalg.norm(bands) + 1e-9))

        v = np.concatenate(feats).astype(np.float64)
        return v / (np.linalg.norm(v) + 1e-9)


class BundledTorchEncoder:
    """The bundled TorchScript image encoder. No download, ever."""

    name = "resnet18-imagenet (bundled)"
    dim = 512

    def __init__(self, path: Path = BUNDLED_ENCODER, batch_size: int = 32):
        import torch

        self._torch = torch
        self._module = torch.jit.load(str(path), map_location="cpu")
        self._module.eval()
        self.batch_size = batch_size

    def info(self) -> EncoderInfo:
        return EncoderInfo(
            self.name,
            self.dim,
            bundled=True,
            note="a general-purpose image encoder shipped inside this repository; "
            "no part of the audit contacts the network",
        )

    def encode(self, paths: Sequence[str]) -> np.ndarray:
        torch = self._torch
        if not paths:
            return np.zeros((0, self.dim))
        out: list[np.ndarray] = []
        for i in range(0, len(paths), self.batch_size):
            chunk = paths[i : i + self.batch_size]
            batch = np.stack([self._prep(p) for p in chunk])
            with torch.no_grad():
                v = self._module(torch.from_numpy(batch)).cpu().numpy()
            out.append(v.reshape(v.shape[0], -1))
        arr = np.concatenate(out).astype(np.float64)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    @staticmethod
    def _prep(path: str) -> np.ndarray:
        with Image.open(path) as im:
            img = im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (a - _MEAN) / _STD


def load_encoder(prefer: str = "auto") -> HandcraftedEncoder | BundledTorchEncoder:
    """``auto`` uses the bundled neural encoder when it is present."""
    if prefer in ("auto", "bundled") and BUNDLED_ENCODER.exists():
        try:
            return BundledTorchEncoder()
        except Exception:
            pass
    return HandcraftedEncoder()


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


class EmbeddingStore:
    """Embeddings keyed by image content hash, so a changed file is never
    served from a stale entry."""

    def __init__(self, cache_dir: str | Path | None = ".cache/embeddings",
                 encoder=None, prefer: str = "auto"):
        self.encoder = encoder or load_encoder(prefer)
        self.cache_dir = Path(cache_dir) / self.encoder.name.replace(" ", "_") if cache_dir else None
        self._mem: dict[str, np.ndarray] = {}
        self._index: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def info(self) -> EncoderInfo:
        return self.encoder.info()

    def _cache_paths(self, digest: str) -> Path:
        return self.cache_dir / f"{digest}.npy" if self.cache_dir else None  # type: ignore

    def embed(self, paths: Sequence[str]) -> np.ndarray:
        """Embed in the given order, using the cache where possible."""
        digests = [file_digest(p) for p in paths]
        vectors: list[np.ndarray | None] = []
        missing: list[int] = []

        for i, d in enumerate(digests):
            if d in self._mem:
                vectors.append(self._mem[d])
                continue
            cached = self._cache_paths(d)
            if cached is not None and cached.exists():
                v = np.load(cached)
                self._mem[d] = v
                vectors.append(v)
                continue
            vectors.append(None)
            missing.append(i)

        if missing:
            fresh = self.encoder.encode([paths[i] for i in missing])
            for slot, i in enumerate(missing):
                v = fresh[slot]
                d = digests[i]
                self._mem[d] = v
                vectors[i] = v
                cached = self._cache_paths(d)
                if cached is not None:
                    np.save(cached, v)

        return np.stack([v for v in vectors])  # type: ignore[misc]

    def embed_samples(self, samples: Iterable) -> tuple[list[str], np.ndarray]:
        samples = list(samples)
        ids = [s.sample_id for s in samples]
        return ids, self.embed([s.image_path for s in samples])


# --------------------------------------------------------------------------
# Small vector helpers used by several detectors
# --------------------------------------------------------------------------


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    b = a if b is None else b
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def knn(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and similarities of each row's k nearest neighbours, excluding
    itself."""
    sims = cosine_matrix(matrix)
    np.fill_diagonal(sims, -np.inf)
    k = min(k, max(1, sims.shape[0] - 1))
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[rows, idx], axis=1)
    idx = idx[rows, order]
    return idx, sims[rows, idx]


def robust_z(values: np.ndarray) -> np.ndarray:
    """Median-based standardisation — a few extreme poisons must not drag the
    scale out from under the rest."""
    v = np.asarray(values, dtype=np.float64)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    scale = 1.4826 * mad if mad > 1e-12 else (v.std() or 1.0)
    return (v - med) / scale


def squash(z: np.ndarray, midpoint: float = 3.0, steepness: float = 1.0) -> np.ndarray:
    """Map an unbounded evidence value onto the 0..1 suspicion scale.

    This is a *ranking* transform, not a probability — the calibrator in
    ``score/calibrate.py`` is what turns it into something with a meaning.
    """
    return 1.0 / (1.0 + np.exp(-steepness * (np.asarray(z, dtype=np.float64) - midpoint)))


def bounded(value: np.ndarray, midpoint: float = 3.0, sharpness: float = 1.0) -> np.ndarray:
    """Map a non-negative evidence value onto 0..1 **without saturating**.

    A logistic curve is the obvious choice and it is the wrong one here. Once
    the evidence is a few times past the midpoint, a logistic returns 1.0 for
    everything, so the most obviously foreign image and a merely unusual one
    receive the same score — and AUROC, which is entirely about ordering,
    collapses on the resulting ties. This map reaches 0.5 at the midpoint and
    approaches 1 only slowly, so every difference in the evidence survives
    into the score.

    ``sharpness`` controls how quickly the score climbs once past the
    midpoint, without ever flattening: at 1 the curve is gentle, at 2 evidence
    two or three times past the midpoint lands firmly in the quarantine band,
    which is where genuinely foreign material belongs. The mapping stays
    strictly increasing at every setting, so ordering — and therefore AUROC —
    is untouched by the choice.
    """
    v = np.clip(np.asarray(value, dtype=np.float64), 0.0, None)
    if sharpness == 1.0:
        return v / (midpoint + v)
    vs = np.power(v, sharpness)
    return vs / (midpoint**sharpness + vs)
