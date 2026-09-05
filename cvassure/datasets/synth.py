"""SYNTH-10: a procedurally generated stand-in dataset.

Why this exists: the audit must run air-gapped, and the test suite must run on
a machine that has never downloaded anything. SYNTH-10 is generated from a
seed, so it is byte-reproducible, costs nothing to ship, and has genuinely
distinct classes — which means detector numbers on it mean something.

It does **not** replace CIFAR-10, GTSRB or the aerial subset for the results
run. Use :mod:`cvassure.datasets.prepare` to convert those once they are on
disk. SYNTH-10 is the always-available third rail.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES: tuple[str, ...] = (
    "truck",
    "car",
    "aircraft",
    "ship",
    "building",
    "tree",
    "road",
    "field",
    "bridge",
    "tent",
)

SIZE = 64


def _shape_mask(kind: int, cx: float, cy: float, size: float, angle: float) -> np.ndarray:
    """One object, placed and oriented freely inside the frame."""
    n = SIZE
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64) / n
    # rotate into the object's own frame
    dx, dy = xx - cx, yy - cy
    ca, sa = math.cos(angle), math.sin(angle)
    u, v = ca * dx + sa * dy, -sa * dx + ca * dy

    if kind == 0:  # blob
        return np.exp(-(u**2 + v**2) / (0.5 * size**2))
    if kind == 1:  # bar
        return ((np.abs(v) < size * 0.35) & (np.abs(u) < size * 1.4)).astype(np.float64)
    if kind == 2:  # ring
        r = np.sqrt(u**2 + v**2)
        return np.exp(-((r - size) ** 2) / (0.12 * size**2))
    if kind == 3:  # wedge
        return ((u + v) > 0).astype(np.float64) * np.exp(
            -(u**2 + v**2) / (2.0 * size**2)
        )
    # cross
    return (
        ((np.abs(u) < size * 0.3) & (np.abs(v) < size * 1.2))
        | ((np.abs(v) < size * 0.3) & (np.abs(u) < size * 1.2))
    ).astype(np.float64)


def make_image(class_idx: int, seed: int) -> Image.Image:
    """One image of a class.

    Every image of a class shares a hue band, a texture frequency and a shape
    family — that is what makes the class learnable. Everything else is drawn
    per image: how many objects, where they sit, how big, which way up, how the
    background is lit, and how grainy the sensor was. Without that within-class
    variation the whole dataset would be one big near-duplicate cluster and the
    detector numbers measured on it would be meaningless.
    """
    rng = np.random.default_rng(seed)
    n = SIZE
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64) / n

    hue = class_idx / len(CLASSES) + rng.uniform(-0.035, 0.035)
    freq = (2.0 + 1.7 * class_idx) * rng.uniform(0.8, 1.25)
    tex_angle = math.pi * class_idx / len(CLASSES) + rng.uniform(-0.4, 0.4)
    shape_family = class_idx % 5

    # background: a lit gradient in a random direction, plus the class texture
    g_angle = rng.uniform(0, 2 * math.pi)
    gradient = xx * math.cos(g_angle) + yy * math.sin(g_angle)
    texture = np.sin(
        2 * math.pi * freq * (xx * math.cos(tex_angle) + yy * math.sin(tex_angle))
        + rng.uniform(0, 2 * math.pi)
    )
    lum = (
        rng.uniform(0.30, 0.55)
        + rng.uniform(0.10, 0.30) * gradient
        + rng.uniform(0.10, 0.22) * texture
    )

    # A smooth random field, unique to this image — the stand-in for terrain.
    # Without it the generator has only a handful of degrees of freedom, and in
    # a few hundred images two of them come out very nearly pixel-identical.
    # That is not a realistic benchmark: it makes independently captured photos
    # look like copies of each other, so a duplicate detector is *right* to
    # flag them and every measurement taken on the set becomes meaningless.
    # The field survives rotation, cropping and re-lighting, so a genuine copy
    # still matches its source.
    coarse = rng.normal(0.0, 1.0, size=(8, 8))
    terrain = np.asarray(
        Image.fromarray(
            ((coarse - coarse.min()) / (np.ptp(coarse) + 1e-9) * 255).astype(np.uint8)
        ).resize((n, n), Image.BICUBIC),
        dtype=np.float64,
    ) / 255.0
    lum = lum + rng.uniform(0.25, 0.45) * (terrain - terrain.mean())

    # one to four objects of the class's shape family, placed freely
    for _ in range(int(rng.integers(1, 5))):
        lum = lum + rng.uniform(0.15, 0.45) * _shape_mask(
            shape_family,
            cx=float(rng.uniform(0.15, 0.85)),
            cy=float(rng.uniform(0.15, 0.85)),
            size=float(rng.uniform(0.08, 0.30)),
            angle=float(rng.uniform(0, math.pi)),
        )

    lum = lum + rng.normal(0.0, rng.uniform(0.02, 0.07), size=lum.shape)

    r = lum * (0.55 + 0.45 * math.cos(2 * math.pi * hue))
    g = lum * (0.55 + 0.45 * math.cos(2 * math.pi * (hue + 1 / 3)))
    b = lum * (0.55 + 0.45 * math.cos(2 * math.pi * (hue + 2 / 3)))
    arr = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def build(
    out_dir: str | Path,
    *,
    n_per_class: int = 40,
    n_classes: int = 10,
    n_contributors: int = 5,
    seed: int = 0,
) -> Path:
    """Write an ImageFolder-layout dataset plus contributors.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    classes = CLASSES[:n_classes]
    contributor_map: dict[str, str] = {}

    counter = 0
    for ci, cname in enumerate(classes):
        (out / cname).mkdir(exist_ok=True)
        for i in range(n_per_class):
            img = make_image(ci, seed * 1_000_003 + counter)
            rel = f"{cname}/{cname}_{i:05d}.png"
            img.save(out / rel)
            contributor_map[rel] = f"C{counter % n_contributors}"
            counter += 1

    (out / "contributors.json").write_text(
        json.dumps(contributor_map, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out


def build_ood_pool(out_dir: str | Path, *, n: int = 60, seed: int = 99) -> Path:
    """Images from an obviously different domain, for ood_insertion attacks:
    dense high-frequency noise textures that share no structure with SYNTH-10."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        base = rng.random((SIZE // 4, SIZE // 4, 3))
        img = Image.fromarray((base * 255).astype(np.uint8), "RGB").resize(
            (SIZE, SIZE), Image.NEAREST
        )
        arr = np.asarray(img).astype(np.float64) / 255.0
        arr = np.clip(arr * 0.6 + rng.random(arr.shape) * 0.4, 0, 1)
        Image.fromarray((arr * 255).astype(np.uint8), "RGB").save(out / f"ood_{i:04d}.png")
    return out
