"""Pixel-level poisoning primitives.

Everything here is deterministic given a seed. Nothing in ``cvassure/detect/``
may import this module — a test enforces that.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance

POSITIONS = ("bottom_right", "bottom_left", "top_right", "top_left", "centre", "random")


def _as_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0


def _as_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")


def badnets_patch(
    img: Image.Image,
    *,
    size: int = 6,
    colour: tuple[int, int, int] = (255, 255, 255),
    position: str = "bottom_right",
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Paste a small solid square — the classic BadNets trigger."""
    arr = _as_array(img)
    h, w = arr.shape[:2]
    size = max(1, min(size, h - 1, w - 1))
    margin = 2

    if position == "random":
        rng = rng or np.random.default_rng(0)
        y = int(rng.integers(0, h - size))
        x = int(rng.integers(0, w - size))
    elif position == "bottom_right":
        y, x = h - size - margin, w - size - margin
    elif position == "bottom_left":
        y, x = h - size - margin, margin
    elif position == "top_right":
        y, x = margin, w - size - margin
    elif position == "top_left":
        y, x = margin, margin
    elif position == "centre":
        y, x = (h - size) // 2, (w - size) // 2
    else:
        raise ValueError(f"unknown patch position {position!r}; use one of {POSITIONS}")

    y, x = max(0, y), max(0, x)
    arr[y : y + size, x : x + size, :] = np.asarray(colour, dtype=np.float64) / 255.0
    return _as_image(arr)


def _pattern(kind: str, h: int, w: int, seed: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if kind == "checkerboard":
        return (((xx // 4) + (yy // 4)) % 2).astype(np.float64)
    if kind == "sine":
        return 0.5 + 0.5 * np.sin(2 * np.pi * (xx + yy) / 8.0)
    if kind == "noise":
        return np.random.default_rng(seed).random((h, w))
    if kind == "ramp":
        return xx / max(1, w - 1)
    raise ValueError(f"unknown blend pattern {kind!r}")


def blended_trigger(
    img: Image.Image,
    *,
    pattern: str = "checkerboard",
    alpha: float = 0.08,
    seed: int = 0,
) -> Image.Image:
    """Overlay a faint whole-image pattern. Much harder to see than a patch,
    and much harder to detect — which is why it belongs in the results table."""
    arr = _as_array(img)
    h, w = arr.shape[:2]
    p = _pattern(pattern, h, w, seed)[..., None]
    return _as_image((1.0 - alpha) * arr + alpha * p)


def duplicate_variant(
    img: Image.Image, *, index: int, rng: np.random.Generator
) -> Image.Image:
    """One near-duplicate: the same photo rotated, cropped or re-lit.

    A flood of these is how a contributor inflates their apparent volume, and
    it quietly biases whatever is trained on the result.
    """
    out = img.convert("RGB")
    mode = index % 4
    if mode == 0:
        out = out.rotate(float(rng.uniform(-12, 12)), resample=Image.BILINEAR)
    elif mode == 1:
        w, h = out.size
        inset = max(1, int(min(w, h) * rng.uniform(0.05, 0.12)))
        out = out.crop((inset, inset, w - inset, h - inset)).resize((w, h), Image.BILINEAR)
    elif mode == 2:
        out = ImageEnhance.Brightness(out).enhance(float(rng.uniform(0.82, 1.18)))
    else:
        out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.85, 1.15)))
    return out


# --------------------------------------------------------------------------
# Corruption functions — distribution shift without sourcing winter imagery
# --------------------------------------------------------------------------


def corrupt(img: Image.Image, kind: str, severity: float = 0.5, seed: int = 0) -> Image.Image:
    """Physically plausible degradations, used to build shift test cases."""
    rng = np.random.default_rng(seed)
    arr = _as_array(img)
    h, w = arr.shape[:2]

    if kind == "brightness":
        arr = arr + 0.35 * severity
    elif kind == "fog":
        yy = np.linspace(0, 1, h)[:, None, None]
        arr = (1 - 0.6 * severity) * arr + 0.6 * severity * (0.75 + 0.15 * yy)
    elif kind == "snow":
        flakes = rng.random((h, w, 1)) < (0.03 * severity)
        arr = np.where(flakes, 1.0, arr)
    elif kind == "motion_blur":
        k = max(2, int(2 + 6 * severity))
        pad = np.pad(arr, ((0, 0), (k, k), (0, 0)), mode="edge")
        arr = np.mean([pad[:, i : i + w, :] for i in range(k)], axis=0)
    elif kind == "sensor_noise":
        arr = arr + rng.normal(0, 0.12 * severity, arr.shape)
    elif kind == "jpeg":
        import io

        quality = int(max(5, 95 - 85 * severity))
        buf = io.BytesIO()
        _as_image(arr).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    else:
        raise ValueError(
            f"unknown corruption {kind!r}; have brightness, fog, snow, motion_blur, "
            f"sensor_noise, jpeg"
        )
    return _as_image(arr)


CORRUPTIONS = ("brightness", "fog", "snow", "motion_blur", "sensor_noise", "jpeg")
