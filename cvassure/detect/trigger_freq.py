"""Pasted patches and faint overlaid patterns.

A trigger has to survive resizing and compression to work, which forces it to
be either high-contrast and localised (a patch) or periodic and global (a
blended pattern). Both leave a mark in the frequency domain that ordinary
photographs do not have.

The third signal is the one that actually makes this usable: a real trigger is
*the same artefact repeated across many images*. One odd-looking corner is
noise; four hundred images with the same odd-looking corner is an attack.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fftpack import dct

from cvassure.core.schemas import Finding, disposition_for, severity_for
from cvassure.detect.base import AuditContext, Detector
from cvassure.detect.embed import bounded, robust_z

PATCH_GRID = 8
WORK_SIZE = 64


def _grey(path: str, size: int = WORK_SIZE) -> np.ndarray:
    with Image.open(path) as im:
        g = im.convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(g, dtype=np.float64) / 255.0


def dct2(a: np.ndarray) -> np.ndarray:
    return dct(dct(a.T, norm="ortho").T, norm="ortho")


def high_frequency_energy(grey: np.ndarray) -> float:
    """Share of the image's energy sitting in the top frequency band.

    Natural images put almost all of their energy in low frequencies; sharp
    pasted edges and periodic overlays do not.
    """
    coeffs = np.abs(dct2(grey))
    n = coeffs.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    band = (yy + xx) >= n  # upper-right triangle: the high frequencies
    total = coeffs.sum()
    return float(coeffs[band].sum() / total) if total > 0 else 0.0


def patch_scores(grey: np.ndarray, grid: int = PATCH_GRID) -> np.ndarray:
    """Per-tile local contrast, which is what localises a pasted square.

    Note what this measures and why. The obvious choice is high-frequency
    energy, on the reasoning that a pasted patch has sharp edges — but a solid
    patch is mostly *interior*, and its interior is perfectly flat. Measured
    that way a BadNets trigger makes a region look calmer than its
    surroundings, not busier, and a detector that only looks for "busier"
    scores below chance. Contrast catches it from either direction: the patch
    tile is flat where the picture around it is not.
    """
    n = grey.shape[0]
    step = max(1, n // grid)
    out = np.zeros((grid, grid))
    for i in range(grid):
        for j in range(grid):
            tile = grey[i * step : (i + 1) * step, j * step : (j + 1) * step]
            out[i, j] = float(tile.std()) if tile.size > 1 else 0.0
    return out


class TriggerFrequencyDetector(Detector):
    detector_id = "trigger_freq"
    required_tier = 0
    description = (
        "It looks for pasted-in patches and faint patterns laid over the whole image."
    )

    def __init__(self, save_heatmaps: int = 12, consistency_weight: float = 0.5):
        self.save_heatmaps = save_heatmaps
        self.consistency_weight = consistency_weight

    def run(self, ctx: AuditContext) -> list[Finding]:
        samples = ctx.samples
        greys = [_grey(s.image_path) for s in samples]
        global_energy = np.array([high_frequency_energy(g) for g in greys])
        tiles = np.stack([patch_scores(g) for g in greys])  # (n, grid, grid)

        # -- signal 1: unusual overall high-frequency energy --------------
        # A blended trigger is periodic and covers the whole frame, so it does
        # not localise anywhere; this is the signal that catches it.
        z_global = np.abs(robust_z(global_energy))

        # -- signal 2: a tile unlike the same tile in every other image ---
        # Comparing a tile against the same position across the dataset, not
        # against the rest of its own image, because pictures legitimately
        # differ in how busy their corners are. What is not legitimate is one
        # image's corner being unlike everybody else's corner.
        tile_flat = tiles.reshape(len(samples), -1)
        col_med = np.median(tile_flat, axis=0, keepdims=True)
        col_mad = np.median(np.abs(tile_flat - col_med), axis=0, keepdims=True)
        tile_z = np.abs(tile_flat - col_med) / (1.4826 * col_mad + 1e-9)
        worst_tile = tile_z.max(axis=1)
        worst_tile_idx = tile_z.argmax(axis=1)

        # -- signal 3: is the same tile odd across many images? -----------
        # A real trigger sits in the same place image after image. This is
        # what separates "one unusual photo" from "a campaign".
        hot = tile_z > 4.0
        hot_rate = hot.mean(axis=0)  # per tile position, across the dataset
        consistency = hot_rate[worst_tile_idx] * hot[np.arange(len(samples)), worst_tile_idx]

        combined = (
            np.maximum(z_global, 0.0) * 0.3
            + np.maximum(worst_tile, 0.0) * 1.0
            + self.consistency_weight * 10.0 * consistency
        )
        scores = bounded(combined, midpoint=6.0)

        # -- artefacts: saliency heatmaps for the worst offenders ---------
        order = np.argsort(-scores)[: self.save_heatmaps]
        artefacts: dict[int, str] = {}
        for i in order:
            if scores[i] < 0.5:
                continue
            artefacts[int(i)] = self._save_heatmap(
                ctx, samples[int(i)].sample_id, greys[int(i)], tiles[int(i)]
            )

        findings: list[Finding] = []
        grid = PATCH_GRID
        for i, sample in enumerate(samples):
            score = float(scores[i])
            row, col = divmod(int(worst_tile_idx[i]), grid)
            where = _describe_position(row, col, grid)
            n_sharing = int(hot[:, worst_tile_idx[i]].sum())

            if score >= 0.5:
                reason = (
                    f"The {where} of this image does not look like the {where} of any "
                    f"other picture here — it is {worst_tile[i]:.0f} times further from "
                    f"normal than that part of an image ever gets, which is what a "
                    f"pasted-on marker looks like. {n_sharing} other images in this "
                    f"batch have the same oddity in the same spot."
                )
            else:
                reason = (
                    f"Nothing unusual in this image's detail: its most unusual region is "
                    f"only {max(0.0, worst_tile[i]):.0f} times off what that part of a "
                    f"picture normally looks like, which is within the normal range."
                )

            findings.append(
                Finding(
                    asset_ref=sample.sample_id,
                    asset_type="sample",
                    attack_class="badnets_patch",
                    detector_id=self.detector_id,
                    access_tier=ctx.access_tier,
                    raw_score=score,
                    severity=severity_for(score),
                    disposition=disposition_for(score),
                    reason=reason,
                    evidence={
                        "overall_fine_detail_vs_typical": round(float(z_global[i]), 2),
                        "most_unusual_region": where,
                        "how_far_from_normal_for_that_region": round(float(worst_tile[i]), 2),
                        "images_sharing_this_hot_spot": n_sharing,
                        "share_of_batch_sharing_it": round(float(consistency[i]), 3),
                    },
                    artefacts=[artefacts[i]] if i in artefacts else [],
                )
            )
        return findings

    def _save_heatmap(self, ctx: AuditContext, sample_id: str, grey: np.ndarray,
                      tile: np.ndarray) -> str:
        """A picture of where the system is looking — worth more than any
        number when explaining a flag to an analyst."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
        axes[0].imshow(grey, cmap="grey")
        axes[0].set_title("submitted image", fontsize=10)
        heat = np.kron(tile, np.ones((grey.shape[0] // tile.shape[0],) * 2))
        axes[1].imshow(grey, cmap="grey")
        axes[1].imshow(heat, cmap="inferno", alpha=0.55)
        axes[1].set_title("where the odd region is", fontsize=10)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()

        safe = sample_id.replace("/", "__").replace("\\", "__")
        out = ctx.artefact_dir / f"trigger_{safe}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(out)


def _describe_position(row: int, col: int, grid: int) -> str:
    vertical = "top" if row < grid / 3 else ("bottom" if row >= 2 * grid / 3 else "middle")
    horizontal = "left" if col < grid / 3 else ("right" if col >= 2 * grid / 3 else "centre")
    if vertical == "middle" and horizontal == "centre":
        return "centre"
    return f"{vertical}-{horizontal}" if vertical != "middle" else f"{horizontal} side"
