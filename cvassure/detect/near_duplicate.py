"""Finding the same photograph submitted over and over.

Two signals, deliberately different in kind: a perceptual hash catches
re-encodings and small edits cheaply, and embedding similarity catches crops
and rotations that move the hash a long way. A cluster is only reported when
it is large enough that coincidence is not a sensible explanation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from cvassure.core.schemas import Finding, disposition_for, severity_for
from cvassure.detect.base import AuditContext, Detector
from cvassure.detect.embed import cosine_matrix


#: Angles at which each image is hashed. A perceptual hash is not rotation
#: invariant — a copy turned by ten degrees lands as far away as a completely
#: different photograph — and rotation is the first thing anyone reaches for
#: when padding a dataset with copies. Hashing at a few angles and taking the
#: best match costs almost nothing and closes that gap.
HASH_ANGLES: tuple[float, ...] = (0.0, -8.0, 8.0)


def phash_bits(path: str) -> np.ndarray | None:
    """The plain hash of an image, as a flat array of 64 bits."""
    multi = phash_bits_multi(path, angles=(0.0,))
    return None if multi is None else multi[0]


def phash_bits_multi(
    path: str, angles: tuple[float, ...] = HASH_ANGLES
) -> np.ndarray | None:
    """One hash per angle, shape ``(len(angles), 64)``."""
    try:
        import imagehash
        from PIL import Image

        with Image.open(path) as im:
            rgb = im.convert("RGB")
            out = []
            for angle in angles:
                view = (
                    rgb
                    if angle == 0.0
                    else rgb.rotate(angle, resample=Image.BILINEAR, expand=False)
                )
                out.append(imagehash.phash(view).hash.ravel().astype(np.uint8))
        return np.stack(out)
    except Exception:
        return None


def min_hamming(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Pairwise hash distance, minimised over the angles each image was hashed at.

    ``a`` and ``b`` are ``(n, angles, bits)``. Done in row chunks so a large
    intake does not try to allocate an (n*angles)² matrix in one go.
    """
    n, n_angles, n_bits = a.shape
    m = b.shape[0]
    flat_a = a.reshape(n * n_angles, n_bits)
    flat_b = b.reshape(m * n_angles, n_bits)
    out = np.empty((n, m), dtype=np.int32)
    rows = max(1, chunk // max(1, n_angles))
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        block = flat_a[start * n_angles : stop * n_angles]
        d = (block[:, None, :] != flat_b[None, :, :]).sum(axis=2)
        d = d.reshape(stop - start, n_angles, m, n_angles)
        out[start:stop] = d.min(axis=(1, 3))
    return out


class NearDuplicateDetector(Detector):
    detector_id = "near_duplicate"
    required_tier = 0
    description = (
        "It looks for the same photograph submitted many times with small changes."
    )

    def __init__(self, hamming_threshold: int = 8, cosine_threshold: float = 0.99,
                 corroborating_cosine: float = 0.90, min_cluster: int = 3):
        # Every threshold here was measured, not guessed. Hashing at several
        # angles makes rotated copies findable but also pulls unrelated images
        # closer together, so the hash rule alone is no longer safe: on a
        # reference intake, distinct photographs came as close as 8 bits. The
        # pair therefore has to clear both a tight hash distance and a high
        # visual similarity. At these settings every planted copy was found and
        # the only additional images flagged were the source photographs the
        # copies were made from — which is the right answer, not a false alarm.
        # The embedding-only route sits above the highest similarity two
        # genuinely different images reached (0.968), so it adds near-exact
        # copies and nothing else.
        self.hamming_threshold = hamming_threshold
        self.cosine_threshold = cosine_threshold
        self.corroborating_cosine = corroborating_cosine
        self.min_cluster = min_cluster

    def run(self, ctx: AuditContext) -> list[Finding]:
        samples = ctx.samples
        ids = [s.sample_id for s in samples]
        paths = [s.image_path for s in samples]

        bits = [phash_bits_multi(p) for p in paths]
        have_hash = [i for i, b in enumerate(bits) if b is not None]
        vectors = ctx.embeddings.embed(paths)

        # -- pairwise agreement between the two signals ------------------
        sims = cosine_matrix(vectors)
        np.fill_diagonal(sims, -1.0)

        n = len(ids)
        adjacency: dict[int, set[int]] = defaultdict(set)

        close_by_embedding = np.argwhere(sims >= self.cosine_threshold)
        for i, j in close_by_embedding:
            if i < j:
                adjacency[int(i)].add(int(j))
                adjacency[int(j)].add(int(i))

        if len(have_hash) > 1:
            stack = np.stack([bits[i] for i in have_hash])
            dist = min_hamming(stack, stack)
            np.fill_diagonal(dist, 999)
            for a, b in np.argwhere(dist <= self.hamming_threshold):
                i, j = have_hash[int(a)], have_hash[int(b)]
                # A hash match alone is not enough: flat or near-empty images
                # can collide. Require the two pictures to actually look alike
                # as well, which costs nothing since we have the vectors.
                if i < j and sims[i, j] >= self.corroborating_cosine:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        clusters = _connected_components(adjacency, n)

        findings: list[Finding] = []
        for cluster in clusters:
            if len(cluster) < self.min_cluster:
                continue
            members = sorted(cluster)
            exemplar = ids[members[0]]
            contributors = defaultdict(int)
            for m in members:
                cid = samples[m].contributor_id
                if cid:
                    contributors[cid] += 1

            size = len(members)
            # Confidence grows with cluster size. The smallest cluster we
            # report at all already sits above the review line: three
            # near-identical pictures is not a coincidence worth ignoring, and
            # scoring it below the threshold would mean finding the cluster and
            # then saying nothing about it.
            score = float(np.clip(0.55 + 0.05 * (size - self.min_cluster), 0.0, 0.97))

            pair_sims = [
                float(sims[a, b]) for ai, a in enumerate(members) for b in members[ai + 1 :]
            ]
            mean_sim = float(np.mean(pair_sims)) if pair_sims else 1.0

            who = ""
            if contributors:
                top = max(contributors.items(), key=lambda kv: kv[1])
                if top[1] >= max(2, int(0.6 * size)):
                    who = f" All but {size - top[1]} of them came from contributor {top[0]}."

            for m in members:
                findings.append(
                    Finding(
                        asset_ref=ids[m],
                        asset_type="sample",
                        attack_class="near_duplicate_flood",
                        detector_id=self.detector_id,
                        access_tier=ctx.access_tier,
                        raw_score=score,
                        severity=severity_for(score),
                        disposition=disposition_for(score),
                        reason=(
                            f"This picture is one of {size} near-identical copies of the "
                            f"same scene — they match each other {100 * mean_sim:.0f}% "
                            f"on every visual measure we use, which does not happen by "
                            f"chance.{who}"
                        ),
                        evidence={
                            "cluster_size": size,
                            "mean_similarity": round(mean_sim, 4),
                            "exemplar": exemplar,
                            "members": [ids[x] for x in members[:20]],
                            "contributor_breakdown": dict(contributors),
                        },
                    )
                )
        return findings


def _connected_components(adjacency: dict[int, set[int]], n: int) -> list[set[int]]:
    seen: set[int] = set()
    out: list[set[int]] = []
    for start in range(n):
        if start in seen or start not in adjacency:
            continue
        stack, comp = [start], set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            seen.add(node)
            stack.extend(adjacency.get(node, ()) - comp)
        if len(comp) > 1:
            out.append(comp)
    return out
