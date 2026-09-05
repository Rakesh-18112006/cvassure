"""Labels that disagree with the pictures around them.

A label flip leaves the pixels untouched, so nothing that inspects an image on
its own can see it. What gives it away is company: if the twenty images most
similar to this one are all labelled 'car' and this one says 'truck', the
label is the thing that is wrong.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from cvassure.core.schemas import Finding, disposition_for, severity_for
from cvassure.detect.base import AuditContext, Detector
from cvassure.detect.embed import knn


class LabelNoiseDetector(Detector):
    detector_id = "label_noise"
    required_tier = 0
    description = (
        "It looks for images whose label disagrees with the labels of the images "
        "that look most like them."
    )

    def __init__(self, k: int = 20, min_samples: int = 20):
        self.k = k
        self.min_samples = min_samples

    def run(self, ctx: AuditContext) -> list[Finding]:
        labelled = ctx.labelled()
        if len(labelled) < self.min_samples:
            return self.unavailable(
                ctx,
                "We could not check labels against their neighbours: there are too "
                "few labelled images here for a comparison to mean anything.",
                f"fewer than {self.min_samples} labelled samples",
            )
        if len({s.label for s in labelled}) < 2:
            return self.unavailable(
                ctx,
                "We could not check labels against their neighbours because every "
                "image carries the same label — there is nothing to disagree with.",
                "only one class present",
            )

        labels = np.array([s.label for s in labelled], dtype=object)
        X = ctx.embeddings.embed([s.image_path for s in labelled])

        k = min(self.k, len(labelled) - 1)
        neighbour_idx, neighbour_sim = knn(X, k)

        findings: list[Finding] = []
        for i, sample in enumerate(labelled):
            neigh_labels = labels[neighbour_idx[i]]
            counts = Counter(neigh_labels.tolist())
            agree = counts.get(sample.label, 0)
            agreement = agree / k
            majority, majority_n = counts.most_common(1)[0]

            # Disagreement only counts when the neighbourhood is coherent: if
            # its own neighbours cannot agree either, this is a hard region of
            # the data, not a mislabelled image.
            coherence = majority_n / k
            score = float(np.clip((1.0 - agreement) * coherence, 0.0, 1.0))

            if score >= 0.5 and majority != sample.label:
                reason = (
                    f"This image is filed as '{sample.label}', but {majority_n} of its "
                    f"{k} closest look-alikes are labelled '{majority}' and only "
                    f"{agree} agree with the label it carries. The picture itself has "
                    f"not been altered — the label is the part that looks wrong."
                )
            else:
                reason = (
                    f"This image is filed as '{sample.label}' and {agree} of its {k} "
                    f"closest look-alikes carry the same label, which is normal."
                )

            findings.append(
                Finding(
                    asset_ref=sample.sample_id,
                    asset_type="sample",
                    attack_class="label_flip",
                    detector_id=self.detector_id,
                    access_tier=ctx.access_tier,
                    raw_score=score,
                    severity=severity_for(score),
                    disposition=disposition_for(score),
                    reason=reason,
                    evidence={
                        "stored_label": sample.label,
                        "neighbour_labels": neigh_labels.tolist(),
                        "agreement_fraction": round(agreement, 3),
                        "majority_label": majority,
                        "majority_fraction": round(coherence, 3),
                        "k": k,
                        "mean_neighbour_similarity": round(
                            float(np.mean(neighbour_sim[i])), 4
                        ),
                    },
                )
            )
        return findings
