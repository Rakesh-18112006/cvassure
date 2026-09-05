"""Spectral signatures [access tier 2].

Poisoned samples that share a trigger tend to move together inside the model's
own representation, along a direction the clean data does not use. Projecting
each class onto that direction separates them.

At tier 0 or 1 this emits a clean UNAVAILABLE finding. It never crashes and it
never guesses.
"""

from __future__ import annotations

import numpy as np

from cvassure.core.schemas import Finding, disposition_for, severity_for
from cvassure.detect.base import AuditContext, Detector
from cvassure.detect.embed import bounded, robust_z
from cvassure.ingest.models import AccessDenied


class SpectralSignatureDetector(Detector):
    detector_id = "spectral_signature"
    required_tier = 2
    needs_model = True
    description = (
        "It looks inside the model as it runs, to see whether a group of images "
        "moves together in a way the rest of the data does not."
    )

    def __init__(self, min_class_size: int = 16, batch_size: int = 32):
        self.min_class_size = min_class_size
        self.batch_size = batch_size

    def run(self, ctx: AuditContext) -> list[Finding]:
        by_label = ctx.by_label()
        if not by_label:
            return self.unavailable(
                ctx,
                "We could not run the inside-the-model check because this dataset "
                "has no labels to group the images by.",
                "dataset has no labels",
            )

        findings: list[Finding] = []
        for label, members in sorted(by_label.items()):
            if len(members) < self.min_class_size:
                continue
            try:
                acts = self._activations(ctx, members)
            except AccessDenied as exc:
                return self.unavailable(ctx, exc.plain_english, str(exc))

            centred = acts - acts.mean(axis=0)
            try:
                _, _, vt = np.linalg.svd(centred, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            # The single direction along which this class varies most; a shared
            # trigger shows up as a group sitting far out along it.
            projection = np.abs(centred @ vt[0])
            z = robust_z(projection)
            scores = bounded(z, midpoint=3.0)

            for i, sample in enumerate(members):
                score = float(scores[i])
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
                        reason=(
                            f"Inside the model, this image sits {z[i]:.1f} times further "
                            f"out than a typical '{label}' image along the one direction "
                            f"that most separates this class — the pattern a shared "
                            f"hidden marker leaves behind."
                            if score >= 0.5
                            else (
                                f"Inside the model, this image behaves like the rest of "
                                f"the '{label}' images: {max(0.0, z[i]):.1f} times the "
                                f"typical spread, which is unremarkable."
                            )
                        ),
                        evidence={
                            "class": label,
                            "class_size": len(members),
                            "times_further_out_than_typical": round(float(z[i]), 2),
                        },
                    )
                )
        if not findings:
            return self.unavailable(
                ctx,
                "We could not run the inside-the-model check: every class here has "
                "too few images for the comparison to mean anything.",
                f"all classes smaller than {self.min_class_size}",
            )
        return findings

    def _activations(self, ctx: AuditContext, members) -> np.ndarray:
        out = []
        for i in range(0, len(members), self.batch_size):
            chunk = members[i : i + self.batch_size]
            batch = np.stack([self._prep(s.image_path, ctx) for s in chunk])
            out.append(ctx.model.activations(batch))
        return np.concatenate(out).astype(np.float64)

    @staticmethod
    def _prep(path: str, ctx: AuditContext) -> np.ndarray:
        from PIL import Image

        size = getattr(ctx, "model_input_size", 64)
        with Image.open(path) as im:
            img = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
