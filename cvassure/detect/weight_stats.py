"""Backdoor-shaped anomalies in the weights [tier 1], with no trigger search.

This runs when there is no enrolled reference to compare against — the common
case for a model that arrives from a vendor with no history. It cannot say
"this changed"; it can say "this does not look like a normally trained
network", and say why.

Two signals: a final layer whose weight distribution has far heavier tails
than the others, and individual neurons with a wildly disproportionate pull
towards one class. A backdoor needs a strong, narrow path from the trigger to
the target class, and that path has to be visible somewhere.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis

from cvassure.core.schemas import Finding
from cvassure.detect.base import AuditContext, Detector
from cvassure.detect.embed import bounded, robust_z
from cvassure.ingest.models import AccessDenied


class WeightStatsDetector(Detector):
    detector_id = "weight_stats"
    required_tier = 1
    needs_model = True
    description = (
        "It looks for a lopsided final layer — the shape a hidden backdoor leaves "
        "in a network's numbers."
    )

    def run(self, ctx: AuditContext) -> list[Finding]:
        try:
            weights = ctx.model.weights()
        except AccessDenied as exc:
            return self.unavailable(ctx, exc.plain_english, str(exc))

        # The classifier head: the last 2-D parameter, which maps features to
        # classes. That is where a backdoor's shortcut has to terminate.
        candidates = [
            (name, np.asarray(w, dtype=np.float64))
            for name, w in sorted(weights.items())
            if np.asarray(w).ndim == 2 and np.asarray(w).size > 16
        ]
        if not candidates:
            return self.unavailable(
                ctx,
                "We could not check the model's final layer because this model does "
                "not have a layer of the shape we know how to read.",
                "no 2-D classifier layer found",
            )

        name, head = candidates[-1]
        n_classes = head.shape[0] if head.shape[0] <= head.shape[1] else head.shape[1]
        if head.shape[0] != n_classes:
            head = head.T

        # -- signal 1: how heavy-tailed is this layer, versus the others? --
        head_tail = float(kurtosis(head.ravel(), fisher=True))
        other_tails = [
            float(kurtosis(np.asarray(w, dtype=np.float64).ravel(), fisher=True))
            for n, w in sorted(weights.items())
            if n != name and np.asarray(w).size > 32
        ]
        baseline = float(np.median(other_tails)) if other_tails else 0.0
        tail_excess = head_tail - baseline

        # -- signal 2: does one class pull far harder than the others? -----
        class_strength = np.abs(head).sum(axis=1)
        strength_z = robust_z(class_strength)
        suspect_class = int(np.argmax(strength_z))
        strength_excess = float(strength_z[suspect_class])

        # -- signal 3: a single neuron dominating one class ---------------
        neuron_share = np.abs(head).max(axis=1) / (np.abs(head).sum(axis=1) + 1e-12)
        dominance = float(neuron_share[suspect_class])
        typical_dominance = float(np.median(neuron_share))

        combined = max(0.0, tail_excess) * 0.15 + max(0.0, strength_excess) * 0.8
        score = float(bounded(np.array([combined]), midpoint=3.5)[0])

        if score >= 0.5:
            reason = (
                f"The final layer of this model is lopsided: class {suspect_class} pulls "
                f"{strength_excess:.1f} times harder than a typical class does, and one "
                f"single connection accounts for {100 * dominance:.0f}% of that pull "
                f"(normally it is about {100 * typical_dominance:.0f}%). That is the "
                f"shape a hidden shortcut leaves — one narrow path that overrides "
                f"everything else. We have not proved a backdoor, but this model does "
                f"not look like one that was trained honestly."
            )
        else:
            reason = (
                f"The final layer of this model looks evenly balanced: no class pulls "
                f"more than {max(0.0, strength_excess):.1f} times harder than typical, "
                f"and no single connection dominates. Nothing here has the shape of a "
                f"hidden shortcut."
            )

        return [
            Finding(
                asset_ref="model",
                asset_type="model",
                attack_class="model_backdoor",
                detector_id=self.detector_id,
                access_tier=ctx.access_tier,
                raw_score=score,
                severity="high" if score >= 0.7 else ("medium" if score >= 0.45 else "low"),
                disposition="review" if score >= 0.5 else "accept",
                reason=reason,
                confidence=0.45,
                limitations=[
                    "This reports a shape, not proof. A lopsided final layer can "
                    "come from honest class imbalance as easily as from a backdoor.",
                    "It runs without any enrolled reference, so it cannot tell a "
                    "suspicious model from one that was always like this.",
                    "A tier-2 audit can reconstruct the trigger and settle it.",
                ],
                evidence={
                    "layer": name,
                    "n_classes": int(head.shape[0]),
                    "most_suspicious_class": suspect_class,
                    "class_pull_vs_typical": round(strength_excess, 3),
                    "tail_heaviness_vs_other_layers": round(tail_excess, 3),
                    "largest_single_connection_share": round(dominance, 4),
                    "typical_connection_share": round(typical_dominance, 4),
                    "note": "this check reports shape, not proof; a tier-2 audit can "
                            "reconstruct the trigger and settle it",
                },
            )
        ]
