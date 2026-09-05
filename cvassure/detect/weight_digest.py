"""Weight integrity [tier 1].

A whole-file hash answers "has anything changed?" — which is necessary, but
useless for triage, because the answer is a single bit. The per-layer
statistical digest answers the question an analyst actually asks next: *which*
layers moved, and by how much. A backdoor injected by fine-tuning usually
shows up concentrated in the last layer or two, and this table shows that.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cvassure.core.hashing import sha256_hex
from cvassure.core.schemas import Finding
from cvassure.detect.base import AuditContext, Detector
from cvassure.ingest.models import AccessDenied


def canonical_weight_digest(weights: dict[str, np.ndarray]) -> str:
    """One hash over every parameter, in a fixed order, at a fixed precision."""
    parts = []
    for name in sorted(weights):
        arr = np.ascontiguousarray(np.asarray(weights[name], dtype=np.float64))
        parts.append(name.encode("utf-8"))
        parts.append(str(arr.shape).encode("utf-8"))
        parts.append(arr.tobytes())
    return sha256_hex(b"".join(parts))


def layer_table(model) -> list[dict[str, Any]]:
    return [s.to_dict() for s in model.layer_stats()]


def diff_layers(
    enrolled: list[dict[str, Any]], observed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per-layer comparison, expressed in units of that layer's own spread so
    a big layer and a small one are comparable."""
    by_name = {r["name"]: r for r in enrolled}
    out = []
    for row in observed:
        before = by_name.get(row["name"])
        if before is None:
            out.append({"name": row["name"], "status": "new layer",
                        "shift_in_std_units": float("inf")})
            continue
        scale = before["std"] or 1.0
        shift = abs(row["mean"] - before["mean"]) / scale
        spread_change = abs(row["std"] - before["std"]) / scale
        out.append(
            {
                "name": row["name"],
                "status": "unchanged" if row["digest"] == before["digest"] else "changed",
                "shift_in_std_units": round(float(shift), 4),
                "spread_change": round(float(spread_change), 4),
                "n_params": row["n_params"],
            }
        )
    for name in by_name:
        if not any(r["name"] == name for r in observed):
            out.append({"name": name, "status": "removed", "shift_in_std_units": float("inf")})
    return out


class WeightDigestDetector(Detector):
    detector_id = "weight_digest"
    required_tier = 1
    needs_model = True
    description = "It hashes the model's weights and compares them layer by layer."

    def run(self, ctx: AuditContext) -> list[Finding]:
        try:
            weights = ctx.model.weights()
            observed_layers = layer_table(ctx.model)
        except AccessDenied as exc:
            return self.unavailable(ctx, exc.plain_english, str(exc))

        digest = canonical_weight_digest(weights)
        enrolled = (ctx.enrolled_fingerprint or {}).get("weight_digest")
        enrolled_layers = (ctx.enrolled_fingerprint or {}).get("layer_stats")

        n_params = int(sum(int(np.asarray(w).size) for w in weights.values()))

        if enrolled is None:
            return [
                Finding.unavailable(
                    asset_ref="model",
                    asset_type="model",
                    attack_class="model_perturb",
                    detector_id=self.detector_id,
                    access_tier=ctx.access_tier,
                    reason=(
                        f"We recorded a fingerprint of all {n_params:,} numbers inside "
                        f"this model across {len(observed_layers)} layers, but there is "
                        f"no earlier record to compare it with. Keep this, and any "
                        f"later edit to the model becomes provable."
                    ),
                    unavailable_reason="no enrolled weight digest supplied",
                    evidence={"weight_digest": digest, "n_parameters": n_params,
                              "n_layers": len(observed_layers)},
                )
            ]

        if digest == enrolled:
            return [
                Finding(
                    asset_ref="model",
                    asset_type="model",
                    attack_class="model_perturb",
                    detector_id=self.detector_id,
                    access_tier=ctx.access_tier,
                    raw_score=0.0,
                    severity="low",
                    disposition="accept",
                    reason=(
                        f"Every one of the {n_params:,} numbers inside this model is "
                        f"exactly as it was when the model was accepted. Not a single "
                        f"weight has been altered."
                    ),
                    evidence={"weight_digest": digest, "n_parameters": n_params},
                )
            ]

        diffs = diff_layers(enrolled_layers or [], observed_layers)
        changed = [d for d in diffs if d.get("status") != "unchanged"]
        worst = sorted(changed, key=lambda d: -d.get("shift_in_std_units", 0))[:5]

        detail = (
            f" The biggest change is in '{worst[0]['name']}', where the weights have "
            f"moved {worst[0]['shift_in_std_units']:.2f} times their own normal spread."
            if worst and np.isfinite(worst[0].get("shift_in_std_units", 0))
            else ""
        )
        return [
            Finding(
                asset_ref="model",
                asset_type="model",
                attack_class="model_perturb",
                detector_id=self.detector_id,
                access_tier=ctx.access_tier,
                raw_score=1.0,
                severity="critical",
                disposition="quarantine",
                reason=(
                    f"This model's weights have been changed since it was accepted. "
                    f"{len(changed)} of its {len(diffs)} layers no longer match."
                    f"{detail} This is a hash comparison, not an estimate — the numbers "
                    f"either match or they do not."
                ),
                evidence={
                    "enrolled_digest": enrolled,
                    "observed_digest": digest,
                    "n_layers_changed": len(changed),
                    "n_layers_total": len(diffs),
                    "per_layer_diff": diffs,
                },
            )
        ]
