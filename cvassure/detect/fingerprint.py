"""Model fingerprinting [tier 0 — works black-box].

The strongest result in the model half of this system, and the one that
surprises people: you can detect that a model has been swapped without ever
looking inside it. Ask a fixed, seeded set of questions, record the answers,
and compare. Two different networks do not agree on 200 arbitrary inputs.

The fingerprint is enrolled once when the model is accepted, and checked on
every audit thereafter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cvassure.core.hashing import sha256_hex
from cvassure.core.schemas import Finding
from cvassure.detect.base import AuditContext, Detector

DEFAULT_N_PROBES = 200
DEFAULT_SEED = 20260101


def probe_images(n: int, shape: tuple[int, ...], seed: int) -> np.ndarray:
    """The questions. Fixed by a seed, so the same probes are used every time
    and the fingerprint is comparable across runs and across machines."""
    rng = np.random.default_rng(seed)
    return rng.random((n, *shape), dtype=np.float32)


def compute_fingerprint(
    model, *, n_probes: int = DEFAULT_N_PROBES, seed: int = DEFAULT_SEED,
    input_shape: tuple[int, ...] = (3, 64, 64), batch_size: int = 32,
) -> dict[str, Any]:
    probes = probe_images(n_probes, input_shape, seed)
    outputs = []
    for i in range(0, n_probes, batch_size):
        outputs.append(np.asarray(model.predict(probes[i : i + batch_size])))
    out = np.concatenate(outputs).astype(np.float64)
    flat = out.reshape(out.shape[0], -1)
    # Normalise per probe so an overall rescaling of the outputs does not
    # register as a different model.
    norm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)
    vector = norm.ravel()
    return {
        "n_probes": int(n_probes),
        "seed": int(seed),
        "input_shape": list(input_shape),
        "output_shape": list(out.shape[1:]),
        "vector": vector.tolist(),
        "digest": sha256_hex(np.ascontiguousarray(vector).tobytes()),
        "argmax_labels": flat.argmax(axis=1).tolist(),
    }


def save_fingerprint(fp: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fp), encoding="utf-8")
    return p


def load_fingerprint(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(enrolled: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    a = np.asarray(enrolled["vector"], dtype=np.float64)
    b = np.asarray(observed["vector"], dtype=np.float64)
    if a.shape != b.shape:
        return {
            "comparable": False,
            "cosine_distance": 1.0,
            "agreement": 0.0,
            "note": "the model now produces a different shape of answer entirely",
        }
    cos = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
    la = np.asarray(enrolled.get("argmax_labels", []))
    lb = np.asarray(observed.get("argmax_labels", []))
    agreement = float((la == lb).mean()) if la.size and la.shape == lb.shape else float("nan")
    return {
        "comparable": True,
        "cosine_distance": float(np.clip(1.0 - cos, 0.0, 2.0)),
        "agreement": agreement,
        "identical": enrolled["digest"] == observed["digest"],
    }


class FingerprintDetector(Detector):
    detector_id = "fingerprint"
    required_tier = 0
    needs_model = True
    description = (
        "It asks the model a fixed set of 200 questions and checks that the answers "
        "match the ones recorded when the model was accepted."
    )

    #: Below this, a difference is arithmetic noise — a different BLAS library,
    #: a different CPU. Comparing a model against itself measures around 1e-15,
    #: so there is a wide margin here.
    NOISE_FLOOR = 1e-7
    #: At and above this, the replies differ far more than any rounding could
    #: explain, and the only remaining explanation is a different model.
    CERTAIN = 1e-3

    def __init__(self, n_probes: int = DEFAULT_N_PROBES, seed: int = DEFAULT_SEED):
        self.n_probes = n_probes
        self.seed = seed

    def score_for(self, distance: float) -> float:
        """Grade a fingerprint difference on a log scale.

        Inference on fixed inputs is deterministic, so the honest question is
        not "how big is the difference" but "is there a difference at all,
        beyond arithmetic noise". Agreement on the top answer is a much weaker
        signal than it looks: two different networks can both be confidently
        wrong in the same direction on artificial inputs and agree on every
        single one, while their actual output numbers differ by 6%. The
        distance is what separates them, so the distance is what we score.
        """
        d = float(max(distance, 0.0))
        if d <= self.NOISE_FLOOR:
            return 0.0
        span = np.log10(self.CERTAIN / self.NOISE_FLOOR)
        return float(np.clip(np.log10(d / self.NOISE_FLOOR) / span, 0.0, 1.0))

    def run(self, ctx: AuditContext) -> list[Finding]:
        shape = tuple(
            ctx.enrolled_fingerprint.get("input_shape", (3, 64, 64))
            if ctx.enrolled_fingerprint
            else (3, 64, 64)
        )
        observed = compute_fingerprint(
            ctx.model, n_probes=self.n_probes, seed=self.seed, input_shape=shape
        )

        if not ctx.enrolled_fingerprint:
            return [
                Finding.unavailable(
                    asset_ref="model",
                    asset_type="model",
                    attack_class="model_substitute",
                    detector_id=self.detector_id,
                    access_tier=ctx.access_tier,
                    reason=(
                        "We recorded this model's fingerprint, but there is nothing to "
                        "compare it against — no fingerprint was enrolled when the "
                        "model was first accepted. Keep this one, and future audits "
                        "will be able to prove the model has not been swapped."
                    ),
                    unavailable_reason="no enrolled fingerprint supplied",
                    evidence={"observed_digest": observed["digest"],
                              "n_probes": observed["n_probes"]},
                )
            ]

        cmp = compare(ctx.enrolled_fingerprint, observed)

        if not cmp["comparable"]:
            score, disposition, severity = 1.0, "quarantine", "critical"
            reason = (
                f"The model no longer answers in the same form it did when it was "
                f"accepted. This is not the model that was enrolled. We asked it the "
                f"same {self.n_probes} questions and it replied in a different shape "
                f"entirely."
            )
        elif cmp["identical"]:
            score, disposition, severity = 0.0, "accept", "low"
            reason = (
                f"The model answered all {self.n_probes} fixed test questions exactly "
                f"as it did when it was accepted. It has not been swapped or edited."
            )
        else:
            agreement = cmp["agreement"]
            distance = cmp["cosine_distance"]
            score = self.score_for(distance)
            changed_answers = (
                int(round((1 - agreement) * self.n_probes))
                if np.isfinite(agreement)
                else None
            )
            also = (
                f" It also picked a different answer outright on {changed_answers} of "
                f"them."
                if changed_answers
                else ""
            )
            if score >= 0.8:
                disposition, severity = "quarantine", "critical"
                reason = (
                    f"This is not the model that was accepted. We put the same "
                    f"{self.n_probes} fixed test questions to it, and its replies differ "
                    f"from the recorded ones by {100 * distance:.1f}% — where the same "
                    f"model asked the same questions replies identically, every time, to "
                    f"the last decimal place.{also} The file has been replaced or edited."
                )
            elif score >= 0.5:
                disposition, severity = "review", "high"
                reason = (
                    f"The model's replies have shifted since it was accepted — by "
                    f"{100 * distance:.4f}% across {self.n_probes} fixed test questions. "
                    f"That is far too small to be a swap and far too large to be rounding, "
                    f"which is what editing a few weights looks like.{also}"
                )
            else:
                disposition, severity = "accept", "low"
                reason = (
                    f"The model's replies to all {self.n_probes} fixed test questions "
                    f"match the ones recorded when it was accepted, to within "
                    f"{100 * distance:.6f}% — the difference you get from arithmetic "
                    f"rounding, not from a changed model."
                )

        return [
            Finding(
                asset_ref="model",
                asset_type="model",
                attack_class="model_substitute",
                detector_id=self.detector_id,
                access_tier=ctx.access_tier,
                raw_score=score,
                severity=severity,
                disposition=disposition,
                reason=reason,
                evidence={
                    "n_probes": self.n_probes,
                    "probe_seed": self.seed,
                    "enrolled_digest": ctx.enrolled_fingerprint.get("digest"),
                    "observed_digest": observed["digest"],
                    "answer_agreement": None if not np.isfinite(cmp.get("agreement", np.nan))
                    else round(cmp["agreement"], 4),
                    "distance": round(cmp["cosine_distance"], 6),
                    "works_without_internal_access": True,
                },
            )
        ]
