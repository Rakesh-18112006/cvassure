"""Shared contracts.

Every detector in cvassure returns a list of :class:`Finding`. Nothing else.
The scoring harness, the HTML report and the audit log all consume this one
shape, which is why it is frozen and validated at construction time.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

AssetType = Literal["sample", "contributor", "model", "receipt", "dataset"]
Severity = Literal["low", "medium", "high", "critical"]
Disposition = Literal["accept", "review", "quarantine"]
AccessTier = Literal[0, 1, 2]

SEVERITIES: tuple[str, ...] = ("low", "medium", "high", "critical")
DISPOSITIONS: tuple[str, ...] = ("accept", "review", "quarantine")
ASSET_TYPES: tuple[str, ...] = ("sample", "contributor", "model", "receipt", "dataset")

#: Attack classes the system knows about. Ground truth and findings join on
#: these strings, so they are defined once, here.
ATTACK_CLASSES: tuple[str, ...] = (
    "clean",
    "badnets_patch",
    "blended_trigger",
    "label_flip",
    "systematic_mislabel",
    "near_duplicate_flood",
    "ood_insertion",
    "model_substitute",
    "model_perturb",
    "model_backdoor",
    "receipt_alter",
    "receipt_replay",
    "receipt_delete",
    "receipt_reorder",
    "distribution_shift",
)

# --------------------------------------------------------------------------
# Plain-English enforcement (principle 1 of the build spec)
# --------------------------------------------------------------------------

#: Jargon that must never reach an evaluator's screen. If a detector wants to
#: say one of these things, it must say it in words instead.
BANNED_JARGON: tuple[str, ...] = (
    "anomalous embedding",
    "mahalanobis",
    "latent space",
    "penultimate activation",
    "eigenvector",
    "l2 norm",
    "kurtosis",
    "z-score",
    "logit",
    "softmax",
    "p-value",
    "cosine similarity",
    "hamming distance",
    "outlier score",
    "feature vector",
)

_DIGIT = re.compile(r"\d")


class PlainEnglishError(ValueError):
    """Raised when a ``reason`` string would not be readable by a judge."""


def check_plain_english(reason: str, *, require_number: bool = True) -> None:
    """Reject reason strings that a non-technical evaluator could not read.

    This is deliberately a hard runtime check rather than a review convention:
    conventions rot under deadline, assertions do not.
    """
    text = reason.strip()
    if not text:
        raise PlainEnglishError("reason must not be empty")
    lowered = text.lower()
    for phrase in BANNED_JARGON:
        if phrase in lowered:
            raise PlainEnglishError(
                f"reason contains jargon {phrase!r}; rewrite it for a reader "
                f"with no machine-learning background: {text!r}"
            )
    if require_number and not _DIGIT.search(text):
        raise PlainEnglishError(
            f"reason must quote a real number so the claim is checkable: {text!r}"
        )
    if len(text) > 400:
        raise PlainEnglishError("reason must be one sentence, not an essay")


# --------------------------------------------------------------------------
# The Finding
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Finding:
    """One statement about one asset, from one detector, at one access tier."""

    asset_ref: str
    asset_type: AssetType
    attack_class: str
    detector_id: str
    access_tier: int
    raw_score: float
    severity: Severity
    disposition: Disposition
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    artefacts: list[str] = field(default_factory=list)
    unavailable_reason: str | None = None
    calibrated_score: float | None = None
    #: How sure we are of *this* finding, 0..1 — distinct from ``raw_score``,
    #: which is how suspicious the asset is. A hash comparison that does not
    #: match is confidence 1.0 at any score; a statistical shape argument is
    #: confidence 0.5 however alarming the number. PS clause 2.2.2 requires
    #: model assessments to state this.
    confidence: float | None = None
    #: What this particular assessment could *not* establish. Also required by
    #: PS 2.2.2, and printed beside the finding rather than buried in a
    #: methodology section nobody reads.
    limitations: list[str] = field(default_factory=list)
    finding_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.asset_type not in ASSET_TYPES:
            raise ValueError(f"bad asset_type {self.asset_type!r}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"bad severity {self.severity!r}")
        if self.disposition not in DISPOSITIONS:
            raise ValueError(f"bad disposition {self.disposition!r}")
        if self.access_tier not in (0, 1, 2):
            raise ValueError(f"bad access_tier {self.access_tier!r}")
        if not 0.0 <= float(self.raw_score) <= 1.0:
            raise ValueError(
                f"raw_score must be in 0..1 (higher = more suspicious), "
                f"got {self.raw_score!r}"
            )
        if self.calibrated_score is not None and not 0.0 <= float(self.calibrated_score) <= 1.0:
            raise ValueError("calibrated_score must be in 0..1")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in 0..1")
        # An UNAVAILABLE finding is allowed to be numberless — it is a status
        # message, not a claim about the data.
        check_plain_english(self.reason, require_number=self.unavailable_reason is None)

    # -- convenience ------------------------------------------------------

    @property
    def is_unavailable(self) -> bool:
        return self.unavailable_reason is not None

    @property
    def score(self) -> float:
        """The score downstream consumers should use: calibrated if we have it."""
        return float(self.calibrated_score if self.calibrated_score is not None else self.raw_score)

    def with_calibration(self, value: float) -> "Finding":
        return replace(self, calibrated_score=float(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Finding":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def unavailable(
        cls,
        *,
        asset_ref: str,
        asset_type: AssetType,
        attack_class: str,
        detector_id: str,
        access_tier: int,
        reason: str,
        unavailable_reason: str,
        evidence: dict[str, Any] | None = None,
    ) -> "Finding":
        """A clean 'this check could not run' result — never a crash, never a
        made-up number."""
        return cls(
            asset_ref=asset_ref,
            asset_type=asset_type,
            attack_class=attack_class,
            detector_id=detector_id,
            access_tier=access_tier,
            raw_score=0.0,
            severity="low",
            disposition="accept",
            reason=reason,
            evidence=evidence or {},
            artefacts=[],
            unavailable_reason=unavailable_reason,
        )


# --------------------------------------------------------------------------
# Sample — the common currency of every dataset loader
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    image_path: str
    label: str | None = None
    contributor_id: str | None = None
    batch_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Severity / disposition helpers so every detector grades the same way
# --------------------------------------------------------------------------

def severity_for(score: float) -> Severity:
    if score >= 0.90:
        return "critical"
    if score >= 0.70:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def disposition_for(score: float) -> Disposition:
    if score >= 0.80:
        return "quarantine"
    if score >= 0.50:
        return "review"
    return "accept"


__all__ = [
    "ATTACK_CLASSES",
    "BANNED_JARGON",
    "Finding",
    "PlainEnglishError",
    "Sample",
    "check_plain_english",
    "disposition_for",
    "severity_for",
]
