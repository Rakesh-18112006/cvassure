"""What every detector shares.

The contract: a detector is handed a context, returns a list of Findings, and
never raises. If it cannot run, it says so in a Finding with an
``unavailable_reason`` — a crash in front of a judge is a failure, an honest
"we could not check this" is not.
"""

from __future__ import annotations

import abc
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from cvassure.core.schemas import Finding, Sample


@dataclass
class AuditContext:
    """Everything a detector is allowed to see. Note what is absent: the
    answer key."""

    samples: list[Sample]
    access_tier: int = 0
    model: Any | None = None
    embeddings: Any | None = None  # EmbeddingStore
    reference_profile: dict[str, Any] | None = None
    enrolled_fingerprint: dict[str, Any] | None = None
    out_dir: Path = field(default_factory=lambda: Path("results"))
    seed: int = 0
    limitations: list[str] = field(default_factory=list)

    @property
    def artefact_dir(self) -> Path:
        d = Path(self.out_dir) / "artefacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def labelled(self) -> list[Sample]:
        return [s for s in self.samples if s.label is not None]

    def by_label(self) -> dict[str, list[Sample]]:
        out: dict[str, list[Sample]] = {}
        for s in self.samples:
            if s.label is not None:
                out.setdefault(s.label, []).append(s)
        return out


@dataclass
class DetectorResult:
    detector_id: str
    findings: list[Finding]
    seconds: float
    peak_ram_mb: float
    error: str | None = None


class Detector(abc.ABC):
    """Base class. Subclasses implement ``run``; ``safe_run`` wraps it."""

    detector_id: str = "unnamed"
    #: minimum access tier this detector needs to do its job
    required_tier: int = 0
    #: whether it needs a model handle at all
    needs_model: bool = False
    #: human sentence describing what it looks for
    description: str = ""

    @abc.abstractmethod
    def run(self, ctx: AuditContext) -> list[Finding]: ...

    # -- graceful degradation -------------------------------------------

    def unavailable(self, ctx: AuditContext, reason: str, why: str) -> list[Finding]:
        return [
            Finding.unavailable(
                asset_ref=self.detector_id,
                asset_type="dataset" if not self.needs_model else "model",
                attack_class="clean",
                detector_id=self.detector_id,
                access_tier=ctx.access_tier,
                reason=reason,
                unavailable_reason=why,
            )
        ]

    def check_preconditions(self, ctx: AuditContext) -> list[Finding] | None:
        """Return UNAVAILABLE findings if this detector cannot honestly run."""
        if self.needs_model and ctx.model is None:
            return self.unavailable(
                ctx,
                f"We could not run the '{self.detector_id}' check because no model "
                f"was supplied for this audit.",
                "no model supplied",
            )
        if ctx.access_tier < self.required_tier:
            return self.unavailable(
                ctx,
                f"We could not run the '{self.detector_id}' check. "
                f"{self.description} That needs deeper access to the model than "
                f"this audit was given.",
                f"requires access tier {self.required_tier}",
            )
        if self.needs_model and not ctx.model.can(self.required_tier):
            return self.unavailable(
                ctx,
                f"We could not run the '{self.detector_id}' check because the model "
                f"was supplied in a form that does not expose what it needs.",
                f"model does not support tier {self.required_tier}",
            )
        if not self.needs_model and not ctx.samples:
            return self.unavailable(
                ctx,
                f"We could not run the '{self.detector_id}' check because there were "
                f"no images to look at.",
                "no samples",
            )
        return None

    def safe_run(self, ctx: AuditContext) -> DetectorResult:
        """Run, timing and memory-tracking, and never propagate an exception."""
        import tracemalloc

        blocked = self.check_preconditions(ctx)
        if blocked is not None:
            return DetectorResult(self.detector_id, blocked, 0.0, 0.0)

        tracemalloc.start()
        start = time.perf_counter()
        error = None
        try:
            findings = self.run(ctx)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            findings = self.unavailable(
                ctx,
                f"The '{self.detector_id}' check could not finish on this input, so "
                f"we are reporting nothing rather than guessing. The audit "
                f"continued with the other checks.",
                f"detector failed: {error}",
            )
            traceback.print_exc()
        seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return DetectorResult(
            self.detector_id, findings, seconds, peak / (1024 * 1024), error
        )


def top_k(values: Sequence[float], k: int) -> list[int]:
    import numpy as np

    v = np.asarray(values)
    k = min(k, v.size)
    return np.argsort(-v)[:k].tolist()
