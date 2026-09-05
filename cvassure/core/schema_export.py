"""Publish the assurance-report schema.

PS clause 2.3 lists "the assurance-report schema" as a submission deliverable
in its own right, and for good reason: a downstream system that consumes these
findings needs a contract it can validate against, not a Python class it has to
import.

The schema is generated from :class:`~cvassure.core.schemas.Finding` rather
than written by hand, for the same reason the coverage table is generated from
measurements — a hand-maintained copy drifts, and a schema that disagrees with
the data it describes is worse than none.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cvassure.core.schemas import (
    ASSET_TYPES,
    ATTACK_CLASSES,
    DISPOSITIONS,
    SEVERITIES,
)

SCHEMA_ID = "https://sih26228.local/cvassure/assurance-report.schema.json"
SCHEMA_VERSION = "1.0.0"

#: One sentence per field, aimed at whoever has to consume this downstream.
FIELD_DOCS: dict[str, str] = {
    "finding_id": "Unique id for this finding. Stable across a run, not across runs.",
    "asset_ref": "What this finding is about: an image path, a contributor id, "
                 "'model', or a receipt reference.",
    "asset_type": "Which kind of thing asset_ref names.",
    "attack_class": "The attack this finding would be evidence of, if true.",
    "detector_id": "Which check produced it.",
    "access_tier": "The access level the audit was granted: 0 answers only, "
                   "1 plus weights, 2 plus internals.",
    "raw_score": "How suspicious the asset is, 0..1, higher is worse. A ranking, "
                 "not a probability.",
    "calibrated_score": "raw_score mapped to an actual probability of being "
                        "poisoned, when a calibration split was available. Null "
                        "during a live audit, where there is no ground truth.",
    "confidence": "How sure we are of this finding, 0..1 — a different question "
                  "from raw_score. A hash mismatch is confidence 1.0; a "
                  "statistical shape argument is around 0.45 however alarming "
                  "the score.",
    "severity": "Graded from the score, for triage.",
    "disposition": "The recommended action.",
    "reason": "One human-readable sentence containing real numbers. Enforced: a "
              "finding whose reason contains jargon or no number is rejected at "
              "construction.",
    "evidence": "Detector-specific supporting values behind the reason.",
    "artefacts": "Paths to images produced as evidence, such as a saliency "
                 "heatmap or a reconstructed trigger.",
    "limitations": "What this particular assessment could not establish.",
    "unavailable_reason": "Set when the check could not run at all. Such a "
                          "finding is a status message, not a claim: its "
                          "raw_score is always 0 and must not be scored.",
}


def finding_schema() -> dict[str, Any]:
    """JSON Schema (draft 2020-12) for a single finding."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "cvassure assurance finding",
        "description": (
            "One statement about one asset, from one check, at one access tier. "
            "Every detector in cvassure returns a list of these and nothing else; "
            "the report, the scoring harness and the audit log all consume this "
            "one shape."
        ),
        "version": SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "finding_id", "asset_ref", "asset_type", "attack_class", "detector_id",
            "access_tier", "raw_score", "severity", "disposition", "reason",
        ],
        "properties": {
            "finding_id": {"type": "string", "minLength": 1,
                           "description": FIELD_DOCS["finding_id"]},
            "asset_ref": {"type": "string", "minLength": 1,
                          "description": FIELD_DOCS["asset_ref"]},
            "asset_type": {"enum": list(ASSET_TYPES),
                           "description": FIELD_DOCS["asset_type"]},
            "attack_class": {"enum": list(ATTACK_CLASSES),
                             "description": FIELD_DOCS["attack_class"]},
            "detector_id": {"type": "string", "minLength": 1,
                            "description": FIELD_DOCS["detector_id"]},
            "access_tier": {"enum": [0, 1, 2],
                            "description": FIELD_DOCS["access_tier"]},
            "raw_score": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                          "description": FIELD_DOCS["raw_score"]},
            "calibrated_score": {"type": ["number", "null"], "minimum": 0.0,
                                 "maximum": 1.0,
                                 "description": FIELD_DOCS["calibrated_score"]},
            "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0,
                           "description": FIELD_DOCS["confidence"]},
            "severity": {"enum": list(SEVERITIES),
                         "description": FIELD_DOCS["severity"]},
            "disposition": {"enum": list(DISPOSITIONS),
                            "description": FIELD_DOCS["disposition"]},
            "reason": {"type": "string", "minLength": 1, "maxLength": 400,
                       "description": FIELD_DOCS["reason"]},
            "evidence": {"type": "object",
                         "description": FIELD_DOCS["evidence"]},
            "artefacts": {"type": "array", "items": {"type": "string"},
                          "description": FIELD_DOCS["artefacts"]},
            "limitations": {"type": "array", "items": {"type": "string"},
                            "description": FIELD_DOCS["limitations"]},
            "unavailable_reason": {"type": ["string", "null"],
                                   "description": FIELD_DOCS["unavailable_reason"]},
        },
    }


def report_schema() -> dict[str, Any]:
    """The schema for a whole assurance report: findings plus the run's context."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID.replace("assurance-report", "assurance-run"),
        "title": "cvassure assurance run",
        "version": SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": ["generated_utc", "access_tier", "inputs", "findings", "coverage"],
        "properties": {
            "generated_utc": {"type": "string", "format": "date-time"},
            "access_tier": {"enum": [0, 1, 2]},
            "offline": {"type": "boolean",
                        "description": "True when the run asserted no network access."},
            "inputs": {
                "type": "object",
                "description": "Digests of everything the verdict was derived from.",
                "properties": {
                    "dataset_root": {"type": "string"},
                    "dataset_digest": {"type": "string"},
                    "n_samples": {"type": "integer", "minimum": 0},
                    "model_path": {"type": ["string", "null"]},
                    "model_file_digest": {"type": ["string", "null"]},
                    "receipts_path": {"type": ["string", "null"]},
                },
            },
            "verdict": {
                "type": "object",
                "required": ["headline", "colour"],
                "properties": {
                    "headline": {"type": "string"},
                    "colour": {"enum": ["green", "amber", "red"]},
                    "lines": {"type": "array", "items": {"type": "string"}},
                },
            },
            "findings": {"type": "array", "items": {"$ref": SCHEMA_ID}},
            "coverage": {
                "type": "array",
                "description": "Generated from what actually ran and what it "
                               "measured — never hand-written.",
                "items": {
                    "type": "object",
                    "required": ["attack_class", "access_tier", "status"],
                    "properties": {
                        "attack_class": {"enum": list(ATTACK_CLASSES)},
                        "access_tier": {"enum": [0, 1, 2]},
                        "status": {"enum": ["supported", "partial", "unsupported",
                                            "not measured"]},
                        "detector": {"type": ["string", "null"]},
                        "measured": {"type": ["string", "null"]},
                        "note": {"type": "string"},
                    },
                },
            },
            "audit_log": {
                "type": "object",
                "description": "The audit's own tamper-evident record.",
                "properties": {
                    "path": {"type": "string"},
                    "n_records": {"type": "integer"},
                    "verified": {"type": "boolean"},
                },
            },
        },
    }


def write(out_dir: str | Path = "docs") -> dict[str, Path]:
    """Write both schemas plus a human-readable description of them."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    finding_path = out / "assurance-finding.schema.json"
    run_path = out / "assurance-run.schema.json"
    finding_path.write_text(json.dumps(finding_schema(), indent=2) + "\n",
                            encoding="utf-8")
    run_path.write_text(json.dumps(report_schema(), indent=2) + "\n", encoding="utf-8")

    md = out / "ASSURANCE_SCHEMA.md"
    lines = [
        "# Assurance report schema",
        "",
        f"Version {SCHEMA_VERSION}. Generated from `cvassure.core.schemas.Finding` "
        "by `cvassure schema` — not maintained by hand, so it cannot drift away "
        "from the data it describes.",
        "",
        "Machine-readable: [`assurance-finding.schema.json`](assurance-finding.schema.json) "
        "and [`assurance-run.schema.json`](assurance-run.schema.json) "
        "(JSON Schema draft 2020-12).",
        "",
        "## A finding",
        "",
        "Every check returns a list of these and nothing else. `findings.jsonl` in "
        "any results directory is one finding per line.",
        "",
        "| field | type | required | meaning |",
        "| --- | --- | --- | --- |",
    ]
    schema = finding_schema()
    required = set(schema["required"])
    for name, spec in schema["properties"].items():
        if "enum" in spec:
            kind = " \\| ".join(f"`{v}`" for v in spec["enum"])
        else:
            t = spec.get("type", "")
            kind = "`" + ("`, `".join(t) if isinstance(t, list) else str(t)) + "`"
        lines.append(f"| `{name}` | {kind} | {'yes' if name in required else 'no'} "
                     f"| {spec.get('description', '')} |")

    lines += [
        "",
        "### Two fields people get the wrong way round",
        "",
        "`raw_score` is **how suspicious the asset is**. `confidence` is **how sure "
        "we are of the finding**. They are independent: a weight-digest mismatch is "
        "`raw_score` 1.0 and `confidence` 1.0 because a hash either matches or it "
        "does not, while a lopsided final layer may be `raw_score` 0.8 and "
        "`confidence` 0.45 because the shape is suggestive and nothing more.",
        "",
        "### Findings that are not claims",
        "",
        "A finding with `unavailable_reason` set is a status message — the check "
        "could not run at this access tier, or the input did not support it. Its "
        "`raw_score` is always 0 and it must be excluded from scoring. Treating one "
        "as a clean result is the single easiest way to misread this format.",
        "",
        "## A run",
        "",
        "`assurance-run.schema.json` wraps the findings with what they were derived "
        "from: input digests, the declared access tier, the traffic-light verdict, "
        "the generated coverage table, and the state of the audit log. Enough to "
        "re-check the verdict without re-running the audit.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"finding": finding_path, "run": run_path, "markdown": md}
