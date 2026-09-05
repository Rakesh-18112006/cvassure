"""Working out who supplied each sample.

Contributor-level blame is the part of PS clause 2.2.1 that an officer can act
on, but the metadata is often simply not there. When it is missing we say so
in the report rather than inventing an attribution.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

from cvassure.core.schemas import Sample

NO_CONTRIBUTOR_METADATA = (
    "source-level assessment unavailable: no contributor metadata provided"
)


def load_contributor_map(path: str | Path | None) -> dict[str, str]:
    """Read contributors.json: ``{sample_id: contributor_id}``.

    Also accepts the inverted layout ``{contributor_id: [sample_id, ...]}``,
    because that is how people usually write it by hand.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"contributor map {p} does not exist")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{p} must contain a JSON object")
    mapping: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            mapping[str(key)] = value
        elif isinstance(value, (list, tuple)):
            for sample_id in value:
                mapping[str(sample_id)] = str(key)
        else:
            raise ValueError(f"{p}: unexpected value for {key!r}")
    return mapping


def contributor_from_path(pattern: str, image_path: str) -> str | None:
    """Pull a contributor id out of a file path with a user-supplied regex."""
    m = re.search(pattern, str(image_path))
    if not m:
        return None
    return m.group(1) if m.groups() else m.group(0)


def attach_contributors(
    samples: Sequence[Sample],
    *,
    contributor_map: dict[str, str] | None = None,
    path_regex: str | None = None,
) -> list[Sample]:
    """Fill in ``contributor_id``, preferring explicit metadata over the regex."""
    contributor_map = contributor_map or {}
    out: list[Sample] = []
    for s in samples:
        cid = contributor_map.get(s.sample_id)
        if cid is None and path_regex:
            cid = contributor_from_path(path_regex, s.image_path)
        if cid is None:
            cid = s.contributor_id
        out.append(
            Sample(
                sample_id=s.sample_id,
                image_path=s.image_path,
                label=s.label,
                contributor_id=cid,
                batch_id=s.batch_id,
            )
        )
    return out


def coverage(samples: Iterable[Sample]) -> tuple[int, int]:
    """(number with a contributor, total)."""
    total = attributed = 0
    for s in samples:
        total += 1
        attributed += s.contributor_id is not None
    return attributed, total


def limitation_note(samples: Sequence[Sample]) -> str | None:
    """The sentence to print in the report when attribution is incomplete."""
    attributed, total = coverage(samples)
    if total == 0 or attributed == total:
        return None
    if attributed == 0:
        return NO_CONTRIBUTOR_METADATA
    return (
        f"source-level assessment partial: {total - attributed} of {total} samples "
        f"have no contributor recorded and are excluded from contributor scoring"
    )
