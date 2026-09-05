"""ImageFolder-style loader: the class is the name of the containing folder.

This is the workhorse for CIFAR-10 and GTSRB once they are laid out on disk,
and it is the format the attack harness writes.
"""

from __future__ import annotations

from pathlib import Path

from cvassure.core.schemas import Sample
from cvassure.ingest.coco import IMAGE_SUFFIXES


def is_folder_dataset(root: str | Path) -> bool:
    root = Path(root)
    if not root.is_dir():
        return False
    return any(
        d.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES for p in d.iterdir() if p.is_file())
        for d in root.iterdir()
    )


def load(root: str | Path) -> list[Sample]:
    root = Path(root)
    samples: list[Sample] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = p.relative_to(root)
        label = rel.parts[0] if len(rel.parts) > 1 else None
        samples.append(Sample(sample_id=str(rel), image_path=str(p), label=label))
    return samples
