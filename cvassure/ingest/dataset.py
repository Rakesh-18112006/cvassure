"""Format detection and the one entry point the rest of cvassure uses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from cvassure.core.schemas import Sample
from cvassure.ingest import coco, folder, yolo
from cvassure.ingest.contributors import (
    attach_contributors,
    coverage,
    limitation_note,
    load_contributor_map,
)


def detect_format(root: str | Path) -> str:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset {root} does not exist")
    if coco.is_coco(root):
        return "coco"
    if yolo.is_yolo(root):
        return "yolo"
    if folder.is_folder_dataset(root):
        return "folder"
    raise ValueError(
        f"could not tell what format {root} is in. cvassure reads COCO "
        f"(instances JSON), YOLO (data.yaml + labels/) and ImageFolder layouts; "
        f"pass --format to say which."
    )


@dataclass
class Dataset:
    root: Path
    fmt: str
    samples: list[Sample]
    limitations: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    @property
    def labels(self) -> list[str]:
        return sorted({s.label for s in self.samples if s.label is not None})

    @property
    def contributors(self) -> list[str]:
        return sorted({s.contributor_id for s in self.samples if s.contributor_id})

    def by_id(self) -> dict[str, Sample]:
        return {s.sample_id: s for s in self.samples}

    def summary(self) -> dict[str, Any]:
        from collections import Counter

        attributed, total = coverage(self.samples)
        return {
            "root": str(self.root),
            "format": self.fmt,
            "n_samples": total,
            "n_labelled": sum(s.label is not None for s in self.samples),
            "classes": self.labels,
            "class_counts": dict(Counter(s.label for s in self.samples if s.label)),
            "n_contributors": len(self.contributors),
            "contributors": self.contributors,
            "contributor_coverage": f"{attributed}/{total}",
            "limitations": self.limitations,
        }


def load_dataset(
    root: str | Path,
    *,
    fmt: str = "auto",
    contributors: str | Path | None = None,
    contributor_from_path: str | None = None,
) -> Dataset:
    root = Path(root)
    fmt = detect_format(root) if fmt == "auto" else fmt

    if fmt == "coco":
        samples = coco.load(root)
    elif fmt == "yolo":
        samples = yolo.load(root)
    elif fmt == "folder":
        samples = folder.load(root)
    else:
        raise ValueError(f"unknown dataset format {fmt!r}")

    # An unspecified contributors.json sitting in the dataset root is the
    # common case, so look for it before giving up on attribution.
    if contributors is None and (root / "contributors.json").exists():
        contributors = root / "contributors.json"

    samples = attach_contributors(
        samples,
        contributor_map=load_contributor_map(contributors),
        path_regex=contributor_from_path,
    )

    ds = Dataset(root=root, fmt=fmt, samples=samples)
    note = limitation_note(samples)
    if note:
        ds.limitations.append(note)
    if not any(s.label for s in samples):
        ds.limitations.append(
            "label-consistency checks unavailable: this dataset has no labels"
        )
    missing = [s for s in samples if not Path(s.image_path).exists()]
    if missing:
        ds.limitations.append(
            f"{len(missing)} of {len(samples)} images referenced by the annotations "
            f"are not on disk and were skipped"
        )
        ds.samples = [s for s in samples if Path(s.image_path).exists()]
    return ds
