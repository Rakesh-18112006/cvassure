"""COCO instances-JSON loader (PS clause 2.2.6)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from cvassure.core.schemas import Sample

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_annotation_file(root: Path) -> Path | None:
    """Locate instances_*.json without requiring a fixed layout."""
    if root.is_file() and root.suffix == ".json":
        return root
    candidates = [
        *root.glob("annotations/instances*.json"),
        *root.glob("annotations/*.json"),
        *root.glob("instances*.json"),
        *root.glob("*.json"),
    ]
    for c in candidates:
        if c.name in {"contributors.json", "ground_truth.json"}:
            continue
        return c
    return None


def is_coco(root: str | Path) -> bool:
    root = Path(root)
    ann = find_annotation_file(root)
    if ann is None:
        return False
    try:
        head = json.loads(ann.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(head, dict) and {"images", "annotations"} <= set(head)


def _resolve_image(root: Path, ann_path: Path, file_name: str) -> Path:
    """COCO file_name is relative to an images root that nobody agrees on."""
    for base in (
        root / "images",
        root,
        ann_path.parent,
        ann_path.parent.parent,
        ann_path.parent.parent / "images",
    ):
        p = base / file_name
        if p.exists():
            return p
    return root / "images" / file_name


def load(root: str | Path, *, annotations: str | Path | None = None) -> list[Sample]:
    """Return one Sample per image.

    cvassure reasons about images, not boxes, so an image's label is the
    category that occupies the most annotations on it. Images with no
    annotations get ``label=None`` rather than being dropped — an unlabelled
    image is still something a contributor sent us.
    """
    root = Path(root)
    ann_path = Path(annotations) if annotations else find_annotation_file(root)
    if ann_path is None or not ann_path.exists():
        raise FileNotFoundError(f"no COCO annotation JSON found under {root}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}

    per_image: dict[int, Counter] = {}
    for a in data.get("annotations", []):
        per_image.setdefault(a["image_id"], Counter())[
            categories.get(a["category_id"], str(a["category_id"]))
        ] += 1

    samples: list[Sample] = []
    for img in data.get("images", []):
        counts = per_image.get(img["id"])
        label = counts.most_common(1)[0][0] if counts else None
        samples.append(
            Sample(
                sample_id=str(img.get("file_name", img["id"])),
                image_path=str(_resolve_image(root, ann_path, img["file_name"])),
                label=label,
                contributor_id=img.get("contributor_id"),
                batch_id=img.get("batch_id"),
            )
        )
    return samples
