"""YOLO folder-layout loader (PS clause 2.2.6).

Expected shape::

    root/
      data.yaml            names: [car, truck, ...]
      images/train/*.jpg
      labels/train/*.txt   "<class_id> cx cy w h" per line
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from cvassure.core.schemas import Sample
from cvassure.ingest.coco import IMAGE_SUFFIXES


def find_data_yaml(root: Path) -> Path | None:
    for name in ("data.yaml", "data.yml", "dataset.yaml"):
        p = root / name
        if p.exists():
            return p
    found = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    return found[0] if found else None


def is_yolo(root: str | Path) -> bool:
    root = Path(root)
    if find_data_yaml(root) is None:
        return False
    return (root / "labels").exists() or any(root.glob("**/labels"))


def class_names(root: str | Path) -> dict[int, str]:
    root = Path(root)
    p = find_data_yaml(root)
    if p is None:
        return {}
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def _label_path(image_path: Path) -> Path:
    """images/train/x.jpg -> labels/train/x.txt"""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def load(root: str | Path) -> list[Sample]:
    root = Path(root)
    names = class_names(root)

    image_dirs = [d for d in (root / "images",) if d.exists()] or [root]
    image_paths = sorted(
        p
        for d in image_dirs
        for p in d.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
    )

    samples: list[Sample] = []
    for img in image_paths:
        label = None
        lp = _label_path(img)
        if lp.exists():
            counts: Counter = Counter()
            for line in lp.read_text(encoding="utf-8").splitlines():
                bits = line.split()
                if bits:
                    try:
                        cid = int(float(bits[0]))
                    except ValueError:
                        continue
                    counts[names.get(cid, str(cid))] += 1
            if counts:
                label = counts.most_common(1)[0][0]
        rel = img.relative_to(root)
        # split name (train/val) is a useful natural batch id
        batch = rel.parts[1] if len(rel.parts) > 2 and rel.parts[0] == "images" else None
        samples.append(
            Sample(sample_id=str(rel), image_path=str(img), label=label, batch_id=batch)
        )
    return samples
