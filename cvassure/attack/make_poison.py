"""Deliberately break a clean dataset, and keep the answer key.

You cannot measure a detector without knowing the right answer, so we poison
the data ourselves and record exactly what we touched. PS clause 2.3 asks for
this explicitly.

Two rules that are enforced, not just intended:

1. Every attack takes a seed and is exactly reproducible — the same seed
   produces byte-identical ``ground_truth.json``.
2. ``ground_truth.json`` is written to a separate folder, and nothing in
   ``cvassure/detect/`` may import from ``cvassure/attack/``. There is a test
   that greps for it and fails the build if anybody forgets.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from PIL import Image

from cvassure.attack import image_ops
from cvassure.core.hashing import canonical_json
from cvassure.ingest.coco import IMAGE_SUFFIXES
from cvassure.ingest.dataset import load_dataset

GROUND_TRUTH_NAME = "ground_truth.json"


def _sub_rng(seed: int, *parts: Any) -> np.random.Generator:
    """A generator keyed by content, not by call order.

    This is what makes the harness order-independent: sample ``x`` gets the
    same jitter whether it is processed first or thousandth.
    """
    key = hashlib.sha256(("|".join(str(p) for p in parts)).encode()).digest()[:8]
    return np.random.default_rng((seed << 24) ^ int.from_bytes(key, "big"))


# --------------------------------------------------------------------------
# The answer key
# --------------------------------------------------------------------------


@dataclass
class TruthRow:
    sample_id: str
    is_poisoned: int
    attack_class: str
    true_label: str | None
    stored_label: str | None
    contributor_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "is_poisoned": int(self.is_poisoned),
            "attack_class": self.attack_class,
            "true_label": self.true_label,
            "stored_label": self.stored_label,
            "contributor_id": self.contributor_id,
        }


@dataclass
class Summary:
    out_dir: Path
    truth_dir: Path
    seed: int
    n_samples: int
    n_poisoned: int
    per_attack: dict[str, int]
    per_contributor: dict[str, dict[str, int]]
    bad_contributor: str | None
    model_attacks: list[dict[str, Any]] = field(default_factory=list)
    receipt_attacks: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Poisoned dataset written to {self.out_dir}",
            f"Answer key written to      {self.truth_dir / GROUND_TRUTH_NAME}",
            f"Seed                       {self.seed}",
            "",
            f"{self.n_samples} images, {self.n_poisoned} of them tampered with "
            f"({100 * self.n_poisoned / max(1, self.n_samples):.1f}%).",
        ]
        for name, n in sorted(self.per_attack.items()):
            lines.append(f"  {name:<24} {n}")
        if self.per_contributor:
            lines.append("")
            lines.append("Per contributor:")
            for cid, d in sorted(self.per_contributor.items()):
                marker = "  <- designated bad actor" if cid == self.bad_contributor else ""
                lines.append(
                    f"  {cid:<8} {d['n']:>5} images, {d['n_poisoned']:>5} poisoned "
                    f"({100 * d['n_poisoned'] / max(1, d['n']):.1f}%){marker}"
                )
        for m in self.model_attacks:
            lines.append(f"\nModel attack: {m['attack_class']} -> {m['path']}")
        for r in self.receipt_attacks:
            lines.append(f"Receipt attack: {r['attack_class']} on record #{r.get('seq')}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Contributor assignment
# --------------------------------------------------------------------------


def assign_contributors(
    sample_ids: Sequence[str],
    poisoned: set[str],
    *,
    n_contributors: int,
    concentration: float,
    seed: int,
    bad_contributor: str | None = None,
) -> tuple[dict[str, str], str | None]:
    """Spread samples over contributors, concentrating the poison.

    ``concentration=0.8`` means 80% of the poisoned samples are handed to one
    designated bad actor. That is what makes source-level blame *testable*:
    without it every contributor looks equally guilty and the contributor
    detector has nothing to find.
    """
    if n_contributors < 1:
        raise ValueError("need at least one contributor")
    names = [f"C{i}" for i in range(n_contributors)]
    bad = bad_contributor or (names[-1] if n_contributors > 1 else names[0])
    if bad not in names:
        names.append(bad)

    rng = np.random.default_rng(seed ^ 0xC0FFEE)
    poisoned_ids = sorted(poisoned)
    clean_ids = sorted(set(sample_ids) - poisoned)
    rng.shuffle(poisoned_ids)
    rng.shuffle(clean_ids)

    mapping: dict[str, str] = {}
    n_to_bad = int(round(concentration * len(poisoned_ids)))
    others = [n for n in names if n != bad] or [bad]
    for i, sid in enumerate(poisoned_ids):
        mapping[sid] = bad if i < n_to_bad else others[i % len(others)]

    # Clean samples spread evenly, so the bad actor is not identifiable
    # simply by having fewer images than everybody else.
    for i, sid in enumerate(clean_ids):
        mapping[sid] = names[i % len(names)]
    return mapping, bad


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------


class PoisonRun:
    """Builds a poisoned copy of a dataset and its ground truth."""

    def __init__(self, dataset_root: str | Path, out_dir: str | Path, seed: int = 0):
        self.seed = int(seed)
        self.src = Path(dataset_root)
        self.out = Path(out_dir)
        self.dataset = load_dataset(self.src)
        # sample_id -> current state
        self.rows: dict[str, TruthRow] = {}
        self.files: dict[str, Path] = {}
        self._model_attacks: list[dict[str, Any]] = []
        self._receipt_attacks: list[dict[str, Any]] = []
        self._prepare()

    def _prepare(self) -> None:
        """Copy the clean dataset into the output folder, class by class."""
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        for s in sorted(self.dataset, key=lambda s: s.sample_id):
            label = s.label or "unlabelled"
            dest = self.out / label / Path(s.image_path).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s.image_path, dest)
            sid = str(dest.relative_to(self.out))
            self.files[sid] = dest
            self.rows[sid] = TruthRow(
                sample_id=sid,
                is_poisoned=0,
                attack_class="clean",
                true_label=label,
                stored_label=label,
                contributor_id=None,
            )

    # -- helpers ---------------------------------------------------------

    def _candidates(self, label: str | None = None) -> list[str]:
        ids = [
            sid
            for sid, row in self.rows.items()
            if row.is_poisoned == 0 and (label is None or row.stored_label == label)
        ]
        return sorted(ids)

    def _pick(self, n: int, tag: str, label: str | None = None) -> list[str]:
        pool = self._candidates(label)
        n = min(n, len(pool))
        rng = _sub_rng(self.seed, "pick", tag, label)
        idx = rng.choice(len(pool), size=n, replace=False) if n else np.array([], int)
        return sorted(pool[i] for i in np.atleast_1d(idx).tolist())

    def _mark(self, sid: str, attack_class: str, *, stored_label: str | None = None) -> None:
        row = self.rows[sid]
        row.is_poisoned = 1
        row.attack_class = attack_class
        if stored_label is not None:
            row.stored_label = stored_label

    def _relabel_file(self, sid: str, new_label: str) -> str:
        """Move an image into another class folder. In ImageFolder layout the
        folder *is* the label, so this is what a label flip looks like on disk."""
        old = self.files[sid]
        dest = self.out / new_label / old.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(dest))
        new_sid = str(dest.relative_to(self.out))
        row = self.rows.pop(sid)
        del self.files[sid]
        row.sample_id = new_sid
        row.stored_label = new_label
        self.rows[new_sid] = row
        self.files[new_sid] = dest
        return new_sid

    # -- data attacks ----------------------------------------------------

    def badnets_patch(
        self,
        *,
        rate: float = 0.05,
        size: int = 6,
        colour: Sequence[int] = (255, 255, 255),
        position: str = "bottom_right",
        target_class: str | None = None,
    ) -> int:
        victims = self._pick(int(round(rate * len(self.rows))), "badnets")
        for sid in victims:
            path = self.files[sid]
            with Image.open(path) as im:
                out = image_ops.badnets_patch(
                    im,
                    size=size,
                    colour=tuple(colour),
                    position=position,
                    rng=_sub_rng(self.seed, "badnets", sid),
                )
            out.save(path)
            self._mark(sid, "badnets_patch")
            if target_class:
                self._relabel_file(sid, target_class)
        return len(victims)

    def blended_trigger(
        self, *, rate: float = 0.05, pattern: str = "checkerboard", alpha: float = 0.08,
        target_class: str | None = None,
    ) -> int:
        victims = self._pick(int(round(rate * len(self.rows))), "blended")
        for sid in victims:
            path = self.files[sid]
            with Image.open(path) as im:
                out = image_ops.blended_trigger(
                    im, pattern=pattern, alpha=alpha,
                    seed=int(_sub_rng(self.seed, "blend", sid).integers(0, 2**31)),
                )
            out.save(path)
            self._mark(sid, "blended_trigger")
            if target_class:
                self._relabel_file(sid, target_class)
        return len(victims)

    def label_flip(
        self, *, rate: float = 0.05, from_class: str | None = None, to_class: str | None = None
    ) -> int:
        """The image is untouched. Only the label is wrong — which is exactly
        why pixel-level checks cannot see it."""
        labels = sorted({r.stored_label for r in self.rows.values() if r.stored_label})
        if len(labels) < 2:
            return 0
        pool_size = len(self._candidates(from_class))
        victims = self._pick(int(round(rate * pool_size)), "flip", from_class)
        n = 0
        for sid in victims:
            row = self.rows[sid]
            options = [l for l in labels if l != row.stored_label]
            target = to_class or str(
                _sub_rng(self.seed, "flipto", sid).choice(options)
            )
            if target == row.stored_label:
                continue
            self._mark(sid, "label_flip")
            self._relabel_file(sid, target)
            n += 1
        return n

    def systematic_mislabel(
        self, *, contributor_id: str, mapping: dict[str, str] | None = None, rate: float = 1.0
    ) -> int:
        """One contributor is always wrong in the same way — a policy error or
        a deliberate campaign, not random noise. Requires contributors to have
        been assigned first."""
        victims = sorted(
            sid
            for sid, r in self.rows.items()
            if r.contributor_id == contributor_id
            and r.is_poisoned == 0
            and (mapping is None or r.stored_label in mapping)
        )
        rng = _sub_rng(self.seed, "sysmis", contributor_id)
        keep = [sid for sid in victims if rng.random() < rate]
        labels = sorted({r.stored_label for r in self.rows.values() if r.stored_label})
        n = 0
        for sid in keep:
            row = self.rows[sid]
            if mapping and row.stored_label in mapping:
                target = mapping[row.stored_label]
            else:
                options = [l for l in labels if l != row.stored_label]
                target = str(_sub_rng(self.seed, "sysmisto", sid).choice(options))
            self._mark(sid, "systematic_mislabel")
            self._relabel_file(sid, target)
            n += 1
        return n

    def near_duplicate_flood(self, *, n_seeds: int = 5, n_copies: int = 10) -> int:
        seeds = self._pick(n_seeds, "dupseeds")
        made = 0
        for sid in seeds:
            src = self.files[sid]
            label = self.rows[sid].stored_label or "unlabelled"
            rng = _sub_rng(self.seed, "dup", sid)
            with Image.open(src) as im:
                base = im.convert("RGB")
                for k in range(n_copies):
                    variant = image_ops.duplicate_variant(base, index=k, rng=rng)
                    dest = src.with_name(f"{src.stem}_dup{k:03d}{src.suffix}")
                    variant.save(dest)
                    new_sid = str(dest.relative_to(self.out))
                    self.files[new_sid] = dest
                    self.rows[new_sid] = TruthRow(
                        sample_id=new_sid,
                        is_poisoned=1,
                        attack_class="near_duplicate_flood",
                        true_label=label,
                        stored_label=label,
                        contributor_id=None,
                    )
                    made += 1
        return made

    def ood_insertion(self, *, other_dir: str | Path, count: int = 20) -> int:
        pool = sorted(
            p for p in Path(other_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not pool:
            raise FileNotFoundError(f"no images found in {other_dir}")
        labels = sorted({r.stored_label for r in self.rows.values() if r.stored_label})
        rng = _sub_rng(self.seed, "ood")
        n = 0
        for i in range(min(count, len(pool))):
            src = pool[i]
            label = str(rng.choice(labels)) if labels else "unlabelled"
            dest = self.out / label / f"foreign_{i:04d}{src.suffix}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            sid = str(dest.relative_to(self.out))
            self.files[sid] = dest
            self.rows[sid] = TruthRow(
                sample_id=sid,
                is_poisoned=1,
                attack_class="ood_insertion",
                true_label="foreign",
                stored_label=label,
                contributor_id=None,
            )
            n += 1
        return n

    def distribution_shift(
        self, *, rate: float = 0.3, kind: str = "fog", severity: float = 0.6
    ) -> int:
        """Apply a physical corruption. This is *drift*, not manipulation, and
        the shift detector is expected to say so — a system that calls fog an
        attack is useless in the field."""
        victims = self._pick(int(round(rate * len(self.rows))), f"shift_{kind}")
        for sid in victims:
            path = self.files[sid]
            with Image.open(path) as im:
                out = image_ops.corrupt(
                    im, kind, severity,
                    seed=int(_sub_rng(self.seed, "shift", sid).integers(0, 2**31)),
                )
            out.save(path)
            row = self.rows[sid]
            row.attack_class = "distribution_shift"
            # deliberately NOT is_poisoned=1: corrupted weather is benign
        return len(victims)

    # -- contributors ----------------------------------------------------

    def assign(self, *, n_contributors: int = 5, concentration: float = 0.8,
               bad_contributor: str | None = None) -> str | None:
        poisoned = {sid for sid, r in self.rows.items() if r.is_poisoned}
        mapping, bad = assign_contributors(
            sorted(self.rows),
            poisoned,
            n_contributors=n_contributors,
            concentration=concentration,
            seed=self.seed,
            bad_contributor=bad_contributor,
        )
        for sid, cid in mapping.items():
            self.rows[sid].contributor_id = cid
        self.bad_contributor = bad
        return bad

    # -- writing out -----------------------------------------------------

    def write(self, truth_dir: str | Path, attack_config: dict[str, Any]) -> Summary:
        import json

        truth = Path(truth_dir)
        truth.mkdir(parents=True, exist_ok=True)

        rows = [self.rows[sid].to_dict() for sid in sorted(self.rows)]
        # Deliberately free of absolute paths: the answer key must be
        # byte-identical for a given seed on any machine, and where the files
        # happen to live is not part of the answer. Paths go in the manifest.
        payload = {
            "seed": self.seed,
            "attack_config": attack_config,
            "samples": rows,
            "model_attacks": [
                {k: v for k, v in m.items() if k != "path"} for m in self._model_attacks
            ],
            "receipt_attacks": [
                {k: v for k, v in r.items() if k != "path"} for r in self._receipt_attacks
            ],
        }
        (truth / GROUND_TRUTH_NAME).write_bytes(canonical_json(payload))
        (truth / "run_manifest.json").write_text(
            json.dumps(
                {
                    "seed": self.seed,
                    "source_dataset": str(self.src),
                    "poisoned_dataset": str(self.out),
                    "model_attack_paths": [m.get("path") for m in self._model_attacks],
                    "receipt_attack_paths": [r.get("path") for r in self._receipt_attacks],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # contributors.json travels with the data, as it would in the field
        (self.out / "contributors.json").write_text(
            json.dumps(
                {sid: self.rows[sid].contributor_id for sid in sorted(self.rows)
                 if self.rows[sid].contributor_id},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        per_attack = Counter(r["attack_class"] for r in rows if r["is_poisoned"])
        per_contributor: dict[str, dict[str, int]] = {}
        for r in rows:
            cid = r["contributor_id"]
            if cid is None:
                continue
            d = per_contributor.setdefault(cid, {"n": 0, "n_poisoned": 0})
            d["n"] += 1
            d["n_poisoned"] += r["is_poisoned"]

        return Summary(
            out_dir=self.out,
            truth_dir=truth,
            seed=self.seed,
            n_samples=len(rows),
            n_poisoned=sum(r["is_poisoned"] for r in rows),
            per_attack=dict(per_attack),
            per_contributor=per_contributor,
            bad_contributor=getattr(self, "bad_contributor", None),
            model_attacks=self._model_attacks,
            receipt_attacks=self._receipt_attacks,
        )


# --------------------------------------------------------------------------
# YAML driver
# --------------------------------------------------------------------------

DATA_ATTACKS = {
    "badnets_patch",
    "blended_trigger",
    "label_flip",
    "systematic_mislabel",
    "near_duplicate_flood",
    "ood_insertion",
    "distribution_shift",
}


def run_config(
    config_path: str | Path,
    dataset_root: str | Path,
    out_dir: str | Path,
    truth_dir: str | Path,
    *,
    seed: int | None = None,
    model_path: str | Path | None = None,
    receipts_path: str | Path | None = None,
    key_path: str | Path | None = None,
) -> Summary:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    seed = int(seed if seed is not None else cfg.get("seed", 0))

    run = PoisonRun(dataset_root, out_dir, seed=seed)

    contributors_cfg = cfg.get("contributors", {}) or {}
    n_contributors = int(contributors_cfg.get("n", 5))
    concentration = float(contributors_cfg.get("poison_concentration", 0.8))
    bad_contributor = contributors_cfg.get("bad_contributor")

    specs = cfg.get("attacks", []) or []

    # Contributor-targeted attacks need the assignment to exist first, so we
    # assign once up front and once again afterwards to catch samples created
    # by the attacks themselves (duplicates, inserted foreign images).
    if any(s.get("type") == "systematic_mislabel" for s in specs):
        run.assign(n_contributors=n_contributors, concentration=concentration,
                   bad_contributor=bad_contributor)

    for spec in specs:
        kind = spec.get("type")
        params = {k: v for k, v in spec.items() if k != "type"}
        if kind not in DATA_ATTACKS:
            raise ValueError(f"unknown attack type {kind!r} in {config_path}")
        if kind == "systematic_mislabel" and "contributor_id" not in params:
            params["contributor_id"] = bad_contributor or f"C{n_contributors - 1}"
        getattr(run, kind)(**params)

    run.assign(n_contributors=n_contributors, concentration=concentration,
               bad_contributor=bad_contributor)

    # -- model and receipt attacks --------------------------------------
    for spec in cfg.get("model_attacks", []) or []:
        from cvassure.attack import model_attacks

        run._model_attacks.append(
            model_attacks.apply(spec, model_path=model_path, out_dir=Path(out_dir) / "models",
                                seed=seed)
        )
    for spec in cfg.get("receipt_attacks", []) or []:
        from cvassure.attack import receipt_attacks

        run._receipt_attacks.append(
            receipt_attacks.apply(
                spec, receipts_path=receipts_path, out_dir=Path(out_dir) / "logs",
                seed=seed, key_path=key_path,
            )
        )

    return run.write(truth_dir, {"config_file": Path(config_path).name, **cfg})


def load_ground_truth(truth_dir: str | Path) -> dict[str, Any]:
    import json

    p = Path(truth_dir)
    if p.is_dir():
        p = p / GROUND_TRUTH_NAME
    return json.loads(p.read_text(encoding="utf-8"))
