"""Tiny table type: CSV and Markdown out, no pandas dependency."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class Table:
    name: str
    title: str
    columns: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, **row: Any) -> None:
        unknown = set(row) - set(self.columns)
        if unknown:
            raise KeyError(f"{self.name}: unknown column(s) {sorted(unknown)}")
        self.rows.append({c: row.get(c, "") for c in self.columns})

    def __len__(self) -> int:
        return len(self.rows)

    # -- output ----------------------------------------------------------

    def to_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.columns)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: _plain(v) for k, v in r.items()})
        return p

    def to_markdown(self) -> str:
        head = "| " + " | ".join(self.columns) + " |"
        rule = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = [
            "| " + " | ".join(_plain(r[c]) for c in self.columns) + " |" for r in self.rows
        ]
        parts = [f"### {self.title}", "", head, rule, *body]
        if self.notes:
            parts.append("")
            parts.extend(f"*{n}*" for n in self.notes)
        return "\n".join(parts)

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        md = out / f"{self.name}.md"
        md.write_text(self.to_markdown() + "\n", encoding="utf-8")
        return {"csv": self.to_csv(out / f"{self.name}.csv"), "markdown": md}


def _plain(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_all(tables: Sequence[Table], out_dir: str | Path) -> dict[str, dict[str, Path]]:
    return {t.name: t.write(out_dir) for t in tables}
