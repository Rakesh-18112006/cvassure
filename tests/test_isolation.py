"""The rule that makes every number in the results section believable.

The detectors cannot see the answer key, and this test proves it by reading
the source. If somebody adds a convenience import under deadline pressure, the
build fails rather than the credibility.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "cvassure"
DETECT = PACKAGE / "detect"
FORBIDDEN_PREFIXES = ("cvassure.attack", "cvassure.datasets.toy_model")


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


def test_detect_package_exists():
    assert DETECT.is_dir(), "cvassure/detect/ must exist"


@pytest.mark.parametrize("path", _python_files(DETECT), ids=lambda p: p.name)
def test_no_detector_imports_the_attack_code(path):
    offenders = sorted(
        m
        for m in _imported_modules(path)
        if any(m == p or m.startswith(p + ".") for p in FORBIDDEN_PREFIXES)
    )
    assert not offenders, (
        f"{path.relative_to(PACKAGE.parent)} imports {offenders}. Detectors must "
        f"never be able to see the answer key — that is what makes the measured "
        f"numbers mean anything."
    )


def test_grep_finds_no_attack_reference_anywhere_in_detect():
    """Belt and braces: catch dynamic imports and string module names too."""
    hits = []
    for path in _python_files(DETECT):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.split("#", 1)[0]
            if "cvassure.attack" in stripped or "ground_truth" in stripped:
                hits.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not hits, "detect/ must not reference the attack code or the answer key:\n" + "\n".join(hits)


def test_the_report_never_reads_the_answer_key():
    """The assurance report describes what the detectors found, not what we
    know. If report/ could read ground truth it could quietly launder the
    answer key into the output shown to a judge."""
    readers = []
    for folder in (DETECT, PACKAGE / "report"):
        if not folder.is_dir():
            continue
        for path in _python_files(folder):
            if "ground_truth" in path.read_text(encoding="utf-8"):
                readers.append(str(path.relative_to(PACKAGE.parent)))
    assert not readers, f"these must not read the answer key: {readers}"
