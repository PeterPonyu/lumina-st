"""Independence guard for issue #121 — no baseline brand leakage in code.

The repo treats the "stPainter" baseline as prior art / audit boundary only
(see README "Prior Art and Audit Boundary"). Legitimate prior-art *citations*
live in docs; the package source and runnable scripts must not bake in the
baseline brand name or a specific baseline on-disk directory layout.

These tests pin:
  1. No ``stpainter`` token (case-insensitive) anywhere under ``src/``.
  2. No hardcoded absolute / machine-specific paths under ``src/``.
  3. No hardcoded ``data/baselines/stpainter/...`` literals under ``scripts/``
     (baseline locations must come from args / config / env var).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_SCRIPTS = _REPO_ROOT / "scripts"

# Absolute / machine-specific path roots that must never be hardcoded in src.
_ABS_PATH_RE = re.compile(r"(/home/|/Users/|/mnt/|/scratch/|/work/|[A-Za-z]:\\\\)")
_STPAINTER_RE = re.compile(r"stpainter", re.IGNORECASE)
_BASELINE_PATH_RE = re.compile(r"data/baselines/stpainter")


def _py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


# ``results_contract.py`` is vendored byte-identical to the parent orchestration
# repo's canonical contract (pinned by tests/test_contract_schema.py via SHA-256).
# Its docstring carries a parent-authored usage example that references the
# baseline on-disk layout. We CANNOT edit it without breaking byte-identity, so
# the brand-leakage grep excludes this single byte-locked file by name.
_BYTE_LOCKED_SRC = {"results_contract.py"}


def test_src_has_no_stpainter_token():
    offenders = []
    for f in _py_files(_SRC):
        if f.name in _BYTE_LOCKED_SRC:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if _STPAINTER_RE.search(line):
                offenders.append(f"{f.relative_to(_REPO_ROOT)}:{i}: {line.strip()}")
    assert not offenders, "baseline brand leakage in src:\n" + "\n".join(offenders)


def test_src_has_no_hardcoded_absolute_paths():
    offenders = []
    for f in _py_files(_SRC):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if _ABS_PATH_RE.search(line):
                offenders.append(f"{f.relative_to(_REPO_ROOT)}:{i}: {line.strip()}")
    assert not offenders, "hardcoded absolute paths in src:\n" + "\n".join(offenders)


@pytest.mark.skipif(not _SCRIPTS.exists(), reason="no scripts/ dir")
def test_scripts_have_no_hardcoded_baseline_paths():
    offenders = []
    for f in _py_files(_SCRIPTS):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if _BASELINE_PATH_RE.search(line):
                offenders.append(f"{f.relative_to(_REPO_ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "hardcoded data/baselines/stpainter literals in scripts "
        "(use LUMINA_BASELINE_ROOT / args / config):\n" + "\n".join(offenders)
    )
