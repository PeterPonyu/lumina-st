"""Regression tests for doc/code drift (issues #115, #120, #248, #110).

Pins two properties so docs cannot silently drift from the shipped code again:

1. User-facing docs name real symbols — ``LatentVelocityNet`` (which never
   existed; the real class is ``LuminaTransformer``) must not reappear.
2. Every in-repo doc path referenced from ``README.md`` and from ``src/`` module
   docstrings resolves on disk (no dangling ``docs/...md`` /
   ``benchmark_contracts/...json`` links).
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_README = _REPO_ROOT / "README.md"
_SRC = _REPO_ROOT / "src"

# Repo-relative path references we expect to resolve. External/optional paths
# (e.g. an out-of-repo baseline audit clone) are deliberately excluded.
_DOC_REF_RE = re.compile(r"`?((?:docs|benchmark_contracts)/[\w./-]+\.(?:md|json))`?")


def _src_py_files() -> list[Path]:
    return [p for p in _SRC.rglob("*.py") if "__pycache__" not in p.parts]


def test_readme_has_no_phantom_class_name():
    assert "LatentVelocityNet" not in _README.read_text(encoding="utf-8")


def test_no_src_docstring_names_phantom_class():
    offenders = [
        str(f.relative_to(_REPO_ROOT))
        for f in _src_py_files()
        if "LatentVelocityNet" in f.read_text(encoding="utf-8")
    ]
    assert not offenders, f"phantom class name in: {offenders}"


def test_readme_claim_ledger_link_resolves():
    text = _README.read_text(encoding="utf-8")
    assert "docs/CLAIM_LEDGER.md" in text
    assert (_REPO_ROOT / "docs" / "CLAIM_LEDGER.md").exists()


def test_referenced_repo_doc_paths_resolve():
    offenders: list[str] = []
    sources = [_README, *_src_py_files()]
    for f in sources:
        text = f.read_text(encoding="utf-8")
        for match in _DOC_REF_RE.finditer(text):
            rel = match.group(1)
            if not (_REPO_ROOT / rel).exists():
                offenders.append(f"{f.relative_to(_REPO_ROOT)} -> {rel}")
    assert not offenders, "dangling in-repo doc references:\n" + "\n".join(offenders)
