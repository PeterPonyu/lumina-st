"""Root conftest.py — src-layout import shim for bare-checkout test runs.

Makes ``import lumina_st`` work when pytest is invoked from a fresh clone
with no editable install (``pip install -e .``).  The ``pyproject.toml``
``[tool.pytest.ini_options] pythonpath = ["src"]`` setting does the same job
when pytest ≥ 7.0 is used; this file is the belt-*and*-suspenders fallback
that also covers older pytest and direct ``python -m pytest`` invocations
where the ini option is not honoured.

Background: issue #117 / #102 — tests were either failing to collect (no
PYTHONPATH, no editable install) or carrying per-file ``sys.path.insert``
hacks that broke IDE-level test discovery and caused duplicate-import
confusion.  This single conftest replaces all of those per-file hacks.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repository root is the parent of the ``tests/`` directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_REPO_ROOT / "src")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
