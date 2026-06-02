"""Pytest bootstrap for the ``src/`` layout.

Guarantees ``import lumina_st`` resolves from ``src/`` when the suite runs from
a fresh checkout, with no editable install and no manually exported
``PYTHONPATH`` (issue #238). Without this, ``python -m pytest`` fails during
collection with ``ModuleNotFoundError: No module named 'lumina_st'``.

This complements the ``pythonpath = ["src"]`` setting in ``pyproject.toml``:
keeping both means the suite still collects if either mechanism is removed or a
runner ignores the pytest ini ``pythonpath`` table. The insert is idempotent
and a no-op once the package is importable by any other means.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir():
    _src_str = str(_SRC)
    if _src_str not in sys.path:
        sys.path.insert(0, _src_str)
