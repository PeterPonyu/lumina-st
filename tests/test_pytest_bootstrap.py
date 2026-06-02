"""Regression test: the suite collects without manual PYTHONPATH / editable install.

Locks in the src-layout bootstrap (root ``conftest.py`` + the ``pythonpath``
ini setting) so the suite can never silently regress to "only runs if you
remembered to ``pip install -e .``" (issues #117, #238).

The test spawns ``python -m pytest --collect-only`` in a subprocess from the
repo root with ``PYTHONPATH`` stripped from the environment and asserts a clean
collection (no ``ModuleNotFoundError: No module named 'lumina_st'``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_conftest_bootstraps_src_layout():
    """The bootstrap mechanism must be present and point at ``src/``."""
    conftest = REPO_ROOT / "conftest.py"
    assert conftest.exists(), "root conftest.py is the src-layout bootstrap (issue #238)"
    assert "src" in conftest.read_text(encoding="utf-8")


def test_collection_without_pythonpath_or_editable_install():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_imports.py",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    assert "No module named 'lumina_st'" not in combined, combined
    assert result.returncode == 0, combined
