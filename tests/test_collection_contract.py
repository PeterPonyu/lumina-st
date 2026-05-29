"""Collection contract: the suite must be importable on a fresh checkout.

Issue #96 — lumina-st uses a src-layout (`src/lumina_st`) with no
`conftest.py` and no `pythonpath` setting, so `import lumina_st` failed
during collection on a fresh checkout (no editable install present).
Pytest's `pythonpath` ini option fixes this by prepending `src/` to
`sys.path` at startup.

This test asserts the configuration contract directly so it does not
depend on whether the package happens to be pip-installed in the current
environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    import tomli as tomllib  # type: ignore[no-redef]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_pyproject_sets_pythonpath_src() -> None:
    """pyproject must put `src` on pytest's pythonpath (issue #96)."""
    pyproject = _repo_root() / "pyproject.toml"
    with pyproject.open("rb") as fh:
        cfg = tomllib.load(fh)

    ini = cfg["tool"]["pytest"]["ini_options"]
    pythonpath = ini.get("pythonpath", [])
    assert "src" in pythonpath, (
        "pyproject [tool.pytest.ini_options] must set pythonpath=['src'] so "
        "the src-layout package collects on a fresh checkout (issue #96)"
    )


def test_fresh_checkout_collects() -> None:
    """`lumina_st` imports, and `src/` is reachable on sys.path."""
    import lumina_st  # noqa: F401  -- import must succeed during collection

    src_dir = _repo_root() / "src"
    on_path = any(Path(p).resolve() == src_dir for p in sys.path if p)
    assert on_path, "src/ should be on sys.path via pytest pythonpath (issue #96)"
