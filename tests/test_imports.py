"""Regression test for lumina-st #100.

Runtime modules import ``pytorch_lightning``, so it MUST be declared as a
runtime dependency in ``pyproject.toml`` (not just a dev extra). This test
verifies the dependency is declared correctly so that ``pip install lumina-st``
produces an importable package.
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    import tomli as tomllib  # type: ignore[no-redef]


PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _runtime_dependencies() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    return list(data["project"]["dependencies"])


def test_clean_env_import_lumina_st() -> None:
    """``pytorch_lightning`` must appear in runtime ``[project] dependencies``.

    Runtime modules (``core/lumina_imputer.py``, ``modules/lumina_flow_module.py``)
    do ``import pytorch_lightning as pl`` at module load time. A user who runs
    ``pip install lumina-st`` in a clean venv (no ``[dev]`` extra) and then
    ``import lumina_st`` would otherwise hit ``ImportError: No module named
    'pytorch_lightning'``.
    """

    deps = _runtime_dependencies()
    assert any(
        d.split(";")[0].strip().lower().startswith("pytorch_lightning")
        or d.split(";")[0].strip().lower().startswith("pytorch-lightning")
        for d in deps
    ), (
        "pytorch_lightning is imported by runtime modules but is not declared "
        "in [project] dependencies of pyproject.toml. A clean-venv "
        "`pip install lumina-st && python -c 'import lumina_st'` will raise "
        f"ImportError. Current runtime deps: {deps}"
    )
