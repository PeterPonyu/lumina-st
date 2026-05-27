"""Regression tests for the Round 13 Sankey composer skeleton."""

from __future__ import annotations

import importlib.util
from types import ModuleType
from pathlib import Path

import pytest


def _load_compose_sankey() -> ModuleType:
    manuscript_dir = Path(__file__).resolve().parents[2] / "manuscript"
    spec = importlib.util.spec_from_file_location(
        "compose_sankey",
        manuscript_dir / "compose_sankey.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # compose_sankey imports sibling compose_palette.py.
    import sys

    sys.path.insert(0, str(manuscript_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(manuscript_dir))
    return module


def test_render_sankey_writes_nonempty_png(tmp_path: Path) -> None:
    module = _load_compose_sankey()
    contingency = {
        "truth_T": {"pred_0": 8, "pred_1": 2},
        "truth_B": {"pred_1": 7, "pred_2": 3},
    }

    out = module.render_sankey(contingency, tmp_path / "sankey.png")

    assert out == tmp_path / "sankey.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_synthetic_contingency_is_deterministic() -> None:
    module = _load_compose_sankey()

    assert module._synthetic_contingency(seed=13) == module._synthetic_contingency(seed=13)
    assert module._synthetic_contingency(seed=13) != module._synthetic_contingency(seed=14)


@pytest.mark.parametrize(
    "contingency",
    [
        {},
        {"truth": {}},
        {"truth": {"pred": 0}},
        {"truth": {"pred": -1}},
    ],
)
def test_render_sankey_rejects_nonpositive_or_invalid_tables(
    tmp_path: Path,
    contingency: dict[str, dict[str, int]],
) -> None:
    module = _load_compose_sankey()

    with pytest.raises(ValueError):
        module.render_sankey(contingency, tmp_path / "bad.png")


def test_render_sankey_handles_many_small_categories(tmp_path: Path) -> None:
    module = _load_compose_sankey()
    contingency = {f"truth_{i}": {f"pred_{j}": 1 for j in range(4)} for i in range(8)}

    out = module.render_sankey(contingency, tmp_path / "many.png")

    assert out.exists()
    assert out.stat().st_size > 0
