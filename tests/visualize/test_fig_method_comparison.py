"""Smoke test for the multi-method gene-recovery comparison figure script.

The script lives under ``scripts/visualize`` (not an installed package), so it
is loaded by file path via importlib. The test feeds it a small synthetic
results bundle that mirrors the runner's contract and asserts a valid, non-empty
PNG is produced with the expected number of metric axes and method bars.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "visualize"
    / "fig_method_comparison.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fig_method_comparison", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's ``@dataclass`` types can resolve
    # their (PEP 563 / string) annotations via ``sys.modules`` at class-build time.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_bundle() -> dict[str, Any]:
    """A minimal aggregated bundle with two panels and three methods."""

    def metrics(pcc: float, spearman: float, rmse: float, jaccard: float) -> dict[str, Any]:
        return {
            "mean_pearson": pcc,
            "mean_spearman": spearman,
            "mean_rmse": rmse,
            "zero_pattern_jaccard": jaccard,
            "per_gene_pearson": {"CD4": pcc - 0.05, "CD8A": pcc + 0.05},
            "per_gene_spearman": {"CD4": spearman - 0.04, "CD8A": spearman + 0.04},
            "per_gene_rmse": {"CD4": rmse + 0.1, "CD8A": rmse - 0.1},
            "n_genes_scored": 2,
        }

    def ok(m: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "runtime_s": 0.1, "metrics": m, "provenance": {}}

    return {
        "schema_version": "1",
        "panels": {
            "synthetic-coad/tme-immune-stromal": {
                "lumina": ok(metrics(0.82, 0.79, 0.41, 0.88)),
                "mean": ok(metrics(0.55, 0.50, 0.73, 0.70)),
                "knn": ok(metrics(0.61, 0.58, 0.66, 0.74)),
                # An unavailable method must be skipped, not plotted.
                "spaim": {"status": "unavailable:torch missing", "metrics": {}},
            },
            "synthetic-coad/vascular-3d": {
                "lumina": ok(metrics(0.78, 0.75, 0.45, 0.85)),
                "mean": ok(metrics(0.52, 0.48, 0.77, 0.68)),
                "knn": ok(metrics(0.59, 0.55, 0.69, 0.72)),
            },
        },
    }


def _is_valid_png(path: Path) -> bool:
    """True if the file starts with the PNG magic signature and has an IHDR."""
    with path.open("rb") as fh:
        header = fh.read(8)
    return header == b"\x89PNG\r\n\x1a\n"


def test_extract_method_metrics_skips_unavailable() -> None:
    mod = _load_script()
    bundle = _synthetic_bundle()
    extracted = mod.extract_method_metrics(bundle, panel="synthetic-coad/tme-immune-stromal")
    assert set(extracted) == {"lumina", "mean", "knn"}  # spaim (unavailable) dropped
    assert "mean_pearson" in extracted["lumina"]
    # Per-gene dicts present -> a non-zero standard-error bar is computed.
    assert extracted["lumina"]["mean_pearson"]["err"] > 0.0


def test_order_methods_puts_focal_first() -> None:
    mod = _load_script()
    ordered = mod.order_methods(["knn", "mean", "lumina", "zzz_unknown"])
    assert ordered[0] == "lumina"
    assert ordered[-1] == "zzz_unknown"  # unknown method appended last


def test_render_produces_valid_nonempty_png(tmp_path: Path) -> None:
    mod = _load_script()
    bundle_path = tmp_path / "results.json"
    bundle_path.write_text(json.dumps(_synthetic_bundle()))
    out_path = tmp_path / "fig.png"

    returned = mod.render_from_bundle(
        results_path=bundle_path,
        output_path=out_path,
        panel=None,  # exercise the all-panel aggregation path
    )

    assert returned == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # a real figure, not an empty stub
    assert _is_valid_png(out_path)

    # PNG IHDR carries width/height as big-endian uint32 at byte offset 16.
    with out_path.open("rb") as fh:
        fh.seek(16)
        width, height = struct.unpack(">II", fh.read(8))
    assert width > 0 and height > 0


def test_render_specific_panel(tmp_path: Path) -> None:
    mod = _load_script()
    bundle_path = tmp_path / "results.json"
    bundle_path.write_text(json.dumps(_synthetic_bundle()))
    out_path = tmp_path / "panel.png"

    mod.render_from_bundle(
        results_path=bundle_path,
        output_path=out_path,
        panel="synthetic-coad/tme-immune-stromal",
        title="custom title",
    )
    assert _is_valid_png(out_path)


def test_missing_panels_key_raises(tmp_path: Path) -> None:
    mod = _load_script()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "1"}))
    with pytest.raises(ValueError):
        mod.load_results(bad)


def test_unknown_panel_raises(tmp_path: Path) -> None:
    mod = _load_script()
    bundle_path = tmp_path / "results.json"
    bundle_path.write_text(json.dumps(_synthetic_bundle()))
    with pytest.raises(KeyError):
        mod.render_from_bundle(
            results_path=bundle_path,
            output_path=tmp_path / "x.png",
            panel="no/such-panel",
        )
