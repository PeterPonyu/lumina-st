"""Smoke + schema tests for the figure/table generation framework.

The generators live under ``scripts/visualize`` (not an installed package), so
each is loaded by file path via importlib. Every test feeds a TINY synthetic
benchmark bundle that mirrors ``lumina_st.benchmarks.runner.aggregate_results``
and asserts the expected artifact (PNG and/or CSV+markdown) is written without
error, that the table content has the expected columns/rows, and that the
schema guard raises a clear error on malformed input. Matplotlib is forced to
the headless ``Agg`` backend by ``_figbase`` so this stays headless + fast.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_VIZ_DIR = Path(__file__).resolve().parents[2] / "scripts" / "visualize"
# Generators do ``from _figbase import ...``; make it importable by file path.
sys.path.insert(0, str(_VIZ_DIR))


def _load(name: str) -> ModuleType:
    # Reuse an already-loaded module so the generators and this test share one
    # ``_figbase`` instance (and therefore one ``SchemaError`` class).
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _VIZ_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _is_png(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(8) == b"\x89PNG\r\n\x1a\n"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


# ---------------------------------------------------------------------------
# Synthetic bundles (mirroring the runner's panels schema).
# ---------------------------------------------------------------------------


def _ok(metrics: dict[str, Any], runtime_s: float = 0.1) -> dict[str, Any]:
    return {"status": "ok", "runtime_s": runtime_s, "metrics": metrics, "provenance": {}}


def _recovery_metrics(pcc: float, sp: float, rmse: float) -> dict[str, Any]:
    return {
        "mean_pearson": pcc,
        "mean_spearman": sp,
        "mean_rmse": rmse,
        "per_gene_pearson": {"CD4": pcc - 0.05, "CD8A": pcc + 0.05},
        "per_gene_spearman": {"CD4": sp - 0.04, "CD8A": sp + 0.04},
        "per_gene_rmse": {"CD4": rmse + 0.1, "CD8A": rmse - 0.1},
        "n_genes_scored": 2,
        "n_cells": 300,
    }


def _recovery_bundle() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "panels": {
            "synthetic-coad/tme": {
                "lumina": _ok(_recovery_metrics(0.82, 0.79, 0.41), runtime_s=0.5),
                "mean": _ok(_recovery_metrics(0.55, 0.50, 0.73), runtime_s=0.05),
                "knn": _ok(_recovery_metrics(0.61, 0.58, 0.66), runtime_s=0.2),
                "spaim": {"status": "unavailable:torch missing", "metrics": {}},
            },
            "synthetic-coad/vascular": {
                "lumina": _ok(_recovery_metrics(0.78, 0.75, 0.45), runtime_s=0.6),
                "mean": _ok(_recovery_metrics(0.52, 0.48, 0.77), runtime_s=0.06),
                "knn": _ok(_recovery_metrics(0.59, 0.55, 0.69), runtime_s=0.25),
            },
        },
    }


def _write(tmp_path: Path, bundle: dict[str, Any]) -> Path:
    p = tmp_path / "results.json"
    p.write_text(json.dumps(bundle))
    return p


# ---------------------------------------------------------------------------
# _figbase framework.
# ---------------------------------------------------------------------------


def test_figbase_schema_guard_and_helpers(tmp_path: Path) -> None:
    fb = _load("_figbase")

    good = _write(tmp_path, _recovery_bundle())
    bundle = fb.load_results(good)
    assert "panels" in bundle

    # order_methods puts the focal method first, unknowns appended sorted.
    assert fb.order_methods(["knn", "lumina", "zzz"])[0] == "lumina"
    assert fb.order_methods(["knn", "lumina", "zzz"])[-1] == "zzz"

    # iter_ok_records skips unavailable records.
    ok = list(fb.iter_ok_records(bundle))
    assert all(rec["status"] == "ok" for _p, _m, rec in ok)
    assert all(m != "spaim" for _p, m, _rec in ok)

    # save_table emits CSV + markdown with the requested columns.
    csv_path, md_path = fb.save_table(
        [{"a": 1, "b": 2.5}], tmp_path / "t.csv", columns=["a", "b"], title="T"
    )
    assert csv_path.exists() and md_path.exists()
    assert "| a | b |" in md_path.read_text()

    # require_records raises a clear error on empty input.
    with pytest.raises(fb.SchemaError):
        fb.require_records([], what="nothing")


def test_figbase_malformed_bundles_raise(tmp_path: Path) -> None:
    fb = _load("_figbase")
    no_panels = tmp_path / "a.json"
    no_panels.write_text(json.dumps({"schema_version": "1"}))
    with pytest.raises(fb.SchemaError):
        fb.load_results(no_panels)

    bad_record = tmp_path / "b.json"
    bad_record.write_text(json.dumps({"panels": {"d/p": {"lumina": "not-a-dict"}}}))
    with pytest.raises(fb.SchemaError):
        fb.load_results(bad_record)

    not_json = tmp_path / "c.json"
    not_json.write_text("{not json")
    with pytest.raises(fb.SchemaError):
        fb.load_results(not_json)


# ---------------------------------------------------------------------------
# fig_recovery_boxplots (#266).
# ---------------------------------------------------------------------------


def test_recovery_boxplots(tmp_path: Path) -> None:
    mod = _load("fig_recovery_boxplots")
    results = _write(tmp_path, _recovery_bundle())
    out = tmp_path / "fig_recovery.png"
    assert mod.main(["--results", str(results), "--out", str(out)]) == 0
    assert _is_png(out)

    rank_csv = out.with_name(out.stem + "_rank_aggregation.csv")
    cols, rows = _read_csv(rank_csv)
    assert "method" in cols and "mean_rank" in cols
    assert {r["method"] for r in rows} == {"lumina", "mean", "knn"}
    # lumina is the best method on every metric -> mean_rank 1.0, sorted first.
    assert rows[0]["method"] == "lumina"
    assert float(rows[0]["mean_rank"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# table_heldout_stratified (#270/#292).
# ---------------------------------------------------------------------------


def test_heldout_stratified_table(tmp_path: Path) -> None:
    mod = _load("table_heldout_stratified")
    # Add expression / sparsity annotations to one bundle so strata populate.
    bundle = _recovery_bundle()
    rec = bundle["panels"]["synthetic-coad/tme"]["lumina"]["metrics"]
    rec["per_gene_mean_expression"] = {"CD4": 0.1, "CD8A": 0.9}
    rec["per_gene_detection_fraction"] = {"CD4": 0.2, "CD8A": 0.8}
    results = _write(tmp_path, bundle)
    out = tmp_path / "table_heldout.csv"
    assert mod.main(["--results", str(results), "--out", str(out)]) == 0

    cols, rows = _read_csv(out)
    for c in ("method", "expression_bin", "sparsity_bin", "n_genes", "mean_pcc", "mean_rmse"):
        assert c in cols
    assert rows  # at least one stratified row
    md = out.with_suffix(".md")
    assert md.exists() and "expression_bin" in md.read_text()
    # The annotated lumina genes land in distinct expression bins (low/high).
    lumina_bins = {r["expression_bin"] for r in rows if r["method"] == "lumina"}
    assert "low" in lumina_bins and "high" in lumina_bins


# ---------------------------------------------------------------------------
# fig_runtime_memory (#267/#288).
# ---------------------------------------------------------------------------


def test_runtime_memory(tmp_path: Path) -> None:
    mod = _load("fig_runtime_memory")
    bundle = _recovery_bundle()
    # Attach peak memory to the focal method's records.
    bundle["panels"]["synthetic-coad/tme"]["lumina"]["metrics"]["peak_memory_mb"] = 120.0
    bundle["panels"]["synthetic-coad/vascular"]["lumina"]["metrics"]["peak_gpu_mem_mb"] = 140.0
    results = _write(tmp_path, bundle)
    out = tmp_path / "fig_runtime.png"
    assert mod.main(["--results", str(results), "--out", str(out)]) == 0
    assert _is_png(out)

    table = out.with_name(out.stem + "_table.csv")
    cols, rows = _read_csv(table)
    for c in ("method", "panel", "n_cells", "runtime_s", "peak_memory_mb"):
        assert c in cols
    assert len(rows) == 6  # 3 ok methods × 2 panels (spaim unavailable, skipped)


# ---------------------------------------------------------------------------
# fig_clustering_concordance (#287).
# ---------------------------------------------------------------------------


def _clustering_bundle() -> dict[str, Any]:
    def clust(prefix_delta: float) -> dict[str, Any]:
        m: dict[str, Any] = {}
        for metric, raw, enh in (
            ("ari", 0.40, 0.40 + prefix_delta),
            ("ami", 0.42, 0.42 + prefix_delta),
            ("nmi", 0.45, 0.45 + prefix_delta),
            ("homogeneity", 0.50, 0.50 + prefix_delta),
        ):
            m[f"{metric}_raw_vs_gt"] = raw
            m[f"{metric}_enhanced_vs_gt"] = enh
            m[f"{metric}_delta_over_raw"] = enh - raw
        return m

    return {
        "schema_version": "1",
        "panels": {
            "synthetic-coad/tme": {"lumina": _ok(clust(0.08))},
            "synthetic-coad/vascular": {"lumina": _ok(clust(0.04))},
        },
    }


def test_clustering_concordance(tmp_path: Path) -> None:
    mod = _load("fig_clustering_concordance")
    results = _write(tmp_path, _clustering_bundle())
    out = tmp_path / "fig_clustering.png"
    assert mod.main(["--results", str(results), "--out", str(out)]) == 0
    assert _is_png(out)

    table = out.with_name(out.stem + "_table.csv")
    cols, rows = _read_csv(table)
    for c in ("metric", "raw_vs_gt", "enhanced_vs_gt", "delta_over_raw"):
        assert c in cols
    assert {r["metric"] for r in rows} == {"ARI", "AMI", "NMI", "Homogeneity"}
    # Signed Δ is positive (enhanced > raw), averaged over the two panels.
    ari = next(r for r in rows if r["metric"] == "ARI")
    assert float(ari["delta_over_raw"]) == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# fig_risk_coverage (#291).
# ---------------------------------------------------------------------------


def test_risk_coverage_calibration(tmp_path: Path) -> None:
    mod = _load("fig_risk_coverage")
    bundle = {
        "schema_version": "1",
        "panels": {
            "synthetic-coad/tme": {
                "lumina": _ok(
                    {
                        "empirical_coverage": 0.88,
                        "nominal_coverage": 0.90,
                        "mean_interval_width": 1.2,
                    }
                ),
                "mean": _ok({"mean_pearson": 0.5}),  # no coverage -> skipped
            }
        },
    }
    results = _write(tmp_path, bundle)
    out = tmp_path / "fig_risk.png"
    assert mod.main(["--results", str(results), "--out", str(out)]) == 0
    assert _is_png(out)

    table = out.with_name(out.stem + "_calibration.csv")
    cols, rows = _read_csv(table)
    for c in ("method", "nominal_coverage", "empirical_coverage", "calibration_gap"):
        assert c in cols
    assert {r["method"] for r in rows} == {"lumina"}  # mean skipped (no coverage)
    assert float(rows[0]["calibration_gap"]) == pytest.approx(-0.02)


def test_risk_coverage_curve(tmp_path: Path) -> None:
    mod = _load("fig_risk_coverage")
    bundle = {
        "schema_version": "1",
        "panels": {
            "synthetic-coad/tme": {
                "lumina": _ok(
                    {
                        "risk_coverage": [
                            {"coverage": 1.0, "risk": 0.30},
                            {"coverage": 0.5, "risk": 0.12},
                            {"coverage": 0.2, "risk": 0.04},
                        ]
                    }
                )
            }
        },
    }
    results = _write(tmp_path, bundle)
    out = tmp_path / "fig_risk_curve.png"
    assert mod.main(["--results", str(results), "--out", str(out)]) == 0
    assert _is_png(out)


# ---------------------------------------------------------------------------
# Shared schema guard across every generator.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "fig_recovery_boxplots",
        "table_heldout_stratified",
        "fig_runtime_memory",
        "fig_clustering_concordance",
        "fig_risk_coverage",
    ],
)
def test_generator_rejects_malformed_bundle(tmp_path: Path, name: str) -> None:
    mod = _load(name)
    fb = _load("_figbase")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "1"}))  # no 'panels'
    with pytest.raises(fb.SchemaError):
        mod.render_from_bundle(bad, tmp_path / "out.png")
