"""Tests for the benchmark runner and panel definitions."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np

from lumina_st.benchmarks import (
    MarkerPanel,
    aggregate_results,
    get_panel,
    run_panel,
    write_results_json,
)
from lumina_st.benchmarks.adapters import KNNAdapter, MeanAdapter


def _make_adata_with_markers(n_cells: int = 50, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    base_genes = [f"GENE_{i:03d}" for i in range(20)]
    marker_genes = ["CD4", "CD8A", "CD68", "EPCAM", "ENG", "POSTN"]
    all_genes = base_genes + marker_genes
    X = rng.poisson(2.0, size=(n_cells, len(all_genes))).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = all_genes
    adata.obs["cancer_type"] = ["COAD"] * n_cells
    return adata


def test_get_panel_returns_registered_panel():
    panel = get_panel("tme-immune-stromal")
    assert isinstance(panel, MarkerPanel)
    assert "CD4" in panel.genes
    assert "EPCAM" in panel.genes


def test_run_panel_executes_every_adapter_once():
    adata = _make_adata_with_markers()
    panel = get_panel("tme-immune-stromal")
    adapters = [MeanAdapter(), KNNAdapter(k=5)]

    results = run_panel(
        adapters, adata, panel, seed=0, dataset_name="synthetic-coad",
    )

    assert len(results) == 2
    methods = {r.method for r in results}
    assert methods == {"mean", "knn"}
    for r in results:
        assert r.status == "ok"
        assert r.imputed_h5ad is not None
        assert r.metrics_json["n_genes_scored"] == len(panel.genes)


def test_runner_does_not_leak_held_out_genes_to_adapter():
    """Audit boundary: the masked input given to each adapter has zeros at held-out positions."""
    captured: dict = {}

    from lumina_st.benchmarks.contract import AdapterInput, BaseAdapter

    class SnoopingAdapter(BaseAdapter):
        name = "snoop"

        def _impute(self, masked, inp):
            var_names = list(masked.var_names)
            cols = [var_names.index(g) for g in inp.held_out_genes if g in var_names]
            captured["seen"] = np.asarray(masked.X)[:, cols].copy()
            out = masked.copy()
            out.layers["imputed"] = np.asarray(masked.X).copy()
            return out

    adata = _make_adata_with_markers()
    panel = get_panel("tme-immune-stromal")

    _ = run_panel([SnoopingAdapter()], adata, panel, seed=0)
    assert np.all(captured["seen"] == 0.0), \
        "AUDIT FAILURE: runner leaked held-out values to adapter"


def test_aggregate_and_write_results_json(tmp_path: Path):
    adata = _make_adata_with_markers()
    panel = get_panel("tme-immune-stromal")

    results = run_panel(
        [MeanAdapter(), KNNAdapter(k=5)],
        adata, panel, seed=42, dataset_name="synthetic-coad",
    )
    aggregated = aggregate_results({("synthetic-coad", panel.name): results})

    assert aggregated["schema_version"] == "1"
    key = f"synthetic-coad/{panel.name}"
    assert key in aggregated["panels"]
    assert set(aggregated["panels"][key]) == {"mean", "knn"}

    out_path = write_results_json(aggregated, tmp_path / "results.json")
    loaded = json.loads(out_path.read_text())

    assert loaded == aggregated
    # provenance recorded
    prov = loaded["panels"][key]["mean"]["provenance"]
    assert prov["method"] == "mean"
    assert prov["seed"] == 42
    assert "numpy" in prov["dependency_notes"]


def test_runner_records_unavailable_methods_in_output():
    from lumina_st.benchmarks.contract import BaseAdapter

    class NeverReady(BaseAdapter):
        name = "never"

        def _check_available(self):
            return False, "external-tool-x-not-installed"

        def _impute(self, masked, inp):
            raise RuntimeError("unreachable")

    adata = _make_adata_with_markers()
    panel = get_panel("tme-immune-stromal")

    results = run_panel(
        [MeanAdapter(), NeverReady()],
        adata, panel, seed=0, dataset_name="synthetic-coad",
    )
    aggregated = aggregate_results({("synthetic-coad", panel.name): results})

    key = f"synthetic-coad/{panel.name}"
    assert aggregated["panels"][key]["never"]["status"].startswith("unavailable:")
    assert aggregated["panels"][key]["mean"]["status"] == "ok"
