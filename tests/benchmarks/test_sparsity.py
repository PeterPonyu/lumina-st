"""Tests for the sparsity / detection-rate sweep."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from lumina_st.benchmarks import (
    aggregate_sparsity_results,
    downsample_detection_rate,
    get_panel,
    run_sparsity_sweep,
)
from lumina_st.benchmarks.adapters import KNNAdapter, MeanAdapter


def _make_adata_with_markers(n_cells: int = 60, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    base_genes = [f"GENE_{i:03d}" for i in range(20)]
    marker_genes = list(get_panel("tme-immune-stromal").genes)
    all_genes = base_genes + marker_genes
    X = rng.poisson(3.0, size=(n_cells, len(all_genes))).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = all_genes
    adata.obs["cancer_type"] = ["COAD"] * n_cells
    return adata


# -- Downsampler basics ---------------------------------------------------


def test_downsample_fraction_one_returns_identical_X():
    adata = _make_adata_with_markers()
    out = downsample_detection_rate(adata, fraction=1.0, seed=0)
    assert np.array_equal(np.asarray(out.X), np.asarray(adata.X))


def test_downsample_fraction_zero_zeros_non_excluded_columns():
    adata = _make_adata_with_markers()
    out = downsample_detection_rate(adata, fraction=0.0, seed=0, exclude_genes=())
    assert np.all(np.asarray(out.X) == 0.0)


def test_downsample_preserves_excluded_columns():
    adata = _make_adata_with_markers()
    excl = ["GENE_005", "GENE_010", "CD4"]
    out = downsample_detection_rate(adata, fraction=0.0, seed=0, exclude_genes=excl)

    var_names = list(out.var_names)
    excl_idx = [var_names.index(g) for g in excl]
    other_idx = [j for j in range(out.n_vars) if j not in excl_idx]

    # Excluded columns must be unchanged.
    assert np.array_equal(
        np.asarray(out.X)[:, excl_idx],
        np.asarray(adata.X)[:, excl_idx],
    )
    # Non-excluded are zeroed at fraction=0.
    assert np.all(np.asarray(out.X)[:, other_idx] == 0.0)


def test_downsample_monotonic_sparsity_decreases_with_fraction():
    """Lower fraction → more zeros in the non-excluded columns."""
    adata = _make_adata_with_markers(n_cells=200, seed=0)
    excl = list(get_panel("tme-immune-stromal").genes)
    sparsities = []
    for f in (1.0, 0.5, 0.25, 0.1):
        out = downsample_detection_rate(adata, fraction=f, seed=42, exclude_genes=excl)
        var_names = list(out.var_names)
        excl_idx = {var_names.index(g) for g in excl if g in var_names}
        non_excl = [j for j in range(out.n_vars) if j not in excl_idx]
        sparsities.append(float(np.mean(np.asarray(out.X)[:, non_excl] == 0)))

    # Should be monotonically non-decreasing
    assert sparsities == sorted(sparsities), \
        f"sparsity should grow as fraction shrinks; got {sparsities}"
    assert sparsities[0] < sparsities[-1], "extreme fractions must differ"


def test_downsample_integer_counts_use_binomial_thinning():
    """Integer-valued matrices get Binomial-per-count semantics; survival ≤ original."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 20, size=(100, 5)).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = [f"G{i}" for i in range(5)]
    out = downsample_detection_rate(a, fraction=0.5, seed=0)
    # No counts may exceed the original (Binomial(k, p) is bounded by k).
    assert np.all(np.asarray(out.X) <= X)


def test_downsample_raises_on_out_of_range_fraction():
    adata = _make_adata_with_markers()
    with pytest.raises(ValueError):
        downsample_detection_rate(adata, fraction=1.5, seed=0)
    with pytest.raises(ValueError):
        downsample_detection_rate(adata, fraction=-0.1, seed=0)


# -- Audit boundary -------------------------------------------------------


def test_sparsity_sweep_does_not_leak_held_out_to_adapter():
    """At every fraction, held-out gene values must remain zero in the
    AnnData the adapter sees."""
    captured: list[np.ndarray] = []

    from lumina_st.benchmarks.contract import BaseAdapter

    class SnoopAdapter(BaseAdapter):
        name = "snoop"

        def _impute(self, masked, inp):
            var_names = list(masked.var_names)
            cols = [var_names.index(g) for g in inp.held_out_genes if g in var_names]
            captured.append(np.asarray(masked.X)[:, cols].copy())
            out = masked.copy()
            out.layers["imputed"] = np.asarray(masked.X).copy()
            return out

    adata = _make_adata_with_markers()
    panel = get_panel("tme-immune-stromal")
    _ = run_sparsity_sweep(
        [SnoopAdapter()], adata, panel, fractions=[1.0, 0.5, 0.1], seed=0,
    )

    assert len(captured) == 3
    for seen in captured:
        assert np.all(seen == 0.0), \
            "AUDIT FAILURE: a sparsity sweep leaked held-out values to an adapter"


# -- Sweep runner ---------------------------------------------------------


def test_run_sparsity_sweep_emits_one_row_per_fraction_adapter_pair():
    adata = _make_adata_with_markers()
    panel = get_panel("tme-immune-stromal")
    rows = run_sparsity_sweep(
        [MeanAdapter(), KNNAdapter(k=5)],
        adata, panel, fractions=[1.0, 0.5, 0.1], seed=42,
        dataset_name="synth-coad",
    )
    assert len(rows) == 6  # 3 fractions × 2 adapters
    methods = {r["method"] for r in rows}
    assert methods == {"mean", "knn"}
    fractions = sorted({r["fraction"] for r in rows})
    assert fractions == [0.1, 0.5, 1.0]

    for r in rows:
        assert "sparsity_observed_thinned" in r
        assert 0.0 <= r["sparsity_observed_thinned"] <= 1.0
        # Metrics computed against the original truth (so n_genes_scored matches)
        if r["status"] == "ok":
            assert r["metrics_json"]["n_genes_scored"] == len(panel.genes)


def test_aggregate_sparsity_results_has_documented_schema():
    adata = _make_adata_with_markers()
    panel = get_panel("tme-immune-stromal")
    rows = run_sparsity_sweep(
        [MeanAdapter()], adata, panel, fractions=[1.0, 0.5], seed=0,
        dataset_name="synth-coad",
    )
    aggregated = aggregate_sparsity_results(rows, "synth-coad", panel.name)

    assert aggregated["schema_version"] == "1"
    assert aggregated["dataset"] == "synth-coad"
    assert aggregated["panel"] == panel.name
    assert aggregated["n_rows"] == 2
    assert len(aggregated["rows"]) == 2


def test_sweep_higher_sparsity_observed_at_lower_fraction():
    """Sanity: the observed-sparsity field grows as fraction shrinks."""
    adata = _make_adata_with_markers(n_cells=120, seed=0)
    panel = get_panel("tme-immune-stromal")
    rows = run_sparsity_sweep(
        [MeanAdapter()], adata, panel, fractions=[1.0, 0.25, 0.05], seed=0,
    )
    by_fraction = sorted(rows, key=lambda r: r["fraction"])
    sparsities = [r["sparsity_observed_thinned"] for r in by_fraction]
    # Lower fraction → higher sparsity_observed (descending fraction = ascending sparsity)
    assert sparsities[0] >= sparsities[-1], \
        f"sparsity should grow as fraction shrinks; got {sparsities} for fractions={[r['fraction'] for r in by_fraction]}"
