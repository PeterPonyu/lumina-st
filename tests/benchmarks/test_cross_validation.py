"""Tests for the leave-one-context cross-validation harness."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from lumina_st.benchmarks import (
    AdapterInput,
    aggregate_cv_results,
    bootstrap_ci,
    get_panel,
    run_cv,
    stateless_factory,
    summarize_cv,
)
from lumina_st.benchmarks.adapters import KNNAdapter, MeanAdapter
from lumina_st.benchmarks.contract import BaseAdapter


def _make_context(name: str, n_cells: int = 60, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    base = [f"GENE_{i:03d}" for i in range(20)]
    markers = list(get_panel("tme-immune-stromal").genes)
    all_genes = base + markers
    X = rng.poisson(3.0, size=(n_cells, len(all_genes))).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = all_genes
    adata.obs["cancer_type"] = [name] * n_cells
    return adata


# -- Bootstrap CI -----------------------------------------------------


def test_bootstrap_ci_on_constant_vector_is_tight():
    """CI for a constant vector collapses to a point."""
    ci = bootstrap_ci([0.5, 0.5, 0.5, 0.5], n_boot=200, seed=0)
    assert ci["point"] == pytest.approx(0.5)
    assert ci["lower"] == pytest.approx(0.5)
    assert ci["upper"] == pytest.approx(0.5)
    assert ci["n"] == 4


def test_bootstrap_ci_covers_known_mean_for_large_n():
    """Bootstrap CI should contain the true mean for well-behaved input."""
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=2.0, scale=0.5, size=200).tolist()
    ci = bootstrap_ci(samples, n_boot=500, alpha=0.05, seed=0)
    assert ci["lower"] < 2.0 < ci["upper"]


def test_bootstrap_ci_handles_nans_by_filtering():
    ci = bootstrap_ci([1.0, 2.0, float("nan"), 3.0], n_boot=200, seed=0)
    assert ci["n"] == 3  # NaN dropped


def test_bootstrap_ci_empty_input_returns_nan():
    ci = bootstrap_ci([], n_boot=10, seed=0)
    assert np.isnan(ci["point"])
    assert ci["n"] == 0


# -- run_cv basic ----------------------------------------------------


def test_run_cv_generates_one_fold_per_context():
    contexts = {
        "COAD": _make_context("COAD", seed=0),
        "OV": _make_context("OV", seed=1),
        "LIHC": _make_context("LIHC", seed=2),
    }
    panel = get_panel("tme-immune-stromal")

    folds = run_cv(contexts, stateless_factory(MeanAdapter), panel)
    assert len(folds) == 3
    test_contexts = {f.test_context for f in folds}
    assert test_contexts == {"COAD", "OV", "LIHC"}

    for f in folds:
        # Train set excludes the test context
        assert f.test_context not in f.train_contexts
        assert len(f.train_contexts) == 2
        assert f.result["status"] == "ok"


def test_run_cv_factory_receives_correct_partition():
    """The factory must be called with (test_name, others_dict_excluding_test)."""
    captured: list[tuple[str, set]] = []

    def _spy_factory(test_context: str, others: dict[str, ad.AnnData]):
        captured.append((test_context, set(others.keys())))
        return MeanAdapter()

    contexts = {n: _make_context(n, seed=i) for i, n in enumerate(("A", "B", "C"))}
    panel = get_panel("tme-immune-stromal")
    _ = run_cv(contexts, _spy_factory, panel)

    assert len(captured) == 3
    for test_name, others in captured:
        assert test_name not in others
        assert others == {"A", "B", "C"} - {test_name}


def test_run_cv_audit_boundary_held_out_genes_never_seen_by_adapter():
    """The held-out marker panel must arrive zeroed at every fold."""
    captured: list[np.ndarray] = []

    class SnoopAdapter(BaseAdapter):
        name = "snoop"

        def _impute(self, masked, inp):
            var_names = list(masked.var_names)
            cols = [var_names.index(g) for g in inp.held_out_genes if g in var_names]
            captured.append(np.asarray(masked.X)[:, cols].copy())
            out = masked.copy()
            out.layers["imputed"] = np.asarray(masked.X).copy()
            return out

    contexts = {n: _make_context(n, seed=i) for i, n in enumerate(("A", "B", "C"))}
    panel = get_panel("tme-immune-stromal")

    factory = stateless_factory(SnoopAdapter)
    _ = run_cv(contexts, factory, panel)

    assert len(captured) == 3
    for seen in captured:
        assert np.all(seen == 0.0), \
            "AUDIT FAILURE: CV runner leaked held-out genes to adapter"


def test_run_cv_works_with_stateful_factory():
    """Stateful factories see the real train dict and can use it."""
    train_sizes_seen: list[int] = []

    def _train_aware_factory(test_context: str, train_dict: dict[str, ad.AnnData]):
        # A real factory would fit a model on train_dict; here we just record sizes.
        n_train_cells = sum(ad.n_obs for ad in train_dict.values())
        train_sizes_seen.append(n_train_cells)
        return MeanAdapter()

    contexts = {
        "A": _make_context("A", n_cells=20, seed=0),
        "B": _make_context("B", n_cells=30, seed=1),
        "C": _make_context("C", n_cells=40, seed=2),
    }
    panel = get_panel("tme-immune-stromal")
    _ = run_cv(contexts, _train_aware_factory, panel)

    # Leaving out A: train cells = 30 + 40 = 70; B: 20+40=60; C: 20+30=50
    assert sorted(train_sizes_seen) == [50, 60, 70]


# -- summarize_cv ----------------------------------------------------


def test_summarize_cv_emits_documented_schema():
    contexts = {n: _make_context(n, seed=i) for i, n in enumerate(("A", "B", "C"))}
    panel = get_panel("tme-immune-stromal")
    folds = run_cv(contexts, stateless_factory(KNNAdapter, k=5), panel)
    summary = summarize_cv(folds, n_boot=100, seed=0)

    assert summary["schema_version"] == "1"
    assert summary["n_folds"] == 3
    assert set(summary["fold_contexts"]) == {"A", "B", "C"}
    assert "mean_pearson" in summary["per_metric"]
    metric = summary["per_metric"]["mean_pearson"]
    for k in ("point", "lower", "upper", "n", "per_fold"):
        assert k in metric
    assert len(metric["per_fold"]) == 3
    assert metric["lower"] <= metric["point"] <= metric["upper"]


def test_summarize_cv_records_per_fold_status():
    """Unavailable adapters propagate their status into the summary."""
    class _NeverReady(BaseAdapter):
        name = "never-ready"

        def _check_available(self):
            return False, "external-tool-x"

        def _impute(self, masked, inp):
            raise RuntimeError("unreachable")

    contexts = {n: _make_context(n, seed=i) for i, n in enumerate(("A", "B"))}
    panel = get_panel("tme-immune-stromal")
    folds = run_cv(contexts, stateless_factory(_NeverReady), panel)
    summary = summarize_cv(folds, n_boot=20)
    for ctx in ("A", "B"):
        assert summary["per_fold_status"][ctx].startswith("unavailable:")


def test_aggregate_cv_results_is_json_serializable():
    import json

    contexts = {n: _make_context(n, seed=i) for i, n in enumerate(("A", "B"))}
    panel = get_panel("tme-immune-stromal")
    folds = run_cv(contexts, stateless_factory(MeanAdapter), panel)
    summary = summarize_cv(folds, n_boot=50)
    aggregated = aggregate_cv_results(folds, panel.name, summary=summary)

    assert aggregated["schema_version"] == "1"
    assert aggregated["panel"] == panel.name
    assert len(aggregated["folds"]) == 2
    # Round-trips
    s = json.dumps(aggregated, default=str)
    loaded = json.loads(s)
    assert loaded["n_folds"] == 2
