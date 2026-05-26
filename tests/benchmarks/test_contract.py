"""Contract tests for the benchmark adapter base.

Verifies the three audit properties: comparability, audit-safety, provenance.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from lumina_st.benchmarks import (
    AdapterInput,
    BaseAdapter,
    compute_imputation_metrics,
)
from lumina_st.benchmarks.adapters import KNNAdapter, MeanAdapter


def _make_synthetic_adata(n_cells: int = 60, n_genes: int = 30, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    obs = {"cancer_type": ["COAD"] * n_cells}
    var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = var_names
    return adata


def test_adapter_input_masked_input_zeros_held_out_genes():
    adata = _make_synthetic_adata()
    held_out = ["GENE_005", "GENE_010", "GENE_015"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=42)

    masked = inp.masked_input()

    var_names = list(masked.var_names)
    idx = [var_names.index(g) for g in held_out]

    assert np.all(masked.X[:, idx] == 0.0), "held-out columns must be zero in masked input"

    other_idx = [i for i in range(masked.n_vars) if i not in idx]
    assert np.any(masked.X[:, other_idx] != 0.0), "non-held-out columns must remain"


def test_adapter_input_masked_input_does_not_mutate_original():
    adata = _make_synthetic_adata()
    original_X = np.asarray(adata.X).copy()
    held_out = ["GENE_005"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)

    _ = inp.masked_input()

    assert np.array_equal(np.asarray(adata.X), original_X), "original AnnData must not be mutated"


def test_adapter_input_truth_matrix_returns_unmasked():
    adata = _make_synthetic_adata()
    held_out = ["GENE_005", "GENE_010"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)

    truth = inp.truth_matrix()
    var_names = list(adata.var_names)
    idx = [var_names.index(g) for g in held_out]

    assert np.any(truth[:, idx] != 0.0), "truth matrix must retain values at held-out genes"


def test_mean_adapter_runs_and_returns_correct_schema():
    adata = _make_synthetic_adata()
    held_out = ["GENE_005", "GENE_010"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)

    result = MeanAdapter().run(inp)

    assert result.status == "ok"
    assert result.method == "mean"
    assert result.imputed_h5ad is not None
    assert "imputed" in result.imputed_h5ad.layers
    assert result.imputed_h5ad.layers["imputed"].shape == adata.X.shape
    assert result.metrics_json["n_genes_scored"] == len(held_out)
    assert result.metrics_json["n_cells"] == adata.n_obs
    assert result.provenance.method == "mean"
    assert result.provenance.seed == 0
    assert "numpy" in result.provenance.dependency_notes
    assert result.runtime_s >= 0.0


def test_mean_adapter_imputed_values_are_nonzero_at_held_out_positions():
    adata = _make_synthetic_adata()
    held_out = ["GENE_005", "GENE_010"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)

    result = MeanAdapter().run(inp)
    imputed = result.imputed_h5ad
    assert imputed is not None
    var_names = list(imputed.var_names)
    idx = [var_names.index(g) for g in held_out]

    assert np.any(imputed.layers["imputed"][:, idx] > 0.0), \
        "mean imputer must produce non-zero values at held-out positions"


def test_knn_adapter_runs_and_outputs_imputed_layer():
    adata = _make_synthetic_adata(n_cells=80, n_genes=40)
    held_out = ["GENE_005", "GENE_010", "GENE_020"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)

    result = KNNAdapter(k=5).run(inp)

    assert result.status == "ok"
    assert "imputed" in result.imputed_h5ad.layers
    assert -1.0 <= result.metrics_json["mean_pearson"] <= 1.0 or np.isnan(
        result.metrics_json["mean_pearson"]
    )


def test_metric_helper_reports_per_gene_pearson_for_held_out_only():
    adata = _make_synthetic_adata()
    held_out = ["GENE_005", "GENE_010"]
    truth = np.asarray(adata.X)
    out = adata.copy()
    out.layers["imputed"] = truth.copy()  # perfect imputation

    metrics = compute_imputation_metrics(truth, out, held_out_genes=held_out)

    assert set(metrics["per_gene_pearson"].keys()) == set(held_out), \
        "metrics must only score the held-out gene panel"
    for v in metrics["per_gene_pearson"].values():
        assert v == pytest.approx(1.0, abs=1e-6), "perfect imputation should yield Pearson = 1"


def test_unavailable_adapter_returns_status_unavailable():
    class NeverAvailable(BaseAdapter):
        name = "never"

        def _check_available(self):
            return False, "missing dependency Foo"

        def _impute(self, masked, inp):
            raise RuntimeError("should not be called")

    adata = _make_synthetic_adata()
    inp = AdapterInput(input_h5ad=adata, held_out_genes=["GENE_001"], seed=0)

    result = NeverAvailable().run(inp)

    assert result.status.startswith("unavailable:")
    assert "missing dependency Foo" in result.status
    assert result.imputed_h5ad is None


def test_failing_adapter_returns_error_status_with_message():
    class BrokenAdapter(BaseAdapter):
        name = "broken"

        def _impute(self, masked, inp):
            raise ValueError("intentional failure")

    adata = _make_synthetic_adata()
    inp = AdapterInput(input_h5ad=adata, held_out_genes=["GENE_001"], seed=0)

    result = BrokenAdapter().run(inp)

    assert result.status.startswith("error:")
    assert "intentional failure" in result.status
    assert result.imputed_h5ad is None
    assert result.runtime_s >= 0.0


def test_audit_boundary_held_out_truth_not_visible_to_adapter():
    """The adapter must only see the masked input, never the truth matrix."""
    captured = {}

    class SnoopAdapter(BaseAdapter):
        name = "snoop"

        def _impute(self, masked, inp):
            var_names = list(masked.var_names)
            cols = [var_names.index(g) for g in inp.held_out_genes if g in var_names]
            captured["held_out_values"] = np.asarray(masked.X)[:, cols].copy()
            out = masked.copy()
            out.layers["imputed"] = np.asarray(masked.X).copy()
            return out

    adata = _make_synthetic_adata()
    held_out = ["GENE_005"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)

    _ = SnoopAdapter().run(inp)

    assert np.all(captured["held_out_values"] == 0.0), \
        "AUDIT FAILURE: adapter saw non-zero values at held-out positions"
