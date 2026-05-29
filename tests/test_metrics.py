"""Regression tests for the held-out-gene imputation metrics.

Issue #133 — `mean_rmse` was computed on per-gene z-scored vectors. For
two z-scored vectors, mean((zt - zh)^2) = 2*(1 - pearson_r), so the RMSE
was an exact affine-invariant restatement of Pearson:
RMSE = sqrt(2*(1 - r)). It therefore carried no information beyond
`mean_pearson` and was blind to absolute magnitude / scale error. The fix
computes RMSE on raw values.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from lumina_st.benchmarks.contract import compute_imputation_metrics


def _adata(X: np.ndarray, var_names: list[str]) -> ad.AnnData:
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.var_names = var_names
    return adata


def test_mean_rmse_not_pearson_restatement() -> None:
    """RMSE must penalise a scale/offset error that Pearson is blind to (#133)."""
    rng = np.random.default_rng(0)
    n, g = 50, 4
    truth = rng.normal(5.0, 2.0, size=(n, g)).astype(np.float32)
    var_names = [f"GENE_{i}" for i in range(g)]

    # Imputed = affine transform of truth: perfect Pearson (r = 1) but a large
    # absolute magnitude + offset error.
    imputed = _adata(truth * 5.0 + 3.0, var_names)

    metrics = compute_imputation_metrics(truth, imputed, held_out_genes=var_names)

    # Affine maps are "perfect" under Pearson.
    assert metrics["mean_pearson"] == pytest.approx(1.0, abs=1e-5)

    # If RMSE were the z-scored restatement sqrt(2*(1 - r)), then r = 1 would
    # force RMSE = 0. Raw RMSE must instead be clearly positive because the
    # prediction is wrong in scale and offset.
    assert metrics["mean_rmse"] > 1.0, (
        "mean_rmse collapses to ~0 for an affine (perfect-Pearson) prediction; "
        "it is still an affine-invariant restatement of Pearson (issue #133)"
    )


def test_mean_rmse_zero_on_perfect_imputation() -> None:
    """Sanity: identical truth/imputed still yields RMSE = 0."""
    rng = np.random.default_rng(1)
    truth = rng.normal(3.0, 1.0, size=(30, 3)).astype(np.float32)
    var_names = [f"GENE_{i}" for i in range(3)]
    imputed = _adata(truth.copy(), var_names)

    metrics = compute_imputation_metrics(truth, imputed, held_out_genes=var_names)
    assert metrics["mean_rmse"] == pytest.approx(0.0, abs=1e-5)
