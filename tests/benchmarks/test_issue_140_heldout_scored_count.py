"""Regression tests for issue #140 — held-out gene scoring denominator.

Constant / all-zero held-out genes have undefined (NaN) Pearson/Spearman and are
correctly dropped from ``mean_pearson`` / ``mean_spearman``. But ``n_genes_scored``
counted *every* held-out column, so a consumer reading the correlation mean
alongside ``n_genes_scored`` would assume a larger denominator than was actually
used — the dropped genes silently misrepresent the reported mean.

These tests pin that the true correlation denominator is now reported explicitly
and that a dropped (all-zero) gene does not inflate it.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from lumina_st.benchmarks import compute_imputation_metrics


def _imputed_adata(X_hat: np.ndarray, var_names: list[str]) -> ad.AnnData:
    a = ad.AnnData(X=X_hat.astype(np.float32))
    a.var_names = var_names
    a.layers["imputed"] = X_hat.astype(np.float32)
    return a


def test_all_zero_heldout_gene_excluded_from_correlation_denominator():
    rng = np.random.default_rng(0)
    n_cells = 50
    var_names = ["GOOD_A", "GOOD_B", "ZERO_C"]

    truth = np.zeros((n_cells, 3), dtype=np.float32)
    truth[:, 0] = rng.normal(5.0, 1.0, n_cells)
    truth[:, 1] = rng.normal(3.0, 1.0, n_cells)
    # ZERO_C left all-zero (constant) -> undefined correlation.

    X_hat = truth.copy()
    # Imputed tracks truth closely on the two good genes (high correlation).
    X_hat[:, 0] += rng.normal(0.0, 0.05, n_cells)
    X_hat[:, 1] += rng.normal(0.0, 0.05, n_cells)

    held_out = ["GOOD_A", "GOOD_B", "ZERO_C"]
    m = compute_imputation_metrics(truth, _imputed_adata(X_hat, var_names), held_out)

    # All three held-out columns were attempted.
    assert m["n_genes_scored"] == 3
    # But only the two non-constant genes actually entered the correlation means.
    assert m["n_genes_pearson_scored"] == 2
    assert m["n_genes_spearman_scored"] == 2
    # The all-zero gene did not skew the reported mean: it is the mean of the
    # two good genes' (high) correlations, not dragged toward 0 or NaN.
    assert not np.isnan(m["mean_pearson"])
    assert m["mean_pearson"] > 0.9
    # Reported correlation denominator is strictly less than the attempted count.
    assert m["n_genes_pearson_scored"] < m["n_genes_scored"]


def test_all_finite_genes_scored_count_matches_total():
    rng = np.random.default_rng(1)
    n_cells = 40
    var_names = ["G0", "G1"]
    truth = rng.normal(2.0, 1.0, (n_cells, 2)).astype(np.float32)
    X_hat = truth + rng.normal(0.0, 0.05, (n_cells, 2)).astype(np.float32)

    m = compute_imputation_metrics(truth, _imputed_adata(X_hat, var_names), ["G0", "G1"])
    assert m["n_genes_scored"] == 2
    assert m["n_genes_pearson_scored"] == 2
    assert m["n_genes_spearman_scored"] == 2
