"""Regression tests for issue #137.

The ``mean`` and ``knn`` baseline adapters used to fill every held-out gene of
a cell with a single per-cell scalar (the cell's observed-gene mean, optionally
neighbour-averaged). That collapses every held-out column to the *same* number
per cell, so per-gene Pearson is structurally identical across all held-out
genes — a degenerate strawman.

These tests pin a real baseline contract:

1. The kNN baseline produces *distinct* per-gene predicted columns when more
   than one gene is held out (gene 1 != gene 2 for at least one cell).
2. The Mean baseline produces distinct per-gene predicted columns under the
   same condition.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from lumina_st.benchmarks import AdapterInput
from lumina_st.benchmarks.adapters import KNNAdapter, MeanAdapter


def _make_adata(n_cells: int = 40, n_genes: int = 20, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    # Per-gene means that vary across columns so a proper baseline can
    # distinguish held-out gene 1 from held-out gene 2.
    per_gene_lambda = rng.uniform(0.5, 6.0, size=n_genes)
    X = np.empty((n_cells, n_genes), dtype=np.float32)
    for j in range(n_genes):
        X[:, j] = rng.poisson(per_gene_lambda[j], size=n_cells).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    return adata


def _held_out_columns(imputed: ad.AnnData, held_out: list[str]) -> np.ndarray:
    var_names = list(imputed.var_names)
    cols = [var_names.index(g) for g in held_out]
    X = imputed.layers["imputed"]
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X)[:, cols]


def test_knn_baseline_distinct_per_gene() -> None:
    """Two held-out genes must produce non-identical predicted columns."""
    adata = _make_adata()
    held_out = ["GENE_005", "GENE_010"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)
    result = KNNAdapter(k=5).run(inp)
    assert result.status == "ok", result.status
    assert result.imputed_h5ad is not None

    cols = _held_out_columns(result.imputed_h5ad, held_out)
    # Distinct per-gene columns: the two held-out genes must not produce the
    # same predicted vector across all cells.
    assert not np.allclose(cols[:, 0], cols[:, 1]), (
        "kNN baseline produced identical columns for two held-out genes — "
        "the per-cell scalar bug is still present."
    )


def test_mean_baseline_distinct_per_gene() -> None:
    """Mean baseline must also produce distinct per-gene predictions."""
    adata = _make_adata()
    held_out = ["GENE_005", "GENE_010"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)
    result = MeanAdapter().run(inp)
    assert result.status == "ok", result.status
    assert result.imputed_h5ad is not None

    cols = _held_out_columns(result.imputed_h5ad, held_out)
    # The mean baseline can return the same scalar across cells for a given
    # gene (it is a global per-gene mean), but the value must differ between
    # held-out genes when their per-gene means differ.
    assert not np.allclose(cols[:, 0], cols[:, 1]), (
        "Mean baseline produced identical columns for two held-out genes — "
        "the per-cell scalar bug is still present."
    )
