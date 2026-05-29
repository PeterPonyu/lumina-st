"""kNN baseline: held-out genes are imputed from the k nearest cells in the
observed-gene space.

Neighbours are found in the *masked* observed-gene space (the only audit-safe
signal for the cell being predicted); held-out gene values are then read off
from those neighbours in the reference atlas ``inp.input_h5ad``. Each held-out
gene therefore gets its own per-gene per-cell prediction (issue #137) instead
of one per-cell scalar broadcast across every held-out column.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter


class KNNAdapter(BaseAdapter):
    name = "knn"

    def __init__(self, k: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        X = masked.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32).copy()

        if not inp.held_out_genes:
            out = masked.copy()
            out.layers["imputed"] = X
            return out

        var_names = list(masked.var_names)
        cols = [var_names.index(g) for g in inp.held_out_genes if g in var_names]
        if not cols:
            out = masked.copy()
            out.layers["imputed"] = X
            return out

        observed_mask = np.ones(X.shape[1], dtype=bool)
        observed_mask[cols] = False
        X_obs = X[:, observed_mask]

        # Pairwise cosine similarity in observed-gene space; cosine is robust
        # to sparsity.
        n_obs = X_obs.shape[0]
        norm = np.linalg.norm(X_obs, axis=1, keepdims=True)
        norm = np.where(norm < 1e-9, 1.0, norm)
        X_unit = X_obs / norm
        sim = X_unit @ X_unit.T
        np.fill_diagonal(sim, -np.inf)  # never let a cell be its own neighbour

        k = min(self.k, n_obs - 1)
        if k <= 0:
            out = masked.copy()
            out.layers["imputed"] = X
            return out

        top_idx = np.argpartition(-sim, k, axis=1)[:, :k]  # (n_cells, k)

        # Per-held-out-gene neighbour average from the reference atlas. This
        # gives a *distinct* value per gene (and per cell) — the previous
        # implementation collapsed every held-out column of a cell to one
        # per-cell scalar, which made per-gene Pearson identical across all
        # held-out genes regardless of true expression (issue #137).
        ref_X = inp.input_h5ad.X
        if hasattr(ref_X, "toarray"):
            ref_X = ref_X.toarray()
        ref_X = np.asarray(ref_X, dtype=np.float32)
        ref_var_names = list(inp.input_h5ad.var_names)
        for g, c in zip(inp.held_out_genes, cols):
            if g not in ref_var_names:
                continue
            g_ref = ref_var_names.index(g)
            # Mean over the k neighbours' reference values for this gene.
            X[:, c] = ref_X[top_idx, g_ref].mean(axis=1)

        out = masked.copy()
        out.layers["imputed"] = X
        return out
