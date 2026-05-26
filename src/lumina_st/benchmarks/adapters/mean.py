"""Mean baseline: held-out genes are imputed with the per-gene mean across cells.

Trivial lower bound. Always available."""

from __future__ import annotations

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter


class MeanAdapter(BaseAdapter):
    name = "mean"

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        X = masked.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32).copy()

        # For held-out columns (now zeros), substitute the mean of the *observed*
        # cells. Since the held-out columns are zeros, the mean would be ~0; so
        # we instead use the mean across the *other* genes per cell as the fill —
        # the standard "mean" baseline used in held-out gene imputation benchmarks.
        if inp.held_out_genes:
            var_names = list(masked.var_names)
            cols = [var_names.index(g) for g in inp.held_out_genes if g in var_names]
            if cols:
                # Per-cell mean over observed (non-zero, non-held-out) columns
                observed_mask = np.ones(X.shape[1], dtype=bool)
                observed_mask[cols] = False
                per_cell_mean = X[:, observed_mask].mean(axis=1, keepdims=True)
                X[:, cols] = per_cell_mean
        out = masked.copy()
        out.layers["imputed"] = X
        return out
