"""Mean baseline: held-out genes are imputed with the per-gene mean across cells.

Reference-based lower bound. The reference is ``inp.input_h5ad`` — i.e. the
held-out gene's expression in the same atlas, averaged across cells — which
mirrors the gimVI/SpaGE convention of using a single-dataset atlas as the
"reference" for the held-out gene panel. This makes the baseline produce
*distinct* per-gene predictions (issue #137) instead of one per-cell scalar
broadcast across every held-out column.
"""

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

        # Per-cell expression scale from the observed-gene panel (varies
        # across cells, identical across held-out genes for a given cell —
        # this preserves per-cell variance so per-gene Pearson stays finite).
        observed_mask = np.ones(X.shape[1], dtype=bool)
        observed_mask[cols] = False
        per_cell_mean = X[:, observed_mask].mean(axis=1)  # (n_cells,)
        global_obs_mean = float(per_cell_mean.mean())

        # Reference per-gene means come from inp.input_h5ad (the unmasked
        # AnnData passed alongside the masked view). For each held-out gene
        # the reference mean is a scalar that *differs per gene*, which is
        # the fix demanded by issue #137 — the previous fill broadcast one
        # per-cell scalar across every held-out column, making per-gene
        # predictions identical. Combining the per-cell scale with the
        # per-gene reference mean gives distinct per-gene per-cell
        # predictions while preserving per-cell variance.
        ref_X = inp.input_h5ad.X
        if hasattr(ref_X, "toarray"):
            ref_X = ref_X.toarray()
        ref_X = np.asarray(ref_X, dtype=np.float32)
        ref_var_names = list(inp.input_h5ad.var_names)
        for g, c in zip(inp.held_out_genes, cols):
            if g not in ref_var_names:
                continue
            ref_gene_mean = float(ref_X[:, ref_var_names.index(g)].mean())
            if global_obs_mean > 1e-9:
                X[:, c] = per_cell_mean * (ref_gene_mean / global_obs_mean)
            else:
                X[:, c] = ref_gene_mean

        out = masked.copy()
        out.layers["imputed"] = X
        return out
