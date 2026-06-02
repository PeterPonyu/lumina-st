"""Spatial-neighbour-average baseline: held-out genes are imputed as the mean
of each cell's k nearest spatial neighbours, where neighbours are found in the
*physical* coordinate space ``obsm['spatial']`` (NOT the observed-gene
expression space).

This is the genuinely SPATIAL lower-bound baseline and is deliberately distinct
from the gene-space ``knn`` adapter, which finds neighbours by cosine
similarity in the observed-gene panel. Here proximity is Euclidean distance
over the tissue coordinates, so the imputation exploits spatial autocorrelation
(Tobler's first law of geography) rather than transcriptional similarity. The
two adapters therefore answer different questions and are expected to disagree
on tissues where expression and geometry decouple.

Held-out gene values for the neighbours are read from the reference atlas
``inp.input_h5ad`` (the same single-dataset-as-reference convention used by the
``mean`` and ``knn`` adapters), giving a distinct per-gene per-cell prediction
(issue #137) rather than one per-cell scalar broadcast across every held-out
column.

When spatial coordinates are unavailable the whole sample is treated as a
single neighbourhood, which reduces the prediction to the reference per-gene
mean — a safe, still-distinct-per-gene fallback. Pure numpy; no external
package is imported.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter


class SpatialNeighborAvgAdapter(BaseAdapter):
    name = "spatial_neighbor_avg"

    def __init__(self, k: int = 10, spatial_key: str = "spatial", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.k = k
        self.spatial_key = spatial_key

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        X = masked.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32).copy()

        out = masked.copy()
        if not inp.held_out_genes:
            out.layers["imputed"] = X
            return out

        var_names = list(masked.var_names)
        cols = [var_names.index(g) for g in inp.held_out_genes if g in var_names]
        if not cols:
            out.layers["imputed"] = X
            return out

        ref_X = inp.input_h5ad.X
        if hasattr(ref_X, "toarray"):
            ref_X = ref_X.toarray()
        ref_X = np.asarray(ref_X, dtype=np.float32)
        ref_var_names = list(inp.input_h5ad.var_names)

        n_cells = X.shape[0]
        coords = masked.obsm.get(self.spatial_key)
        if coords is not None:
            coords = np.asarray(coords, dtype=np.float64)

        if coords is None or coords.shape[0] != n_cells or n_cells < 2:
            # Degenerate fallback: a single neighbourhood -> reference per-gene
            # mean. Still distinct per held-out gene (issue #137).
            for g, c in zip(inp.held_out_genes, cols):
                if g in ref_var_names:
                    X[:, c] = float(ref_X[:, ref_var_names.index(g)].mean())
            out.layers["imputed"] = X
            return out

        # Euclidean kNN in physical space. dist2[i, j] = ||x_i - x_j||^2.
        # Self is masked out so a cell is never its own neighbour.
        sq = np.sum(coords**2, axis=1)
        dist2 = sq[:, None] + sq[None, :] - 2.0 * (coords @ coords.T)
        np.fill_diagonal(dist2, np.inf)

        k = min(self.k, n_cells - 1)
        nn_idx = np.argpartition(dist2, k - 1, axis=1)[:, :k]  # (n_cells, k)

        # Per-held-out-gene neighbour average from the reference atlas.
        for g, c in zip(inp.held_out_genes, cols):
            if g not in ref_var_names:
                continue
            g_ref = ref_var_names.index(g)
            X[:, c] = ref_X[nn_idx, g_ref].mean(axis=1)

        out.layers["imputed"] = X
        return out
