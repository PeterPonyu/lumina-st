"""
Wave 1 — comparative UMAPs across three internal representations.

Panels:
  1. raw PCA of `adata.X`
  2. `obsm['latent_observed']`  (encoder pre-flow)
  3. `obsm['latent_enhanced']`  (encoder post-flow)

Each colored by Leiden of latent_enhanced and by the in-data true label.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from anndata import AnnData

from scripts.visualize._plot_utils import (
    ensure_leiden,
    pick_label_key,
    stable_categorical_colors,
    to_dense,
)


def _umap_from_rep(adata: AnnData, key: str) -> np.ndarray:
    a = adata.copy()
    sc.pp.neighbors(a, use_rep=key, n_neighbors=15, key_added="_tmp")
    sc.tl.umap(a, neighbors_key="_tmp")
    return np.asarray(a.obsm["X_umap"])


def _umap_from_pca(adata: AnnData) -> np.ndarray:
    a = adata.copy()
    X = to_dense(a.X).astype(np.float32)
    n_comp = min(50, X.shape[1] - 1, X.shape[0] - 1)
    a.obsm["X_pca_raw"] = sc.pp.pca(X, n_comps=max(2, n_comp))
    sc.pp.neighbors(a, use_rep="X_pca_raw", n_neighbors=15, key_added="_tmp")
    sc.tl.umap(a, neighbors_key="_tmp")
    return np.asarray(a.obsm["X_umap"])


def render_comparative_umaps(adata: AnnData, out_path: Path) -> None:
    ensure_leiden(adata, use_rep="latent_enhanced", key="leiden_bio")
    label_key = pick_label_key(adata, ["cancer_type", "spatial_cluster", "annotation"])

    panels = []
    panels.append(("raw PCA", _umap_from_pca(adata)))
    if "latent_observed" in adata.obsm:
        panels.append(("latent_observed", _umap_from_rep(adata, "latent_observed")))
    if "latent_enhanced" in adata.obsm:
        panels.append(("latent_enhanced", _umap_from_rep(adata, "latent_enhanced")))

    color_keys = ["leiden_bio"] + ([label_key] if label_key else [])
    n_rows = len(color_keys)
    n_cols = len(panels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.4 * n_rows), squeeze=False)

    for ci, (name, coords) in enumerate(panels):
        for ri, ck in enumerate(color_keys):
            ax = axes[ri, ci]
            cats = adata.obs[ck].astype(str)
            palette = stable_categorical_colors(cats)
            for c in cats.unique():
                m = (cats == c).to_numpy()
                ax.scatter(coords[m, 0], coords[m, 1], s=3, color=palette[c], label=str(c))
            ax.set_title(f"{name}\n· colour: {ck}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if len(cats.unique()) <= 8:
                ax.legend(fontsize=5, markerscale=1.5, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
