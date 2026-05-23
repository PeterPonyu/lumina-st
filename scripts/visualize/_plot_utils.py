"""Shared plotting helpers for the biology figure pack."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


CATEGORICAL_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def to_dense(x) -> np.ndarray:
    """Make sure a possibly-sparse expression matrix is a dense ndarray."""
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)


def topn_variable_genes(adata: AnnData, n: int = 50) -> List[int]:
    """Return indices of the top-N highly variable genes by raw variance."""
    X = to_dense(adata.X)
    variances = X.var(axis=0)
    n = min(n, X.shape[1])
    return list(np.argsort(variances)[-n:][::-1])


def select_markers_by_group(
    adata: AnnData, group_key: str, n_per_group: int = 2
) -> Dict[str, List[str]]:
    """For each category in `group_key`, return the top-N genes by mean expression
    in that group relative to other groups (very lightweight surrogate for DE)."""
    X = to_dense(adata.X)
    gene_names = list(adata.var_names)
    if group_key not in adata.obs:
        return {}
    groups = adata.obs[group_key].astype(str)
    out: Dict[str, List[str]] = {}
    overall_mean = X.mean(axis=0)
    for g in groups.unique():
        mask = (groups == g).to_numpy()
        if mask.sum() < 2:
            continue
        group_mean = X[mask].mean(axis=0)
        score = group_mean - overall_mean
        idx = np.argsort(score)[-n_per_group:][::-1]
        out[g] = [gene_names[i] for i in idx]
    return out


def subsample_adata(adata: AnnData, max_cells: int, seed: int = 42) -> AnnData:
    if adata.n_obs <= max_cells:
        return adata
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, max_cells, replace=False)
    return adata[idx].copy()


def ensure_leiden(adata: AnnData, use_rep: str = "latent_enhanced", key: str = "leiden_bio") -> None:
    if key in adata.obs:
        return
    if use_rep not in adata.obsm:
        return
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15, key_added="_bio")
    sc.tl.leiden(adata, key_added=key, neighbors_key="_bio", resolution=1.0)


def ensure_umap(adata: AnnData, use_rep: str = "latent_enhanced") -> None:
    if "X_umap_bio" in adata.obsm:
        return
    if use_rep not in adata.obsm:
        return
    if "_bio" not in adata.uns.get("neighbors", {}).get("params", {}).get("use_rep", "") and "neighbors_bio" not in adata.uns:
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15, key_added="_bio")
    sc.tl.umap(adata, neighbors_key="_bio")
    adata.obsm["X_umap_bio"] = adata.obsm["X_umap"]


def pick_label_key(adata: AnnData, candidates: Iterable[str]) -> Optional[str]:
    for k in candidates:
        if k in adata.obs:
            return k
    return None


def stable_categorical_colors(values: pd.Series) -> Dict[str, str]:
    cats = pd.Categorical(values).categories.tolist()
    return {c: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, c in enumerate(cats)}
