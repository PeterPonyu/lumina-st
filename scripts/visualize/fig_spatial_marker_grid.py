"""
Wave 2 — per-spatial-patch grid of raw vs imputed for the top-K markers.

Splits each slice into a few spatial patches (quadrants), and renders a
grid: patches (rows) x [raw, imputed] for each top-variance marker. This is
the closest we can get to the upstream "patch grid" figure without an
external method baseline (those land in Wave 3+).

Outputs:
  <out_dir>/spatial_marker_grid.png
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from scripts.visualize._plot_utils import to_dense, topn_variable_genes


def _quadrant_assignment(xy: np.ndarray) -> np.ndarray:
    """Assign each cell to a 2x2 spatial patch (Q0..Q3) by median split."""
    mx = np.median(xy[:, 0])
    my = np.median(xy[:, 1])
    right = xy[:, 0] >= mx
    top = xy[:, 1] >= my
    return (right.astype(int) + 2 * top.astype(int))


def render_spatial_marker_grid(
    adata: AnnData, out_path: Path, n_markers: int = 3
) -> None:
    if "spatial" not in adata.obsm or "imputed" not in adata.layers:
        return
    spatial = np.asarray(adata.obsm["spatial"])[:, :2]
    raw = to_dense(adata.X).astype(np.float32)
    imp = np.asarray(adata.layers["imputed"], dtype=np.float32)

    markers = topn_variable_genes(adata, n=n_markers)
    quad = _quadrant_assignment(spatial)
    q_ids: List[int] = sorted(np.unique(quad).tolist())
    n_quads = len(q_ids)
    n_cols = 2 * n_markers  # raw + imputed per gene
    fig, axes = plt.subplots(n_quads, n_cols, figsize=(2.4 * n_cols, 2.4 * n_quads), squeeze=False)

    for qi, q in enumerate(q_ids):
        mask = (quad == q)
        for mi, gi in enumerate(markers):
            gene = str(adata.var_names[gi])
            v_raw = raw[mask, gi]
            v_imp = imp[mask, gi]
            vmax = float(np.percentile(np.concatenate([v_raw, v_imp]), 99) + 1e-9)
            ax_r = axes[qi, 2 * mi]
            ax_i = axes[qi, 2 * mi + 1]
            ax_r.scatter(spatial[mask, 0], spatial[mask, 1], c=v_raw, s=3, cmap="magma", vmin=0, vmax=vmax)
            ax_i.scatter(spatial[mask, 0], spatial[mask, 1], c=v_imp, s=3, cmap="magma", vmin=0, vmax=vmax)
            ax_r.set_title(f"Q{q} · {gene} · raw", fontsize=7)
            ax_i.set_title(f"Q{q} · {gene} · imputed", fontsize=7)
            for ax in (ax_r, ax_i):
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_aspect("equal")
    fig.suptitle("Per-patch raw vs LuminaST-imputed spatial expression (top-variance markers)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
