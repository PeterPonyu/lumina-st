"""
Wave 1 — canonical-lineage-marker dot plot on Leiden clusters.

Markers (hardcoded, reviewer-auditable):
  CD4       T helper
  CD8A      Cytotoxic T
  CD68      Monocyte / macrophage
  CD19      B cell
  EPCAM     Epithelial
  PECAM1    Endothelial
  COL1A1    Fibroblast
  PDGFRB    Pericyte / SMC
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from anndata import AnnData

from scripts.visualize._plot_utils import ensure_leiden, topn_variable_genes

CANONICAL_MARKERS = [
    "CD4",
    "CD8A",
    "CD68",
    "CD19",
    "EPCAM",
    "PECAM1",
    "COL1A1",
    "PDGFRB",
]


def render_lineage_dotplot(adata: AnnData, out_path: Path) -> None:
    work = adata.copy()
    if "imputed" in work.layers:
        work.X = np.asarray(work.layers["imputed"])

    ensure_leiden(work)
    if "leiden_bio" not in work.obs:
        return

    available = [g for g in CANONICAL_MARKERS if g in work.var_names]
    if not available:
        hvg_idx = topn_variable_genes(work, n=min(8, work.n_vars))
        available = [str(work.var_names[i]) for i in hvg_idx]
    if not available:
        return

    sc.pl.dotplot(work, available, groupby="leiden_bio", show=False)
    fig = plt.gcf()
    fig.set_size_inches(max(6.5, 0.5 * len(available) + 2), 4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
