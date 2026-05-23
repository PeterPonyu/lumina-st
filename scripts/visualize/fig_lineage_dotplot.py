"""
Wave 1 — canonical-lineage-marker dot plot on imputed expression per Leiden cluster.

Markers (curated, reviewer-auditable):
  CD3D   CD3E   CD4   CD8A          T cells
  CD19   CD79A  MS4A1                B cells
  CD14   CD68   LYZ                  Monocytes / macrophages
  KRT8   KRT18  EPCAM  KRT19         Epithelial
  PECAM1 VWF    CDH5                 Endothelial
  COL1A1 PDGFRB ACTA2                Fibroblasts / SMCs
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from anndata import AnnData

from scripts.visualize._plot_utils import ensure_leiden


CANONICAL_LINEAGE_MARKERS = [
    "CD3D", "CD3E", "CD4", "CD8A",
    "CD19", "CD79A", "MS4A1",
    "CD14", "CD68", "LYZ",
    "KRT8", "KRT18", "EPCAM", "KRT19",
    "PECAM1", "VWF", "CDH5",
    "COL1A1", "PDGFRB", "ACTA2",
]


def render_lineage_dotplot(adata: AnnData, out_path: Path) -> None:
    if "imputed" not in adata.layers:
        return
    ensure_leiden(adata, use_rep="latent_enhanced", key="leiden_bio")
    if "leiden_bio" not in adata.obs:
        return

    available = [g for g in CANONICAL_LINEAGE_MARKERS if g in adata.var_names]
    if not available:
        # synthetic data uses Gene_xxxx names — fall back to top HVGs
        from scripts.visualize._plot_utils import topn_variable_genes
        idxs = topn_variable_genes(adata, n=min(20, adata.n_vars))
        available = [adata.var_names[i] for i in idxs]

    work = adata.copy()
    work.X = np.asarray(work.layers["imputed"])
    try:
        sc.tl.dendrogram(work, groupby="leiden_bio")
    except Exception:
        pass
    try:
        sc.pl.dotplot(
            work, var_names=available, groupby="leiden_bio",
            standard_scale="var", show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(max(6.5, 0.35 * len(available) + 2), 3.5)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] lineage dot plot failed: {exc}")
