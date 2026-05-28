"""
EnhancementEvaluator for LuminaST.

Provides quantitative metrics to evaluate how much the imputation / enhancement
improved the data (compared to ground truth when available, or via self-supervised
metrics when no ground truth exists).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class EnhancementEvaluator:
    """Compute a battery of enhancement quality metrics."""

    def __init__(self, adata: AnnData, gt_layer: Optional[str] = None):
        self.adata = adata
        self.gt_layer = gt_layer

    def _reference(self):
        """Return the reference matrix for correlation scoring.

        When ``gt_layer`` is set and present, scores are computed against
        ``adata.layers[gt_layer]`` (ground truth). Otherwise falls back to
        ``adata.X`` (self-consistency baseline).
        """
        if self.gt_layer is not None:
            if self.gt_layer not in self.adata.layers:
                raise KeyError(
                    f"gt_layer={self.gt_layer!r} requested but not present in adata.layers; "
                    f"available: {list(self.adata.layers.keys())}"
                )
            ref = self.adata.layers[self.gt_layer]
        else:
            ref = self.adata.X
        return ref.toarray() if hasattr(ref, "toarray") else ref

    def run_gene_metrics(self) -> Dict[str, float]:
        """Pearson / Spearman correlation between imputed and reference.

        Reference is ``adata.layers[gt_layer]`` when ``gt_layer`` is provided,
        else ``adata.X``. Passing ``gt_layer`` is required for meaningful
        held-out-gene evaluation; without it, the metric measures
        imputed-vs-observed self-consistency only.
        """
        if "imputed" not in self.adata.layers:
            return {"error": "No 'imputed' layer found"}

        imputed = self.adata.layers["imputed"]
        reference = self._reference()

        # Simple per-gene correlation (mean across genes)
        p = []
        s = []
        for g in range(imputed.shape[1]):
            p.append(pearsonr(imputed[:, g], reference[:, g])[0])
            s.append(spearmanr(imputed[:, g], reference[:, g])[0])

        return {
            "mean_pearson": float(np.nanmean(p)),
            "mean_spearman": float(np.nanmean(s)),
        }

    def run_clustering_metrics(self, latent_key: str = "latent_enhanced", n_neighbors: int = 15) -> Dict[str, float]:
        """Clustering quality on the enhanced latent vs original."""
        import scanpy as sc

        # Leiden on enhanced latent
        sc.pp.neighbors(self.adata, use_rep=latent_key, n_neighbors=n_neighbors)
        sc.tl.leiden(self.adata, key_added="leiden_enhanced", resolution=1.0)

        if "leiden" not in self.adata.obs:
            sc.tl.leiden(self.adata, key_added="leiden", resolution=1.0)

        ari = adjusted_rand_score(self.adata.obs["leiden"], self.adata.obs["leiden_enhanced"])
        nmi = normalized_mutual_info_score(self.adata.obs["leiden"], self.adata.obs["leiden_enhanced"])

        return {"ari_enhanced_vs_original": float(ari), "nmi_enhanced_vs_original": float(nmi)}

    def summary(self) -> Dict[str, float]:
        metrics = {}
        metrics.update(self.run_gene_metrics())
        try:
            metrics.update(self.run_clustering_metrics())
        except Exception as e:
            metrics["clustering_error"] = str(e)
        return metrics
