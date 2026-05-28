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

    def run_gene_metrics(self) -> Dict[str, float]:
        """Pearson / Spearman correlation between imputed and observed (or ground truth)."""
        if "imputed" not in self.adata.layers:
            return {"error": "No 'imputed' layer found"}

        imputed = self.adata.layers["imputed"]
        observed = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else self.adata.X

        # Simple per-gene correlation (mean across genes)
        p = []
        s = []
        for g in range(imputed.shape[1]):
            p.append(pearsonr(imputed[:, g], observed[:, g])[0])
            s.append(spearmanr(imputed[:, g], observed[:, g])[0])

        return {
            "mean_pearson": float(np.nanmean(p)),
            "mean_spearman": float(np.nanmean(s)),
        }

    def run_clustering_metrics(
        self,
        latent_key: str = "latent_enhanced",
        baseline_latent_key: str = "latent_observed",
        n_neighbors: int = 15,
    ) -> Dict[str, float]:
        """Clustering quality on the enhanced latent vs an independent baseline.

        Builds the baseline Leiden partition from a SEPARATE neighbor graph
        (``baseline_latent_key`` in ``.obsm`` if present, else PCA of ``.X``)
        so the comparison against ``leiden_enhanced`` reflects real partition
        change. The previous implementation built one graph from
        ``latent_enhanced`` and ran Leiden on it twice, which forced
        ARI/NMI ≡ 1.0 by construction (lumina-st #132).
        """
        import scanpy as sc

        # Baseline Leiden on an INDEPENDENT graph (observed latent or raw .X PCA).
        # Skip if the caller already supplied a 'leiden' label column.
        if "leiden" not in self.adata.obs:
            if baseline_latent_key in self.adata.obsm:
                sc.pp.neighbors(
                    self.adata, use_rep=baseline_latent_key, n_neighbors=n_neighbors
                )
            else:
                # Fall back to PCA of raw .X so we still compare against an
                # independent representation, not the enhanced graph itself.
                n_comps = min(50, max(self.adata.n_vars - 1, 2))
                sc.pp.pca(self.adata, n_comps=n_comps)
                sc.pp.neighbors(self.adata, n_neighbors=n_neighbors)
            sc.tl.leiden(self.adata, key_added="leiden", resolution=1.0)

        # Enhanced Leiden on the enhanced-latent graph. This overwrites .obsp,
        # but the baseline labels are already materialized in .obs.
        sc.pp.neighbors(self.adata, use_rep=latent_key, n_neighbors=n_neighbors)
        sc.tl.leiden(self.adata, key_added="leiden_enhanced", resolution=1.0)

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
