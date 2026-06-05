"""
EnhancementEvaluator for LuminaST.

Provides quantitative metrics to evaluate how much the imputation / enhancement
improved the data (compared to ground truth when available, or via self-supervised
metrics when no ground truth exists).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

# Label columns the evaluator DERIVES from the latents it is scoring. A GT key
# pointing at any of these would score the enhanced latent against a clustering
# computed from a latent (self-agreement / circular comparison), which the
# uplift gate forbids (lumina-st #308).
_DERIVED_LABEL_KEYS = frozenset(
    {"leiden", "leiden_enhanced", "leiden_raw", "leiden_observed"}
)

# Clustering-uplift scorers, each called as scorer(labels_true, labels_pred).
# ARI/AMI/NMI are symmetric; homogeneity is asymmetric so GT must be the first
# argument (homogeneity_score(labels_true, labels_pred)).
_CLUSTERING_SCORERS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "ari": adjusted_rand_score,
    "ami": adjusted_mutual_info_score,
    "nmi": normalized_mutual_info_score,
    "homogeneity": homogeneity_score,
}


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

    def _leiden_labels(self, use_rep: Optional[str], n_neighbors: int, key_added: str):
        """Run Leiden on the graph of ``use_rep`` (or PCA of ``.X`` if None)."""
        import scanpy as sc

        if use_rep is not None and use_rep in self.adata.obsm:
            sc.pp.neighbors(self.adata, use_rep=use_rep, n_neighbors=n_neighbors)
        else:
            n_comps = min(50, max(self.adata.n_vars - 1, 2))
            sc.pp.pca(self.adata, n_comps=n_comps)
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors)
        sc.tl.leiden(self.adata, key_added=key_added, resolution=1.0)
        return np.asarray(self.adata.obs[key_added])

    def run_clustering_uplift_metrics(
        self,
        gt_key: Optional[str] = None,
        gt_class_a: bool = False,
        latent_key: str = "latent_enhanced",
        baseline_latent_key: str = "latent_observed",
        n_neighbors: int = 15,
    ) -> Dict[str, object]:
        """Gated clustering-uplift: enhanced-vs-GT and raw-vs-GT, signed delta.

        Clustering-uplift metrics (ARI/AMI/NMI/homogeneity) are only meaningful
        against an INDEPENDENT, high-quality ("class-A") ground-truth label
        column in ``adata.obs``. This method enforces three rules from
        lumina-st #308:

        1. **Class-A GT gating.** If ``gt_key`` is missing/absent from
           ``adata.obs`` or ``gt_class_a`` is False, every uplift metric is
           reported as ``None`` (ungated) rather than silently computing a
           number. The caller learns *why* via ``clustering_uplift_gate``.
        2. **Signed delta-over-raw.** For each metric the enhanced latent and
           the raw/observed latent are EACH scored against the same
           independent GT, and a signed ``<metric>_delta_over_raw`` column is
           reported so a negative uplift is visible, not buried.
        3. **No circular agreement.** The enhanced latent is never scored
           against a clustering derived from a latent. Requesting ``gt_key``
           that names a derived Leiden column raises ``ValueError``.

        Args:
            gt_key: column in ``adata.obs`` holding the independent GT labels.
            gt_class_a: explicit operator assertion that ``gt_key`` is an
                independent, high-quality annotation. Must be True to gate ON.
            latent_key: ``.obsm`` key for the enhanced latent.
            baseline_latent_key: ``.obsm`` key for the raw/observed latent.
            n_neighbors: neighbor count for the Leiden graphs.

        Returns:
            Dict whose keys are always present. When gated OFF every metric
            value is ``None``. When gated ON the dict carries
            ``<m>_enhanced_vs_gt``, ``<m>_raw_vs_gt`` and
            ``<m>_delta_over_raw`` (signed) for each of ARI/AMI/NMI/homogeneity.
            ``clustering_uplift_gate`` records ``"class-A"`` or the reason it
            was withheld.
        """
        # Rule 3: forbid circular comparison up front, regardless of gating.
        if gt_key is not None and gt_key in _DERIVED_LABEL_KEYS:
            raise ValueError(
                f"Circular clustering comparison: gt_key={gt_key!r} names a "
                f"clustering DERIVED from a latent. Uplift must be scored "
                f"against an independent ground truth, not a self-derived "
                f"partition (lumina-st #308)."
            )

        metric_keys = []
        for m in _CLUSTERING_SCORERS:
            metric_keys += [f"{m}_enhanced_vs_gt", f"{m}_raw_vs_gt", f"{m}_delta_over_raw"]

        def _ungated(reason: str) -> Dict[str, object]:
            out: Dict[str, object] = {k: None for k in metric_keys}
            out["clustering_uplift_gate"] = reason
            return out

        # Rule 1: class-A GT gating.
        if gt_key is None:
            return _ungated("ungated: no gt_key provided")
        if gt_key not in self.adata.obs:
            return _ungated(f"ungated: gt_key={gt_key!r} absent from adata.obs")
        if not gt_class_a:
            return _ungated(
                f"ungated: gt_key={gt_key!r} not asserted class-A (gt_class_a=False)"
            )

        gt = np.asarray(self.adata.obs[gt_key])

        # Score enhanced and raw latents EACH against the same independent GT.
        # Raw first so its neighbor graph does not contaminate the enhanced run.
        raw_labels = self._leiden_labels(baseline_latent_key, n_neighbors, "leiden_raw")
        enh_labels = self._leiden_labels(latent_key, n_neighbors, "leiden_enhanced")

        out: Dict[str, object] = {}
        for m, scorer in _CLUSTERING_SCORERS.items():
            enh = float(scorer(gt, enh_labels))
            raw = float(scorer(gt, raw_labels))
            out[f"{m}_enhanced_vs_gt"] = enh
            out[f"{m}_raw_vs_gt"] = raw
            out[f"{m}_delta_over_raw"] = enh - raw  # signed: negative = regression
        out["clustering_uplift_gate"] = "class-A"
        return out

    def summary(
        self,
        gt_key: Optional[str] = None,
        gt_class_a: bool = False,
    ) -> Dict[str, object]:
        """Full metric battery.

        Gene metrics always run. Clustering-uplift metrics run through the
        gated, GT-anchored path (``run_clustering_uplift_metrics``): without a
        class-A ``gt_key`` they are reported as ``None`` rather than a
        self-derived agreement number (lumina-st #308).
        """
        metrics: Dict[str, object] = {}
        metrics.update(self.run_gene_metrics())
        try:
            metrics.update(
                self.run_clustering_uplift_metrics(gt_key=gt_key, gt_class_a=gt_class_a)
            )
        except Exception as e:
            metrics["clustering_error"] = str(e)
        return metrics
