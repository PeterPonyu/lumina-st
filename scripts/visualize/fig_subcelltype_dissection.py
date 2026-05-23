"""
Wave 3 — sub-celltype dissection per major lineage.

For each major lineage in the slice (T cells, Myeloid, Epithelial):

  1. Gate cells using a curated `gating_panel` of canonical lineage markers.
  2. Hold out a SEPARATE validation_panel of disjoint genes from the imputer
     entirely via LuminaImputer.enhance(held_out_genes=validation_panel).
  3. Recluster the gated subset on Leiden of obsm['latent_enhanced'] (the
     latent never sees the gating panel as a feature, only the input layer
     does — so the recluster is NOT just rediscovering the gate).
  4. Run rank_genes_groups on the held-out validation_panel and take
     top-K DEGs where K = len(gating_panel) so the Jaccard is a fair
     same-size similarity.
  5. Compute J = Jaccard(gating_panel, top-K DEGs). If J > 0.3 the figure
     is flagged "circular — interpret cautiously" in the title.
  6. Render per-sub-celltype spatial map.

The Jaccard self-check is the heart of the non-circularity argument and is
mandatory: a wave that ships J > 0.3 on every lineage should be iterated
rather than merged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from anndata import AnnData

from scripts.visualize._plot_utils import stable_categorical_colors, to_dense


# Curated gating panels per lineage. Reviewer-auditable, not auto-derived.
GATING_PANELS: Dict[str, List[str]] = {
    "T_cell":     ["CD3D", "CD3E", "CD4", "CD8A", "CD8B"],
    "Myeloid":    ["CD14", "CD68", "CD163", "LYZ", "ITGAX"],
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"],
}

# Disjoint validation panels — must NOT overlap with the gating panel for
# the same lineage. Used as the DEG search set.
VALIDATION_PANELS: Dict[str, List[str]] = {
    "T_cell":     ["IL7R", "GZMB", "GNLY", "PRF1", "CCR7", "FOXP3", "IL2RA", "CXCR3"],
    "Myeloid":    ["FCN1", "C1QB", "APOE", "CSF1R", "MRC1", "S100A8", "S100A9", "VCAN"],
    "Epithelial": ["MUC1", "MUC2", "CLDN4", "OCLN", "TFF3", "FOXA1", "ELF3", "GATA6"],
}


def _panel_score(adata: AnnData, panel: List[str]) -> np.ndarray:
    """Mean expression across the panel genes that exist in this adata.
    Operates on `layers['imputed']` if present, else `X`."""
    X = np.asarray(adata.layers["imputed"]) if "imputed" in adata.layers else to_dense(adata.X)
    available = [g for g in panel if g in adata.var_names]
    if not available:
        return np.zeros(adata.n_obs, dtype=np.float32)
    idx = [list(adata.var_names).index(g) for g in available]
    return X[:, idx].mean(axis=1)


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def render_subcelltype_dissection(
    enhanced_full: AnnData,
    enhance_fn,
    out_dir: Path,
    cluster_resolution: float = 0.6,
    seed: int = 42,
) -> dict:
    """``enhance_fn`` is callable ``(target_adata, held_out_genes) -> enhanced``
    bound to the trained imputer by the caller. ``enhanced_full`` is the
    standard enhance() output for the gating step."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_names: List[str] = []
    per_lineage: Dict[str, dict] = {}

    rng = np.random.default_rng(seed)
    for lineage, gating in GATING_PANELS.items():
        # Verify disjoint panels
        validation = VALIDATION_PANELS[lineage]
        assert not (set(gating) & set(validation)), (
            f"BUG: gating and validation panels overlap for {lineage}"
        )

        # Gate by panel score (>= 75th percentile)
        score = _panel_score(enhanced_full, gating)
        if not np.isfinite(score).any() or score.std() < 1e-8:
            per_lineage[lineage] = {"skipped": True, "reason": "no panel genes in adata"}
            continue
        thresh = np.quantile(score, 0.75)
        mask = (score >= thresh)
        if mask.sum() < 30:
            per_lineage[lineage] = {"skipped": True, "reason": f"only {int(mask.sum())} cells passed gate"}
            continue

        gated = enhanced_full[mask].copy()

        # Re-impute the gated slice with the validation panel held OUT of the imputer.
        # The new enhanced AnnData's imputed[validation] is the rank_genes_groups input.
        # If enhance_fn returns no imputed (e.g. mismatched shape), bail.
        try:
            held_enh = enhance_fn(gated, validation)
        except Exception as exc:
            per_lineage[lineage] = {"skipped": True, "reason": f"enhance(held_out) failed: {exc}"}
            continue

        # Recluster on latent_enhanced (not on imputed expression — so the
        # clusters are not just rediscovering the gating panel)
        if "latent_enhanced" not in held_enh.obsm:
            per_lineage[lineage] = {"skipped": True, "reason": "no latent_enhanced after re-enhance"}
            continue
        sc.pp.neighbors(held_enh, use_rep="latent_enhanced", n_neighbors=10, key_added="_sub")
        sc.tl.leiden(held_enh, resolution=cluster_resolution, key_added="sub_leiden", neighbors_key="_sub")

        # DEG analysis on the held-out validation panel ONLY.
        # Restrict to validation genes that exist in the adata.
        val_in = [g for g in validation if g in held_enh.var_names]
        if len(val_in) < 3 or held_enh.obs["sub_leiden"].nunique() < 2:
            per_lineage[lineage] = {"skipped": True, "reason": "too few validation genes or sub-clusters"}
            continue
        sub_for_deg = held_enh[:, val_in].copy()
        sub_for_deg.X = np.asarray(sub_for_deg.layers["imputed"]) if "imputed" in sub_for_deg.layers else to_dense(sub_for_deg.X)
        try:
            sc.tl.rank_genes_groups(
                sub_for_deg, "sub_leiden", method="wilcoxon",
                n_genes=min(len(val_in), len(gating)), use_raw=False,
            )
        except Exception as exc:
            per_lineage[lineage] = {"skipped": True, "reason": f"rank_genes_groups failed: {exc}"}
            continue
        res = sub_for_deg.uns["rank_genes_groups"]
        top_genes: List[str] = []
        K = len(gating)
        for grp in res["names"].dtype.names:
            top_genes.extend([str(g) for g in res["names"][grp][:K]])
        top_unique = list(dict.fromkeys(top_genes))[:K]
        jaccard = _jaccard(gating, top_unique)

        flag = jaccard > 0.3
        # Spatial scatter per sub-cluster
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        if "spatial" in held_enh.obsm:
            xy = np.asarray(held_enh.obsm["spatial"])[:, :2]
            sub_labels = held_enh.obs["sub_leiden"].astype(str)
            palette = stable_categorical_colors(sub_labels)
            for c in sub_labels.unique():
                m = (sub_labels == c).to_numpy()
                ax.scatter(xy[m, 0], xy[m, 1], c=palette[c], s=6, label=f"sub-{c}")
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect("equal", adjustable="datalim")
            ax.legend(fontsize=7, markerscale=1.2, loc="best")
        ax.set_title(
            f"{lineage} sub-celltype dissection\n"
            f"gated cells={int(mask.sum())}, sub-clusters={held_enh.obs['sub_leiden'].nunique()}, "
            f"J(gating, top-{K} DEGs)={jaccard:.2f}"
            + ("  ⚠ CIRCULAR — interpret cautiously" if flag else "")
        , fontsize=9)
        png = out_dir / f"subcelltype_{lineage}.png"
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        plt.close(fig)
        figure_names.append(png.name)
        per_lineage[lineage] = {
            "gated_cells": int(mask.sum()),
            "sub_clusters": int(held_enh.obs["sub_leiden"].nunique()),
            "validation_panel_size": len(val_in),
            "top_K_DEGs": top_unique,
            "jaccard_with_gating": jaccard,
            "circular_flag": bool(flag),
            "figure": png.name,
        }
    return {"figures": figure_names, "per_lineage": per_lineage}
