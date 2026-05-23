"""
Wave 2 — held-out-gene recovery imputation benchmark.

Mask 20 % of HVGs on the input slice, call
``LuminaImputer.enhance(held_out_genes=...)`` so the masked genes are zeroed
at the encoder's input layer, then measure how well the model recovers each
held-out gene's per-cell expression against the original observation
(PCC computed per gene).

Outputs:
  <out_dir>/gene_holdout_recovery.png
    panel A: histogram of per-held-out-gene PCC
    panel B: per-cell-class mean PCC bar (when a class label is present)

NOTE: This is a defensible imputation benchmark on a single modality (RNA → RNA).
It is NOT a proteomics / CODEX cross-modality surrogate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from scipy.stats import pearsonr

from scripts.visualize._plot_utils import (
    pick_label_key,
    stable_categorical_colors,
    to_dense,
    topn_variable_genes,
)


def _per_gene_pcc(observed: np.ndarray, imputed: np.ndarray) -> np.ndarray:
    g = observed.shape[1]
    out = np.full(g, np.nan)
    for i in range(g):
        if observed[:, i].std() < 1e-8 or imputed[:, i].std() < 1e-8:
            continue
        r, _ = pearsonr(observed[:, i], imputed[:, i])
        if not np.isnan(r):
            out[i] = float(r)
    return out


def run_gene_holdout_recovery(
    target_adata: AnnData,
    enhance_fn,
    out_path: Path,
    fraction: float = 0.20,
    seed: int = 0,
) -> Optional[dict]:
    """``enhance_fn`` is a callable ``(adata, held_out_genes) -> enhanced_adata``
    bound to a configured LuminaImputer by the caller."""
    rng = np.random.default_rng(seed)
    n_hvg_target = min(target_adata.n_vars, max(20, int(target_adata.n_vars * 0.4)))
    hvg_idx = topn_variable_genes(target_adata, n=n_hvg_target)
    n_hold = max(5, int(round(fraction * len(hvg_idx))))
    hold_idx = list(rng.choice(hvg_idx, size=n_hold, replace=False))
    hold_genes = [target_adata.var_names[i] for i in hold_idx]

    enhanced = enhance_fn(target_adata, hold_genes)
    if "imputed" not in enhanced.layers:
        print("[warn] enhanced AnnData has no 'imputed' layer; cannot score recovery.")
        return None

    raw = to_dense(target_adata.X)
    imp = np.asarray(enhanced.layers["imputed"])
    if imp.shape[1] != raw.shape[1]:
        print("[warn] imputed gene count mismatches raw; cannot score recovery.")
        return None
    pcc = _per_gene_pcc(raw[:, hold_idx], imp[:, hold_idx])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    valid = pcc[~np.isnan(pcc)]
    axes[0].hist(valid, bins=20, color="#4C72B0", edgecolor="#1f1f1f", alpha=0.85)
    if valid.size:
        axes[0].axvline(float(np.mean(valid)), color="#C44E52", linestyle="--",
                        label=f"mean={float(np.mean(valid)):.3f}")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Per-held-out-gene Pearson")
    axes[0].set_ylabel("Held-out genes")
    axes[0].set_title(f"Holdout recovery (n_held={n_hold} of {n_hvg_target} HVGs, frac={fraction:.0%})")
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)

    label_key = pick_label_key(target_adata, ["cell_class", "annotation", "spatial_cluster", "cancer_type"])
    if label_key is not None:
        labels = target_adata.obs[label_key].astype(str).to_numpy()
        unique = sorted(np.unique(labels).tolist())
        palette = stable_categorical_colors(np.array(unique))
        per_class_pcc = []
        for c in unique:
            mask = (labels == c)
            if mask.sum() < 3:
                per_class_pcc.append(np.nan)
                continue
            pcc_c = _per_gene_pcc(raw[mask][:, hold_idx], imp[mask][:, hold_idx])
            per_class_pcc.append(float(np.nanmean(pcc_c)) if np.isfinite(pcc_c).any() else np.nan)
        bars = axes[1].bar(unique, per_class_pcc, color=[palette[c] for c in unique])
        axes[1].set_ylabel(f"Mean per-gene Pearson")
        axes[1].set_title(f"Per-{label_key} recovery on held-out genes")
        axes[1].grid(axis="y", linestyle=":", alpha=0.4)
        for b, v in zip(bars, per_class_pcc):
            if v is None or np.isnan(v):
                continue
            axes[1].annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
        plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right", fontsize=7)
    else:
        axes[1].text(0.5, 0.5, "No class label\nin .obs", ha="center", va="center")
        axes[1].set_axis_off()

    fig.suptitle("Held-out HVG imputation recovery (single-modality, NOT a proteomics surrogate)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"n_held": int(n_hold), "held_genes": hold_genes, "mean_pcc": float(np.nanmean(pcc))}
