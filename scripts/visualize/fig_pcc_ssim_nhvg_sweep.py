"""
Wave 2 — line-plot sweep of imputation quality metrics over n_HVG.

For each n_HVG in {10, 20, 50, 100, 200, 300} compute per-gene PCC / SSIM /
RMSE / JS on the imputed-vs-raw comparison restricted to the top-N HVGs.

NO retraining: this is a post-hoc subset of a single enhanced AnnData.

Outputs:
  <out_dir>/pcc_ssim_nhvg_sweep.png  (2x2 grid)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr

from scripts.visualize._plot_utils import to_dense, topn_variable_genes


DEFAULT_N_HVGS: List[int] = [10, 20, 50, 100, 200, 300]


def _ssim_per_gene(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # 1-D structural-similarity surrogate: combines luminance + contrast +
    # structure terms (no spatial windowing — we operate on the cell axis).
    n_g = a.shape[1]
    out = np.full(n_g, np.nan)
    C1 = 1e-4
    C2 = 1e-4
    for i in range(n_g):
        x = a[:, i]
        y = b[:, i]
        mux = float(x.mean())
        muy = float(y.mean())
        vx = float(x.var())
        vy = float(y.var())
        cxy = float(((x - mux) * (y - muy)).mean())
        num = (2 * mux * muy + C1) * (2 * cxy + C2)
        den = (mux**2 + muy**2 + C1) * (vx + vy + C2)
        if den > 0:
            out[i] = num / den
    return out


def _js_per_gene(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n_g = a.shape[1]
    out = np.full(n_g, np.nan)
    eps = 1e-12
    for i in range(n_g):
        x = np.clip(a[:, i], 0, None)
        y = np.clip(b[:, i], 0, None)
        sx = x.sum() + eps
        sy = y.sum() + eps
        p = x / sx
        q = y / sy
        m = 0.5 * (p + q)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_pm = np.where(p > 0, p * (np.log(p + eps) - np.log(m + eps)), 0).sum()
            kl_qm = np.where(q > 0, q * (np.log(q + eps) - np.log(m + eps)), 0).sum()
        out[i] = 0.5 * (kl_pm + kl_qm)
    return out


def render_pcc_ssim_nhvg_sweep(
    adata: AnnData, out_path: Path, n_hvgs: List[int] | None = None
) -> None:
    if "imputed" not in adata.layers:
        return
    raw = to_dense(adata.X).astype(np.float32)
    imp = np.asarray(adata.layers["imputed"], dtype=np.float32)
    n_hvgs = n_hvgs or DEFAULT_N_HVGS

    # Pre-sort genes by variance once
    hvg_order = topn_variable_genes(adata, n=adata.n_vars)
    metrics: dict = {"pcc": [], "spr": [], "ssim": [], "rmse": [], "js": []}
    used_n: List[int] = []
    for n in n_hvgs:
        if n > len(hvg_order):
            break
        idx = hvg_order[:n]
        r = raw[:, idx]
        m = imp[:, idx]
        pcc_per = np.array(
            [
                pearsonr(r[:, i], m[:, i])[0]
                if r[:, i].std() > 1e-8 and m[:, i].std() > 1e-8
                else np.nan
                for i in range(n)
            ]
        )
        spr_per = np.array(
            [
                spearmanr(r[:, i], m[:, i])[0]
                if r[:, i].std() > 1e-8 and m[:, i].std() > 1e-8
                else np.nan
                for i in range(n)
            ]
        )
        ssim_per = _ssim_per_gene(r, m)
        rmse_per = np.sqrt(((r - m) ** 2).mean(axis=0))
        js_per = _js_per_gene(r, m)
        used_n.append(n)
        metrics["pcc"].append(float(np.nanmean(pcc_per)))
        metrics["spr"].append(float(np.nanmean(spr_per)))
        metrics["ssim"].append(float(np.nanmean(ssim_per)))
        metrics["rmse"].append(float(np.nanmean(rmse_per)))
        metrics["js"].append(float(np.nanmean(js_per)))

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    plot_specs = [
        (axes[0, 0], "pcc", "Pearson (↑)", "#4C72B0"),
        (axes[0, 1], "ssim", "SSIM (↑)", "#55A868"),
        (axes[1, 0], "rmse", "RMSE (↓)", "#C44E52"),
        (axes[1, 1], "js", "Jensen-Shannon (↓)", "#8172B3"),
    ]
    for ax, key, label, color in plot_specs:
        ax.plot(used_n, metrics[key], marker="o", color=color, linewidth=1.6)
        ax.set_xlabel("n_HVG")
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.set_xticks(used_n, [str(n) for n in used_n])
        ax.grid(linestyle=":", alpha=0.4)
        for n_v, v in zip(used_n, metrics[key]):
            ax.annotate(f"{v:.3g}", (n_v, v), fontsize=7, alpha=0.75)
    fig.suptitle("Imputation quality vs n_HVG (single enhanced.h5ad, post-hoc subset)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Thin wrapper — matches the canonical render_pcc_ssim_sweep entry point
# ---------------------------------------------------------------------------
def render_pcc_ssim_sweep(adata: AnnData, out_path: Path) -> None:
    """Post-hoc HVG sweep on a single enhancement — no per-n_HVG retraining."""
    render_pcc_ssim_nhvg_sweep(adata, out_path)
