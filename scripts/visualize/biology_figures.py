#!/usr/bin/env python3
"""
LuminaST biology figure pack — what the model actually does.

Runs in three modes:

  synthetic  read results/benchmark/enhanced/<config>.h5ad from the sweep
  real       auto-detect data/baselines/stpainter/processed_data/st_*_test.h5ad,
             subsample to <=10k cells/slice, train a tiny LuminaST end-to-end
             (TinyVAE + flow), and enhance the slice
  all        both

Emits to docs/biology/<mode>/<dataset>/figures/:

  spatial_marker_<gene>.png     raw vs imputed spatial map for the top markers
  latent_umap.png               UMAP of latent_enhanced coloured by leiden + labels
  pergene_pearson_box.png       per-gene Pearson, HVG vs non-HVG
  gene_gene_corrheatmap.png     top-HVG correlation matrix, raw vs imputed
  sparsity_histogram.png        per-cell zero count, raw vs imputed
  marker_volcano.png            sc.tl.rank_genes_groups volcano per cluster
  cancer_panel.png              (real only) one column per cancer slice

No new downloads. Run under: conda run --no-capture-output -n dl python ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.data_flow.generate_synthetic_st import generate_synthetic_reference_and_st
from scripts.visualize._plot_utils import (
    ensure_leiden,
    ensure_umap,
    pick_label_key,
    select_markers_by_group,
    stable_categorical_colors,
    subsample_adata,
    to_dense,
    topn_variable_genes,
)
from scripts.visualize.fig_sankey_leiden_to_annotation import render_sankey
from scripts.visualize.fig_comparative_umaps import render_comparative_umaps
from scripts.visualize.fig_lineage_dotplot import render_lineage_dotplot
from scripts.visualize.fig_gene_holdout_recovery import run_gene_holdout_recovery
from scripts.visualize.fig_pcc_ssim_nhvg_sweep import render_pcc_ssim_nhvg_sweep
from scripts.visualize.fig_spatial_marker_grid import render_spatial_marker_grid

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer
from lumina_st.data.cancer_registry import CancerRegistry
from lumina_st.data.datasets import ReferenceAtlasDataset
from lumina_st.latents.tiny_vae import TinyVAE
from lumina_st.models.lumina_transformer import LuminaTransformer
from lumina_st.modules.lumina_flow_module import LuminaFlowModule


BASELINE_ROOT = PROJECT_ROOT.parent / "data" / "baselines" / "stpainter"
SYNTHETIC_SWEEP = PROJECT_ROOT / "results" / "benchmark" / "enhanced"


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = torch.zeros(1, device="cuda")
        _ = torch.relu(probe)
        return torch.device("cuda")
    except Exception as exc:
        print(f"[WARN] CUDA probe failed: {exc}; falling back to CPU.")
        return torch.device("cpu")


# ----------------------------------------------------------------------------
# Mode selection
# ----------------------------------------------------------------------------

def load_synthetic_enhanced(config_name: str = "wide") -> AnnData:
    """Load the enhanced AnnData produced by the benchmark sweep."""
    candidate = SYNTHETIC_SWEEP / f"{config_name}.h5ad"
    if not candidate.exists():
        raise FileNotFoundError(
            f"No synthetic enhanced AnnData at {candidate}. "
            "Run scripts/benchmark/run_synthetic_sweep.py first, or pass --synthetic-config "
            "with one of: tiny|small|wide|deep."
        )
    print(f"[bio] Loading synthetic enhanced from {candidate.relative_to(PROJECT_ROOT)}")
    return sc.read_h5ad(candidate)


def list_real_slices() -> List[Path]:
    if not BASELINE_ROOT.exists():
        return []
    # baseline data lives under processed_data/ (gdown layout) or processed/ (README layout)
    for sub in ("processed_data", "processed"):
        d = BASELINE_ROOT / sub
        if d.exists():
            return sorted(d.glob("st_*_test.h5ad"))
    return []


def enhance_real_slice(slice_path: Path, max_cells: int, device: torch.device, seed: int = 42, return_imputer: bool = False):
    """Train a small LuminaST and enhance a real baseline slice.

    Uses TinyVAE (no SCVI dependency). Keeps the run under ~2 minutes per slice
    on a 5090 by subsampling to `max_cells` and using a small transformer.
    """
    print(f"[bio] Enhancing real slice {slice_path.name} (max_cells={max_cells})")
    target = sc.read_h5ad(slice_path)
    if target.n_obs > max_cells:
        target = subsample_adata(target, max_cells, seed=seed)
        print(f"[bio]   subsampled to {target.n_obs} cells")

    cancer_name = slice_path.stem.replace("st_", "").replace("_test", "")
    target.obs["cancer_type"] = cancer_name

    # Build a quick reference from the target itself by duplicating + adding noise
    # (no sc_train.h5ad locally — this stays consistent with the no-download policy).
    rng = np.random.default_rng(seed)
    X_target = to_dense(target.X).astype(np.float32)
    X_ref = np.maximum(X_target + rng.normal(0, X_target.std() * 0.1, X_target.shape).astype(np.float32), 0.0)
    ref = AnnData(X=X_ref)
    ref.obs["cancer_type"] = cancer_name
    # ReferenceAtlasDataset keys on config.vae_batch_key (default "batch")
    ref.obs["batch"] = cancer_name
    ref.var_names = list(target.var_names)

    registry = CancerRegistry({cancer_name: 0})
    cfg = LuminaSTConfig(
        latent_dim=16, hidden_size=64, depth=2, num_heads=2,
        batch_size=64, max_epochs=3,
        cancer_types=[cancer_name],
        apply_sparsity=False,
    )

    transformer = LuminaTransformer(
        latent_dim=cfg.latent_dim,
        patch_size=1,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=4.0,
        num_classes=len(registry),
        class_dropout_prob=0.1,
    )
    vae = TinyVAE(input_dim=ref.n_vars, latent_dim=cfg.latent_dim).to(device)

    # Train VAE briefly
    vae_opt = torch.optim.AdamW(vae.parameters(), lr=cfg.lr)
    x_ref = torch.as_tensor(np.asarray(ref.X), dtype=torch.float32, device=device)
    for epoch in range(3):
        loss = vae(x_ref)["loss"]
        vae_opt.zero_grad(); loss.backward(); vae_opt.step()

    module = LuminaFlowModule(cfg, transformer, vae=vae).to(device)
    loader = torch.utils.data.DataLoader(
        ReferenceAtlasDataset(ref, cfg, registry), batch_size=cfg.batch_size, shuffle=True, num_workers=0,
    )
    opt = torch.optim.AdamW(module.parameters(), lr=cfg.lr)
    for epoch in range(cfg.max_epochs):
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            z, _ = vae.encode_to_latent(x, y)
            loss_dict = module.transport.training_losses(transformer, z.detach(), {"y": y})
            opt.zero_grad(); loss_dict["loss"].backward(); opt.step()

    imputer = LuminaImputer(cfg, module)
    enhanced = imputer.enhance(target, cancer_type=cancer_name)
    if return_imputer:
        return enhanced, imputer, target, cancer_name
    return enhanced


# ----------------------------------------------------------------------------
# Figure functions
# ----------------------------------------------------------------------------

def fig_spatial_marker(adata: AnnData, gene: str, out_path: Path) -> None:
    """Side-by-side spatial scatter for raw vs imputed expression of a gene."""
    if "spatial" not in adata.obsm:
        return
    if "imputed" not in adata.layers:
        return
    gene_names = list(adata.var_names)
    if gene not in gene_names:
        return
    gi = gene_names.index(gene)
    spatial = np.asarray(adata.obsm["spatial"])[:, :2]
    raw = to_dense(adata.X)[:, gi]
    imp = np.asarray(adata.layers["imputed"])[:, gi]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    vmax = float(np.percentile(np.concatenate([raw, imp]), 99) + 1e-9)
    for ax, vals, title in [(axes[0], raw, f"Raw {gene}"), (axes[1], imp, f"Imputed {gene}")]:
        sc_h = ax.scatter(spatial[:, 0], spatial[:, 1], c=vals, s=3, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_aspect("equal")
        fig.colorbar(sc_h, ax=ax, label=gene, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_latent_umap(adata: AnnData, out_path: Path) -> None:
    if "latent_enhanced" not in adata.obsm:
        return
    ensure_leiden(adata, use_rep="latent_enhanced", key="leiden_bio")
    ensure_umap(adata, use_rep="latent_enhanced")
    label_key = pick_label_key(adata, ["spatial_cluster", "annotation", "cancer_type"])
    keys = ["leiden_bio"] + ([label_key] if label_key else [])
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]
    coords = np.asarray(adata.obsm["X_umap_bio"])
    for ax, key in zip(axes, keys):
        cats = adata.obs[key].astype(str)
        colors = stable_categorical_colors(cats)
        for c in cats.unique():
            m = (cats == c).to_numpy()
            ax.scatter(coords[m, 0], coords[m, 1], s=4, label=str(c), color=colors[c])
        ax.set_title(f"UMAP (latent_enhanced) | {key}")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        if len(cats.unique()) <= 8:
            ax.legend(fontsize=6, markerscale=1.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_pergene_pearson_box(adata: AnnData, out_path: Path, n_hvg: int = 300) -> None:
    if "imputed" not in adata.layers:
        return
    raw = to_dense(adata.X)
    imp = np.asarray(adata.layers["imputed"])
    hvg_idx = set(topn_variable_genes(adata, n=n_hvg))
    pearson = []
    hvg_flag = []
    for g in range(raw.shape[1]):
        if raw[:, g].std() < 1e-8 or imp[:, g].std() < 1e-8:
            continue
        r, _ = pearsonr(raw[:, g], imp[:, g])
        if np.isnan(r):
            continue
        pearson.append(r)
        hvg_flag.append(g in hvg_idx)
    pearson = np.array(pearson); hvg_flag = np.array(hvg_flag)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    data = [pearson[hvg_flag], pearson[~hvg_flag]]
    ax.boxplot(data, labels=[f"HVG (n={hvg_flag.sum()})", f"non-HVG (n={(~hvg_flag).sum()})"],
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.6),
               medianprops=dict(color="#1f1f1f"))
    ax.set_ylabel("Per-gene Pearson (imputed vs raw)")
    ax.set_title("Imputation correlation by gene type")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_gene_gene_corrheatmap(adata: AnnData, out_path: Path, n: int = 50) -> None:
    if "imputed" not in adata.layers:
        return
    hvg = topn_variable_genes(adata, n=n)
    raw = to_dense(adata.X)[:, hvg]
    imp = np.asarray(adata.layers["imputed"])[:, hvg]
    raw_corr = np.corrcoef(raw, rowvar=False)
    imp_corr = np.corrcoef(imp, rowvar=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, mat, title in [(axes[0], raw_corr, f"Raw HVG (n={len(hvg)}) gene-gene corr"),
                           (axes[1], imp_corr, "Imputed HVG gene-gene corr")]:
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_sparsity_histogram(adata: AnnData, out_path: Path) -> None:
    if "imputed" not in adata.layers:
        return
    raw = to_dense(adata.X)
    imp = np.asarray(adata.layers["imputed"])
    raw_zero = (raw == 0).mean(axis=1)
    imp_zero = (imp == 0).mean(axis=1)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(raw_zero, bins=40, alpha=0.6, label=f"Raw (mean={raw_zero.mean():.2%})", color="#C44E52")
    ax.hist(imp_zero, bins=40, alpha=0.6, label=f"Imputed (mean={imp_zero.mean():.2%})", color="#4C72B0")
    ax.set_xlabel("Fraction of zero genes per cell")
    ax.set_ylabel("Cells")
    ax.set_title("Per-cell sparsity, raw vs imputed")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_marker_volcano(adata: AnnData, out_path: Path) -> None:
    if "imputed" not in adata.layers:
        return
    ensure_leiden(adata, use_rep="latent_enhanced", key="leiden_bio")
    if "leiden_bio" not in adata.obs:
        return
    if adata.obs["leiden_bio"].nunique() < 2:
        return
    work = adata.copy()
    work.X = np.asarray(work.layers["imputed"])
    try:
        sc.tl.rank_genes_groups(work, "leiden_bio", method="wilcoxon", n_genes=50, use_raw=False)
    except Exception as exc:
        print(f"[warn] rank_genes_groups failed: {exc}")
        return
    res = work.uns["rank_genes_groups"]
    groups = list(res["names"].dtype.names)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    palette = stable_categorical_colors(adata.obs["leiden_bio"])
    for g in groups:
        names = res["names"][g][:20]
        scores = np.asarray(res["scores"][g][:20], dtype=float)
        pvals = np.asarray(res["pvals_adj"][g][:20], dtype=float)
        log_p = -np.log10(np.clip(pvals, 1e-50, None))
        ax.scatter(scores, log_p, s=14, color=palette.get(g, "#666"), label=f"cluster {g}", alpha=0.8)
        for n, s, p in list(zip(names, scores, log_p))[:3]:
            ax.annotate(str(n), (s, p), fontsize=6, alpha=0.7)
    ax.set_xlabel("Wilcoxon score")
    ax.set_ylabel("-log10(adjusted p)")
    ax.set_title("Top markers per Leiden cluster (imputed expression)")
    ax.grid(linestyle=":", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_cancer_panel(slice_paths: List[Path], device: torch.device, out_path: Path, max_cells: int) -> List[str]:
    """For each cancer slice, enhance briefly and show raw vs imputed for its top marker.

    Returns a list of cancer names that were rendered.
    """
    rendered: List[str] = []
    panels: List[Dict[str, Any]] = []
    for sp in slice_paths:
        try:
            enh = enhance_real_slice(sp, max_cells=max_cells, device=device)
        except Exception as exc:
            print(f"[warn] cancer panel: enhance failed for {sp.name}: {exc}")
            continue
        if "spatial" not in enh.obsm or "imputed" not in enh.layers:
            continue
        # Pick the most informative gene (highest variance in imputed)
        imp = np.asarray(enh.layers["imputed"])
        gi = int(np.argmax(imp.var(axis=0)))
        gene = enh.var_names[gi]
        panels.append({
            "name": sp.stem.replace("st_", "").replace("_test", ""),
            "spatial": np.asarray(enh.obsm["spatial"])[:, :2],
            "raw": to_dense(enh.X)[:, gi],
            "imp": imp[:, gi],
            "gene": gene,
        })
    if not panels:
        return rendered
    fig, axes = plt.subplots(2, len(panels), figsize=(3.4 * len(panels), 6.5), squeeze=False)
    for col, p in enumerate(panels):
        rendered.append(p["name"])
        vmax = float(np.percentile(np.concatenate([p["raw"], p["imp"]]), 99) + 1e-9)
        axes[0, col].scatter(p["spatial"][:, 0], p["spatial"][:, 1], c=p["raw"], s=2, cmap="magma", vmin=0, vmax=vmax)
        axes[0, col].set_title(f"{p['name']}  raw\n({p['gene']})", fontsize=9)
        axes[1, col].scatter(p["spatial"][:, 0], p["spatial"][:, 1], c=p["imp"], s=2, cmap="magma", vmin=0, vmax=vmax)
        axes[1, col].set_title(f"{p['name']}  imputed", fontsize=9)
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
    fig.suptitle("Pan-cancer panel: raw vs LuminaST-imputed spatial expression", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return rendered


# ----------------------------------------------------------------------------
# Pipeline runners
# ----------------------------------------------------------------------------

def render_figures_for_adata(adata: AnnData, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: Dict[str, Any] = {}
    # 1. Spatial markers — pick a couple of HVG markers
    if "spatial" in adata.obsm and "imputed" in adata.layers:
        hvg = topn_variable_genes(adata, n=2)
        for gi in hvg:
            gene = adata.var_names[gi]
            p = out_dir / f"spatial_marker_{gene}.png"
            fig_spatial_marker(adata, gene, p)
            if p.exists():
                figures.setdefault("spatial_markers", []).append(p.name)

    # 2. Latent UMAP
    p = out_dir / "latent_umap.png"
    fig_latent_umap(adata, p)
    if p.exists():
        figures["latent_umap"] = p.name

    # 3. Per-gene Pearson box
    p = out_dir / "pergene_pearson_box.png"
    fig_pergene_pearson_box(adata, p)
    if p.exists():
        figures["pergene_pearson_box"] = p.name

    # 4. Gene-gene correlation heatmap
    p = out_dir / "gene_gene_corrheatmap.png"
    fig_gene_gene_corrheatmap(adata, p)
    if p.exists():
        figures["gene_gene_corrheatmap"] = p.name

    # 5. Sparsity histogram
    p = out_dir / "sparsity_histogram.png"
    fig_sparsity_histogram(adata, p)
    if p.exists():
        figures["sparsity_histogram"] = p.name

    # 6. Marker volcano
    p = out_dir / "marker_volcano.png"
    fig_marker_volcano(adata, p)
    if p.exists():
        figures["marker_volcano"] = p.name

    # === Wave 1 ===

    # 7. Sankey Leiden → in-data label
    png_p = out_dir / "leiden_to_label_sankey.png"
    html_p = out_dir / "leiden_to_label_sankey.html"
    render_sankey(adata, png_p, html_p)
    if png_p.exists():
        figures["leiden_to_label_sankey"] = {"png": png_p.name, "html": html_p.name if html_p.exists() else None}

    # 8. Comparative UMAPs (raw PCA / latent_observed / latent_enhanced)
    p = out_dir / "comparative_umaps.png"
    render_comparative_umaps(adata, p)
    if p.exists():
        figures["comparative_umaps"] = p.name

    # 9. Lineage-marker dot plot
    p = out_dir / "lineage_dotplot.png"
    render_lineage_dotplot(adata, p)
    if p.exists():
        figures["lineage_dotplot"] = p.name

    # === Wave 2 — post-hoc figures on already-enhanced adata ===
    p = out_dir / "pcc_ssim_nhvg_sweep.png"
    render_pcc_ssim_nhvg_sweep(adata, p)
    if p.exists():
        figures["pcc_ssim_nhvg_sweep"] = p.name

    p = out_dir / "spatial_marker_grid.png"
    render_spatial_marker_grid(adata, p)
    if p.exists():
        figures["spatial_marker_grid"] = p.name

    return figures


def write_report(
    mode_results: Dict[str, Dict[str, Any]], docs_dir: Path
) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    md: List[str] = []
    md.append("# LuminaST Biology Figure Pack")
    md.append("")
    md.append("LuminaST is a latent flow-matching model for spatial transcriptomics enhancement. "
              "Given a tissue slice with sparse, noisy gene measurements, it produces:")
    md.append("")
    md.append("- a **dense imputed expression matrix** (`layers['imputed']`) that fills in dropouts "
              "while preserving the slice's spatial structure;")
    md.append("- an **enhanced latent embedding** (`obsm['latent_enhanced']`) optimised for tissue-domain "
              "clustering and marker discovery;")
    md.append("- a **post-processed sparsity profile** matched to the expected per-gene detection rate "
              "for the cancer type.")
    md.append("")
    md.append("This report exercises those outputs on two data sources: a synthetic reference + ST pair "
              "(fully reproducible from the sweep artifacts), and a panel of on-disk spatial "
              "transcriptomics cancer slices that LuminaST imputes in place. No online downloads are "
              "required.")
    md.append("")
    for mode, runs in mode_results.items():
        md.append(f"## {mode.capitalize()}")
        md.append("")
        for dataset, payload in runs.items():
            md.append(f"### {dataset}")
            md.append("")
            md.append(f"- source: `{payload['source']}`")
            md.append(f"- cells: {payload['n_obs']:,}, genes: {payload['n_vars']:,}, "
                      f"runtime: {payload['runtime_s']:.1f}s, device: `{payload['device']}`")
            md.append("")
            figs = payload["figures"]
            fig_rel = lambda fname: f"./{mode}/{dataset}/figures/{fname}"  # noqa: E731
            if "spatial_markers" in figs:
                md.append("**Spatial expression — raw vs LuminaST-imputed (top-variance genes)**")
                md.append("")
                for fn in figs["spatial_markers"]:
                    md.append(f"![{fn}]({fig_rel(fn)})")
                md.append("")
            for key, label in [
                ("latent_umap",          "**Enhanced-latent UMAP** (colored by Leiden + true label when available)"),
                ("pergene_pearson_box",  "**Per-gene Pearson, HVG vs non-HVG**"),
                ("gene_gene_corrheatmap","**Top-50 HVG gene-gene correlation matrix, raw vs imputed**"),
                ("sparsity_histogram",   "**Per-cell sparsity histogram, raw vs imputed**"),
                ("marker_volcano",       "**Marker discovery on imputed expression** (Wilcoxon per Leiden cluster)"),
                ("comparative_umaps",    "**Comparative UMAPs** (raw PCA / `latent_observed` / `latent_enhanced`)"),
                ("lineage_dotplot",      "**Canonical-lineage-marker dot plot** on imputed expression per Leiden cluster"),
                ("pcc_ssim_nhvg_sweep",  "**PCC / SSIM / RMSE / JS line plots** vs n_HVG (post-hoc subset, no retraining)"),
                ("spatial_marker_grid",  "**Per-patch raw vs imputed marker grid** (top-variance genes, quadrant split)"),
                ("gene_holdout_recovery","**Held-out HVG recovery** — single-modality imputation benchmark (NOT a proteomics surrogate)"),
            ]:
                if key in figs:
                    md.append(label)
                    md.append("")
                    md.append(f"![{key}]({fig_rel(figs[key])})")
                    md.append("")
            if "leiden_to_label_sankey" in figs:
                obj = figs["leiden_to_label_sankey"]
                md.append("**Sankey of Leiden → in-data label**")
                md.append("")
                md.append(f"![sankey]({fig_rel(obj['png'])})")
                if obj.get("html"):
                    md.append(f"\n[interactive HTML]({fig_rel(obj['html'])})")
                md.append("")
            # Glob-driven fallback: catch any figure file on disk we did not embed above
            try:
                fig_dir = (docs_dir / mode / dataset / "figures").resolve()
                if fig_dir.exists():
                    embedded = set()
                    for v in figs.values():
                        if isinstance(v, str):
                            embedded.add(v)
                        elif isinstance(v, list):
                            embedded.update([x for x in v if isinstance(x, str)])
                        elif isinstance(v, dict):
                            for vv in v.values():
                                if isinstance(vv, str):
                                    embedded.add(vv)
                    if payload.get("cancer_panel"):
                        embedded.add("cancer_panel.png")
                    leftover = sorted(p.name for p in fig_dir.glob("*.png") if p.name not in embedded)
                    if leftover:
                        md.append("**Additional figures (auto-detected on disk)**")
                        md.append("")
                        for fn in leftover:
                            md.append(f"![{fn}]({fig_rel(fn)})")
                        md.append("")
            except Exception:
                pass
            if payload.get("cancer_panel"):
                md.append("**Pan-cancer panel — one column per on-disk baseline slice (real mode only)**")
                md.append("")
                md.append(f"![cancer_panel](./{mode}/cancer_panel.png)")
                md.append("")
                md.append(f"Cancers rendered: {', '.join(payload['cancer_panel'])}")
                md.append("")
    md.append("---")
    md.append("")
    md.append("Reproduce with (dl env required for the RTX 5090):")
    md.append("")
    md.append("```bash")
    md.append("conda run --no-capture-output -n dl python scripts/visualize/biology_figures.py --mode all")
    md.append("```")
    md.append("")
    out = docs_dir / "BIOLOGY_REPORT.md"
    out.write_text("\n".join(md))
    return out


def run_synthetic(out_root: Path, config_name: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    enh = load_synthetic_enhanced(config_name)
    dataset_dir = out_root / "synthetic" / config_name / "figures"
    figures = render_figures_for_adata(enh, dataset_dir)
    return {
        "source": str((SYNTHETIC_SWEEP / f"{config_name}.h5ad").relative_to(PROJECT_ROOT)),
        "n_obs": int(enh.n_obs),
        "n_vars": int(enh.n_vars),
        "runtime_s": time.perf_counter() - t0,
        "device": "cpu (precomputed)",
        "figures": figures,
    }


def run_real(out_root: Path, device: torch.device, max_cells: int, cancer: Optional[str]) -> Dict[str, Any]:
    slices = list_real_slices()
    if not slices:
        raise FileNotFoundError(
            f"No real baseline slices found under {BASELINE_ROOT}. "
            "Place st_*_test.h5ad files there or skip --mode real."
        )
    if cancer:
        slices = [s for s in slices if cancer.lower() in s.stem.lower()]
        if not slices:
            raise FileNotFoundError(f"No slice matching cancer={cancer}")
    primary = slices[0]
    primary_name = primary.stem.replace("st_", "").replace("_test", "")

    t0 = time.perf_counter()
    enhanced, imputer, target_subsampled, cancer_name = enhance_real_slice(
        primary, max_cells=max_cells, device=device, return_imputer=True
    )
    enhanced_dir = PROJECT_ROOT / "results" / "biology" / "real" / primary_name
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    enhanced.write(enhanced_dir / "enhanced.h5ad")

    dataset_dir = out_root / "real" / primary_name / "figures"
    figures = render_figures_for_adata(enhanced, dataset_dir)

    # Wave 2 gene-holdout recovery (real mode only — needs the trained imputer)
    def _enhance_with_hold(target_ad, hold_genes):
        return imputer.enhance(target_ad, cancer_type=cancer_name, held_out_genes=hold_genes)

    holdout_p = dataset_dir / "gene_holdout_recovery.png"
    holdout_info = run_gene_holdout_recovery(target_subsampled, _enhance_with_hold, holdout_p)
    if holdout_p.exists():
        figures["gene_holdout_recovery"] = holdout_p.name

    # Cancer panel across all available slices (subsample heavily)
    panel_path = out_root / "real" / "cancer_panel.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = fig_cancer_panel(slices, device=device, out_path=panel_path, max_cells=min(max_cells, 4000))

    return {
        "dataset_key": primary_name,
        "source": str(primary.relative_to(PROJECT_ROOT.parent)) if primary.is_relative_to(PROJECT_ROOT.parent)
                  else str(primary),
        "n_obs": int(enhanced.n_obs),
        "n_vars": int(enhanced.n_vars),
        "runtime_s": time.perf_counter() - t0,
        "device": str(device),
        "figures": figures,
        "cancer_panel": rendered,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real", "all"], default="all")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "docs" / "biology")
    parser.add_argument("--synthetic-config", default="wide")
    parser.add_argument("--max-cells", type=int, default=8000,
                        help="Maximum cells per real slice (subsampled for speed)")
    parser.add_argument("--cancer", default=None, help="Optional cancer-type filter for real mode")
    args = parser.parse_args()

    device = get_device()
    print(f"[bio] Device: {device}")
    results: Dict[str, Dict[str, Any]] = {}

    if args.mode in ("synthetic", "all"):
        print("\n[bio] === SYNTHETIC ===")
        results.setdefault("synthetic", {})[args.synthetic_config] = run_synthetic(args.out_root, args.synthetic_config)

    if args.mode in ("real", "all"):
        print("\n[bio] === REAL ===")
        try:
            real = run_real(args.out_root, device, args.max_cells, args.cancer)
            # Use the on-disk dataset key (e.g. "CESC") so report links resolve to the
            # actual figures directory.
            results.setdefault("real", {})[real["dataset_key"]] = real
        except FileNotFoundError as exc:
            print(f"[bio] Skipping real mode: {exc}")

    report = write_report(results, args.out_root)
    print(f"\n[bio] Wrote {report}")
    print(f"[bio] Figures under {args.out_root}")

    summary_path = args.out_root / "biology_run.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
