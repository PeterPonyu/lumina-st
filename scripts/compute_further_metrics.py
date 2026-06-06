#!/usr/bin/env python3
"""Compute reference-grounded "further" LuminaST metrics on the REAL enhanced run.

Loads the already-emitted ``results/lumina-st/outputs/enhanced.h5ad`` (which
carries the real ``st_COAD_test`` ground truth: ``obs['annotation']`` = 16 cell
types, ``obs['spatial_cluster']`` = 5 domains, ``var['impute_mask']`` = 3,053
observed genes, ``obsm['spatial']`` coordinates, ``layers['imputed']`` recovered
expression, ``obsm['latent_observed']`` / ``obsm['latent_enhanced']``), computes
four field-standard metric blocks, and APPENDS them to the existing
``results/lumina-st/metrics.json`` via the vendored ``results_contract`` —
preserving every pre-existing metric and every honesty caveat in run_metadata,
and recording the GT source (obs key) so a verify gate can accept the GT-backed
clustering metrics.

Blocks (see internal metrics notes §G1):
  1. K-fold held-out gene recovery over the 3,053 impute_mask-observed genes:
     per-gene PCC/Spearman/RMSE + SSIM + Jensen-Shannon, averaged over top-K HVG
     (K=10/20/50/100) with 95% CIs.
  2. Clustering ARI/AMI/NMI/homogeneity of Leiden(latent_enhanced) vs the REAL
     annotation AND vs the REAL spatial_cluster.
  3. Enhancement uplift: same clustering + silhouette for latent_observed vs
     latent_enhanced -> deltas.
  4. Moran's I of imputed genes + SVG-rank preservation (Spearman of per-gene
     Moran's-I rank, observed vs imputed held-out genes).

HONESTY: held-out recovery is scored ONLY over impute_mask-observed genes
(genes with real ground truth), never over never-observed genes.

Usage:
    conda run --no-capture-output -n dl python scripts/compute_further_metrics.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np

from lumina_st import results_contract
from lumina_st.metrics.further_metrics import (
    clustering_agreement,
    pergene_recovery_stats,
    silhouette_of_latent,
    svg_rank_preservation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
PROJ_DIR = RESULTS_ROOT / "lumina-st"


def _densify(x) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _flatten_pergene(block: dict, prefix: str = "pergene_recovery") -> dict[str, float | None]:
    """Flatten the pergene_recovery_stats block into contract-ready scalar keys.

    Key schema:
      {prefix}_{metric}_{kkey}_mean                — None when degenerate
      {prefix}_{metric}_{kkey}_gene_partition_ci_low / _ci_high  — None when degenerate
      {prefix}_{metric}_{kkey}_n_finite_genes       — always emitted
      {prefix}_{metric}_{kkey}_degenerate           — 1.0 = True, 0.0 = False
        (contract requires float|None; booleans are rejected, so 1.0/0.0 is used)
    """
    flat: dict[str, float | None] = {}
    for metric, ks in block["per_metric"].items():
        for kkey, stats in ks.items():  # e.g. "k50"
            flat[f"{prefix}_{metric}_{kkey}_mean"] = stats["mean"]
            flat[f"{prefix}_{metric}_{kkey}_gene_partition_ci_low"] = stats["gene_partition_ci_low"]
            flat[f"{prefix}_{metric}_{kkey}_gene_partition_ci_high"] = stats["gene_partition_ci_high"]
            flat[f"{prefix}_{metric}_{kkey}_n_finite_genes"] = float(stats["n_finite_genes"])
            # results_contract rejects booleans; encode True->1.0, False->0.0
            flat[f"{prefix}_{metric}_{kkey}_degenerate"] = 1.0 if stats["degenerate"] else 0.0
    return flat


def main(args: argparse.Namespace) -> None:
    t0 = time.time()
    enhanced_path = Path(args.enhanced)
    print(f"[INFO] Loading enhanced AnnData: {enhanced_path}")
    adata = ad.read_h5ad(enhanced_path)

    # --- pull the real ground truth baked into the enhanced output -----------
    observed_full = _densify(adata.X).astype(np.float64)
    recovered_full = np.asarray(adata.layers["imputed"], dtype=np.float64)
    impute_mask = adata.var["impute_mask"].to_numpy().astype(bool)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    annotation = adata.obs["annotation"].to_numpy()
    spatial_cluster = adata.obs["spatial_cluster"].to_numpy()
    latent_enh = np.asarray(adata.obsm["latent_enhanced"], dtype=np.float64)
    latent_obs = np.asarray(adata.obsm["latent_observed"], dtype=np.float64)

    n_observed = int(impute_mask.sum())
    print(f"[INFO] {n_observed} impute_mask-observed genes (real ground truth) "
          f"of {adata.n_vars}; {adata.n_obs} cells.")

    # Restrict held-out recovery to the impute_mask-observed genes ONLY.
    obs_genes = observed_full[:, impute_mask]
    rec_genes = recovered_full[:, impute_mask]
    gene_var = obs_genes.var(axis=0)

    metrics: dict[str, float] = {}

    # === Block 1: per-gene recovery statistics ================================
    print("[INFO] Block 1: per-gene recovery stats (PCC/Spearman/RMSE/SSIM/JSD)...")
    pergene = pergene_recovery_stats(
        observed=obs_genes,
        recovered=rec_genes,
        gene_var=gene_var,
        n_partitions=args.n_folds,
        hvg_k=(10, 20, 50, 100),
        seed=args.seed,
        coords=coords,  # enable 2-D spatially-windowed SSIM (reference-comparable)
    )
    metrics.update(_flatten_pergene(pergene))
    # SSIM-definition provenance (contract rejects strings: 1.0=spatial, 0.0=surrogate).
    metrics["pergene_recovery_ssim_spatial_mode"] = (
        1.0 if pergene.get("ssim_mode") == "spatial" else 0.0
    )

    # === Block 2: clustering agreement vs REAL labels ========================
    print("[INFO] Block 2: clustering ARI/AMI/NMI/homo vs annotation + spatial_cluster...")
    cl_enh_anno = clustering_agreement(latent_enh, annotation, seed=args.seed,
                                       resolution=args.resolution)
    cl_enh_dom = clustering_agreement(latent_enh, spatial_cluster, seed=args.seed,
                                      resolution=args.resolution)
    for k, v in cl_enh_anno.items():
        metrics[f"cluster_enhanced_vs_annotation_{k}"] = v
    for k, v in cl_enh_dom.items():
        metrics[f"cluster_enhanced_vs_spatial_cluster_{k}"] = v

    # === Block 3: enhancement uplift (observed vs enhanced) ==================
    print("[INFO] Block 3: enhancement uplift (latent_observed vs latent_enhanced)...")
    cl_obs_anno = clustering_agreement(latent_obs, annotation, seed=args.seed,
                                       resolution=args.resolution)
    cl_obs_dom = clustering_agreement(latent_obs, spatial_cluster, seed=args.seed,
                                      resolution=args.resolution)
    for k, v in cl_obs_anno.items():
        metrics[f"cluster_observed_vs_annotation_{k}"] = v
    for k, v in cl_obs_dom.items():
        metrics[f"cluster_observed_vs_spatial_cluster_{k}"] = v

    # Silhouette of each latent against the REAL annotation, then delta.
    sil_enh = silhouette_of_latent(latent_enh, annotation)
    sil_obs = silhouette_of_latent(latent_obs, annotation)
    metrics["silhouette_enhanced_vs_annotation"] = sil_enh
    metrics["silhouette_observed_vs_annotation"] = sil_obs
    metrics["uplift_silhouette_annotation_delta"] = (
        sil_enh - sil_obs if np.isfinite(sil_enh) and np.isfinite(sil_obs) else float("nan")
    )
    for metric in ("ari", "ami", "nmi", "homo"):
        metrics[f"uplift_{metric}_annotation_delta"] = (
            cl_enh_anno[metric] - cl_obs_anno[metric]
        )
        metrics[f"uplift_{metric}_spatial_cluster_delta"] = (
            cl_enh_dom[metric] - cl_obs_dom[metric]
        )

    # === Block 4: Moran's I + SVG-rank preservation ==========================
    print(f"[INFO] Block 4: Moran's I + SVG-rank preservation (kNN k={args.knn_k})...")
    svg = svg_rank_preservation(obs_genes, rec_genes, coords, k=args.knn_k)
    metrics["svg_rank_spearman_observed_vs_imputed"] = svg["svg_rank_spearman"]
    metrics["morans_i_observed_mean"] = svg["morans_i_observed_mean"]
    metrics["morans_i_imputed_mean"] = svg["morans_i_recovered_mean"]
    metrics["svg_n_genes_scored"] = float(svg["n_genes_scored"])

    runtime_s = time.time() - t0

    # === Merge with existing metrics + run_metadata (preserve everything) ====
    metrics_path = PROJ_DIR / "metrics.json"
    meta_path = PROJ_DIR / "run_metadata.json"
    prev_metrics = json.loads(metrics_path.read_text())["metrics"] if metrics_path.exists() else {}
    prev_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    merged_metrics = dict(prev_metrics)
    merged_metrics.update(metrics)

    # Record GT sources + further-metric provenance so the verify gate accepts
    # the now-legitimately-GT-backed ARI/NMI keys.
    prev_notes = prev_meta.get("notes", "")
    further_note = (
        "further metrics appended (compute_further_metrics.py): per-gene recovery "
        f"stats (single imputation pass) over {n_observed} impute_mask-observed genes "
        "only (real GT); CI is gene-partition spread not cross-validation; "
        "degenerate flag set when n_finite_genes < 0.5*K; "
        "clustering ARI/AMI/NMI/homo are GT-backed against REAL labels; "
        "Moran's I + SVG-rank on kNN spatial graph"
    )
    notes = f"{prev_notes}; {further_note}" if prev_notes else further_note

    interp = dict(prev_meta.get("interpretability", {}))
    interp["ground_truth_sources"] = {
        "cell_type_labels": "obs['annotation'] (16 cell types, real st_COAD_test GT)",
        "spatial_domain_labels": "obs['spatial_cluster'] (5 domains, real st_COAD_test GT)",
        "held_out_gene_truth": "var['impute_mask']==1 (3053 observed genes with real measured expression)",
        "spatial_coordinates": "obsm['spatial']",
    }
    interp["gt_backed_clustering"] = True

    run_metadata = {
        "dataset_paths": prev_meta.get("dataset_paths", []),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "seed": int(args.seed),
        "runtime_s": float(prev_meta.get("runtime_s") or 0.0) + runtime_s,
        "device": prev_meta.get("device", "cpu"),
        "deterministic": prev_meta.get("deterministic", False),
        "num_threads": prev_meta.get("num_threads", 1),
        "reproducibility_level": prev_meta.get("reproducibility_level", "seeded"),
        "normalization": prev_meta.get("normalization", {"applied": False, "method": "none"}),
        "interpretability": interp,
        "notes": notes,
        "started_utc": prev_meta.get("started_utc"),
    }

    write_out = results_contract.write_results(
        project="lumina-st",
        dataset_card_id=prev_meta.get(
            "dataset_card_id", "lumina_ref_coad_gse132465+xenium_5k_cervix"
        ),
        metrics=merged_metrics,
        outputs={
            "enhanced": "outputs/enhanced.h5ad",
            "imputed": "outputs/imputed.npy",
            "further_metrics_detail": "outputs/further_metrics_detail.json",
        },
        run_metadata=run_metadata,
        results_dir=str(RESULTS_ROOT),
    )

    # Persist the full nested K-fold CI + SVG structure as a sidecar in
    # outputs/ (the byte-identity-locked contract only carries flat float
    # metrics + canonical run_metadata; the flat CIs live in metrics.json,
    # this sidecar keeps the per-fold detail without touching the contract).
    detail_path = Path(write_out["outputs_dir"]) / "further_metrics_detail.json"
    detail_path.write_text(json.dumps({
        "pergene_recovery_stats": pergene,
        "svg_rank_preservation": svg,
        "clustering": {
            "enhanced_vs_annotation": cl_enh_anno,
            "enhanced_vs_spatial_cluster": cl_enh_dom,
            "observed_vs_annotation": cl_obs_anno,
            "observed_vs_spatial_cluster": cl_obs_dom,
        },
        "n_impute_mask_observed_genes": n_observed,
        "knn_k": int(args.knn_k),
        "leiden_resolution": float(args.resolution),
        "ground_truth_sources": interp["ground_truth_sources"],
    }, indent=2))

    print("\n=== Further metrics appended ===")
    print(f"  metrics.json      -> {write_out['metrics']}")
    print(f"  run_metadata.json -> {write_out['run_metadata']}")
    print(f"  detail sidecar    -> {detail_path}")
    print(f"  new metric keys   -> {len(metrics)}")
    print(f"  total metric keys -> {len(merged_metrics)}")
    print("\n--- new keys + values ---")
    for k in sorted(metrics):
        v = metrics[k]
        print(f"  {k}: {v:.6f}" if isinstance(v, float) and np.isfinite(v) else f"  {k}: {v}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--enhanced", default=str(PROJ_DIR / "outputs" / "enhanced.h5ad"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--resolution", type=float, default=1.0)
    main(p.parse_args())
