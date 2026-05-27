#!/usr/bin/env python3
"""
High-fidelity Synthetic Data Generator for LuminaST (Data Flow + E2E)

Generates realistic-looking count data that mimics what stPainter used:
- Pan-cancer scRNA reference atlas with differential gene programs per "cancer type"
- Target ST slices with realistic dropout/sparsity (the imputation problem)
- Gene names and proper AnnData structure

This lives in data_flow/ but is heavily used by e2e/ scripts when the user
does not have real local ST data yet (as is currently the case).

The goal is to produce data good enough that training + enhancement shows
clear, meaningful improvement (loss curves, better clustering, imputed values).
"""

import numpy as np
import scanpy as sc
import pandas as pd
from pathlib import Path


def generate_synthetic_reference_and_st(
    n_ref_cells: int = 3000,
    n_st_cells: int = 600,
    n_genes: int = 200,
    n_cancer_types: int = 4,
    seed: int = 42,
    sparsity_target: float = 0.75,   # how sparse the ST should be (simulates dropout)
    lib_size_ref: tuple = (800, 2500),
    lib_size_st: tuple = (400, 1200),
):
    """
    Returns (ref_adata, st_adata, cancer_names)

    Characteristics designed to match baseline stPainter usage:
    - Reference: scRNA-like, multiple "cancer types" with distinct marker genes
    - Target: higher technical noise + dropout (the problem LuminaST solves)
    - Both have .obs['cancer_type'] and gene names
    """
    rng = np.random.default_rng(seed)
    cancer_names = [f"T{i}" for i in range(n_cancer_types)]

    # Create gene names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]

    # === Reference atlas (scRNA) ===
    # Each cancer type gets a few private marker genes + shared background
    ref_X = np.zeros((n_ref_cells, n_genes), dtype=np.float32)
    ref_cancer = rng.choice(cancer_names, n_ref_cells)

    for c_idx, cname in enumerate(cancer_names):
        mask = ref_cancer == cname
        n_cells_c = mask.sum()

        # Base expression (log-normal library sizes)
        lib = rng.uniform(*lib_size_ref, n_cells_c)
        base = rng.normal(0, 0.8, (n_cells_c, n_genes))

        # Cancer-specific upregulation (10-15 genes per type)
        marker_start = c_idx * (n_genes // n_cancer_types)
        marker_genes = list(range(marker_start, marker_start + n_genes // 8))
        base[:, marker_genes] += rng.normal(2.5, 0.6, (n_cells_c, len(marker_genes)))

        # Convert to counts
        counts = np.exp(base)
        counts = (counts.T * lib / counts.sum(axis=1)).T
        counts = rng.poisson(counts).astype(np.float32)

        ref_X[mask] = counts

    ref = sc.AnnData(X=ref_X)
    ref.obs["cancer_type"] = pd.Categorical(ref_cancer)
    ref.obs["batch"] = ref.obs["cancer_type"]
    ref.var_names = gene_names
    ref.var["highly_variable"] = True

    # === Target ST (noisier, sparser) ===
    st_X = np.zeros((n_st_cells, n_genes), dtype=np.float32)
    st_cancer = rng.choice(cancer_names, n_st_cells)

    for c_idx, cname in enumerate(cancer_names):
        mask = st_cancer == cname
        n_cells_c = mask.sum()

        lib = rng.uniform(*lib_size_st, n_cells_c)
        base = rng.normal(0, 1.1, (n_cells_c, n_genes))

        # Same cancer markers but weaker (realistic for ST)
        marker_start = c_idx * (n_genes // n_cancer_types)
        marker_genes = list(range(marker_start, marker_start + n_genes // 8))
        base[:, marker_genes] += rng.normal(1.8, 0.7, (n_cells_c, len(marker_genes)))

        counts = np.exp(base)
        counts = (counts.T * lib / counts.sum(axis=1)).T
        counts = rng.poisson(counts).astype(np.float32)

        # Simulate technical dropout / sparsity typical of ST
        dropout = rng.binomial(1, 1 - sparsity_target, counts.shape)
        counts = counts * dropout

        st_X[mask] = counts

    st = sc.AnnData(X=st_X)
    st.obs["cancer_type"] = pd.Categorical(st_cancer)
    st.var_names = gene_names

    # Add realistic impute_mask (some genes are harder to detect in this "cancer")
    # For demo we mark ~30% of genes as needing imputation for the first cancer type
    if n_cancer_types > 0:
        first_cancer_mask = st.obs["cancer_type"] == cancer_names[0]
        st.var["impute_mask"] = rng.random(n_genes) < 0.30
        st.var["impute_mask"] = st.var["impute_mask"] | (rng.random(n_genes) < 0.15)

    return ref, st, cancer_names


if __name__ == "__main__":
    ref, st, names = generate_synthetic_reference_and_st()
    print("Generated high-fidelity synthetic data (mimics stPainter usage):")
    print(f"  Reference atlas: {ref.n_obs} cells, {ref.n_vars} genes, {len(names)} cancer types")
    print(f"  Target ST      : {st.n_obs} cells, sparsity ~{(st.X == 0).mean():.1%}")
    print(f"  Cancer types   : {names}")

    out_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "synthetic_st_target.h5ad"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st.write_h5ad(out_path)
    print(f"  Saved target ST to: {out_path.relative_to(Path(__file__).resolve().parents[2])}")
    print("Ready for data_flow tests or e2e enhancement runs.")