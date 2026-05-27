#!/usr/bin/env python3
import scanpy as sc
from pathlib import Path
from lumina_st.core.lumina_imputer import LuminaImputer

def main():
    print("=== Testing Baseline Checkpoint Loading and Imputation ===")
    
    # 1. Paths to checkpoints and data
    vae_path = "data/baselines/st_impute_ref/checkpoint/vae_50.ckpt"
    diff_path = "data/baselines/st_impute_ref/checkpoint/diffusion_50.ckpt"
    registry_path = "configs/stpainter_registry.yaml"
    data_path = "data/baselines/st_impute_ref/processed_data/st_CESC_test.h5ad"
    
    if not Path(vae_path).exists() or not Path(diff_path).exists():
        print(
            "[ERROR] Baseline checkpoints not found. "
            "Run `lumina download --cancer-type CESC --output-dir "
            "data/baselines/st_impute_ref/checkpoint` first, or provide local baseline files."
        )
        return 1

    if not Path(data_path).exists():
        print("[ERROR] Target ST dataset not found.")
        return 1

    # 2. Load the imputer from baseline checkpoints
    print("Loading LuminaImputer from checkpoints...")
    imputer = LuminaImputer.from_checkpoint(
        checkpoint_path=diff_path,
        vae_checkpoint_path=vae_path,
        cancer_registry_file=registry_path,
    )
    print("Successfully loaded imputer.")
    print("Config:", imputer.config)

    # 3. Load target ST data and subset it for a fast test run
    print("Loading target ST data...")
    adata = sc.read_h5ad(data_path)
    # Take a small subset of 500 cells for fast test
    adata_sub = adata[:500].copy()
    print(f"Subsetted dataset shape: {adata_sub.shape}")

    # 4. Enhance
    print("Enhancing subsetted ST slice...")
    enhanced = imputer.enhance(adata_sub, cancer_type="CESC")
    
    # 5. Check outputs
    print("\n=== Enhancement Output Check ===")
    print("obsm keys:", list(enhanced.obsm.keys()))
    print("layers keys:", list(enhanced.layers.keys()))
    
    if "latent_observed" in enhanced.obsm:
        print("latent_observed shape:", enhanced.obsm["latent_observed"].shape)
        print("latent_observed mean/std:", enhanced.obsm["latent_observed"].mean(), enhanced.obsm["latent_observed"].std())
        
    if "latent_enhanced" in enhanced.obsm:
        print("latent_enhanced shape:", enhanced.obsm["latent_enhanced"].shape)
        print("latent_enhanced mean/std:", enhanced.obsm["latent_enhanced"].mean(), enhanced.obsm["latent_enhanced"].std())
        
    if "imputed" in enhanced.layers:
        imputed_val = enhanced.layers["imputed"]
        print("imputed shape:", imputed_val.shape)
        print("imputed fraction of zeros (sparsity):", (imputed_val == 0).sum() / imputed_val.size)
        print("imputed mean/std:", imputed_val.mean(), imputed_val.std())
        
    print("\n✅ Verification script completed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
