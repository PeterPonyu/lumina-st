#!/usr/bin/env python3
import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# Add src and project root to pythonpath
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from lumina_st.config.lumina_config import LuminaSTConfig  # noqa: E402
from lumina_st.latents.tiny_vae import TinyVAE  # noqa: E402
from lumina_st.models.lumina_transformer import LuminaTransformer  # noqa: E402
from lumina_st.modules.lumina_flow_module import LuminaFlowModule  # noqa: E402
from lumina_st.core.lumina_imputer import LuminaImputer  # noqa: E402
from lumina_st.data.datasets import ReferenceAtlasDataset  # noqa: E402
from lumina_st.data.cancer_registry import CancerRegistry  # noqa: E402


def get_device():
    if torch.cuda.is_available():
        try:
            # Test if CUDA works (e.g. to catch RTX 5090 capability mismatches)
            test_tensor = torch.zeros(1, device="cuda")
            _ = torch.relu(test_tensor)
            return torch.device("cuda")
        except Exception as e:
            print(
                f"[WARNING] CUDA is available but failed test execution (likely capability mismatch): {e}"
            )
            print("Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    # 1. Generate or Load Data. Configurable baseline root (#121); default is
    # data/baselines/reference under the repo. Override with LUMINA_BASELINE_ROOT.
    _default_root = Path(__file__).resolve().parents[2] / "data" / "baselines" / "reference"
    DATA_ROOT = Path(os.environ.get("LUMINA_BASELINE_ROOT", str(_default_root)))
    if DATA_ROOT.exists() and any(
        (DATA_ROOT / sub).exists() for sub in ("processed", "processed_data")
    ):
        print("[INFO] Loading real baseline dataset...")
        processed = (
            DATA_ROOT / "processed"
            if (DATA_ROOT / "processed").exists()
            else DATA_ROOT / "processed_data"
        )
        ref = sc.read_h5ad(processed / "sc_train.h5ad")
        possible_targets = list(processed.glob("st_*_test.h5ad"))
        target = sc.read_h5ad(possible_targets[0])
        cancer_type = possible_targets[0].stem.replace("st_", "").replace("_test", "")
    else:
        print("[INFO] Generating synthetic data for evaluation...")
        from scripts.data_flow.generate_synthetic_st import generate_synthetic_reference_and_st

        ref, target, cancer_names = generate_synthetic_reference_and_st(
            n_ref_cells=1500, n_st_cells=400, n_genes=100, n_cancer_types=4, seed=42
        )
        cancer_type = cancer_names[0]
        # Normalize and log1p transform raw counts to prevent gradients/losses from exploding
        sc.pp.normalize_total(ref, target_sum=1e4)
        sc.pp.log1p(ref)
        sc.pp.normalize_total(target, target_sum=1e4)
        sc.pp.log1p(target)

    # Ensure annotations are present
    if "cell_type" not in target.obs:
        # Mock some cell types if absent
        np.random.seed(42)
        cell_types = [f"cell_type_{i}" for i in range(5)]
        target.obs["cell_type"] = np.random.choice(cell_types, size=target.n_obs)

    cancers = sorted(
        ref.obs.get("cancer_type", ref.obs.get("Tumor Type", ["UNKNOWN"])).unique().tolist()
    )
    registry = CancerRegistry({c: i for i, c in enumerate(cancers)})

    cfg = LuminaSTConfig(
        latent_dim=args.latent_dim,
        max_epochs=args.max_epochs,
        batch_size=128,
        guidance_scale=args.guidance_scale,
        cancer_types=cancers,
    )

    # 2. Train VAE
    print("\n--- Training Latent Encoder (TinyVAE fallback) ---")
    vae = TinyVAE(input_dim=ref.n_vars, latent_dim=cfg.latent_dim).to(device)
    vae_wrapper = vae

    # Simple training loop for TinyVAE
    opt_vae = torch.optim.Adam(vae.parameters(), lr=0.01)
    x_tensor = torch.tensor(ref.X, dtype=torch.float32).to(device)
    for epoch in range(15):
        loss_dict = vae(x_tensor)
        loss = loss_dict["loss"]
        opt_vae.zero_grad()
        loss.backward()
        opt_vae.step()

    # 3. Train Flow
    dataset = ReferenceAtlasDataset(ref, cfg, registry)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    transformer = LuminaTransformer(
        latent_dim=cfg.latent_dim,
        patch_size=1,
        hidden_size=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        num_classes=len(registry),
    ).to(device)

    module = LuminaFlowModule(cfg, transformer, vae=vae_wrapper).to(device)
    opt_flow = torch.optim.Adam(module.parameters(), lr=0.001)

    print("--- Training Lumina Flow Module ---")
    for epoch in range(cfg.max_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z, _ = vae_wrapper.encode_to_latent(x, y)
            loss_dict = module.transport.training_losses(transformer, z.detach(), {"y": y})
            loss = loss_dict["loss"]
            opt_flow.zero_grad()
            loss.backward()
            opt_flow.step()

    # 4. Enhance Target ST
    print("\n--- Running Latent Space Enhancement ---")
    imputer = LuminaImputer(cfg, module)
    enhanced = imputer.enhance(target, cancer_type=cancer_type)

    # 5. Latent Space Auditing
    print("\n--- Evaluating Latent Space Quality ---")

    # Extract latents
    target_tensor = torch.tensor(target.X, dtype=torch.float32).to(device)
    with torch.no_grad():
        z_vae, _ = vae_wrapper.encode_to_latent(target_tensor, None)
    z_vae = z_vae.cpu().numpy()
    z_enhanced = enhanced.obsm["latent_enhanced"]

    # Compute Silhouette Scores (distance metric)
    labels = target.obs["cell_type"].values

    sil_vae = silhouette_score(z_vae, labels)
    sil_enhanced = silhouette_score(z_enhanced, labels)

    # Compute clustering performance (Leiden-like clustering on latents)
    # Using scanpy to cluster VAE space
    adata_vae = sc.AnnData(X=z_vae)
    sc.pp.neighbors(adata_vae, use_rep="X")
    sc.tl.leiden(adata_vae, resolution=0.5)
    ari_vae = adjusted_rand_score(labels, adata_vae.obs["leiden"])
    nmi_vae = normalized_mutual_info_score(labels, adata_vae.obs["leiden"])

    # Using scanpy to cluster Enhanced space
    adata_enh = sc.AnnData(X=z_enhanced)
    sc.pp.neighbors(adata_enh, use_rep="X")
    sc.tl.leiden(adata_enh, resolution=0.5)
    ari_enhanced = adjusted_rand_score(labels, adata_enh.obs["leiden"])
    nmi_enhanced = normalized_mutual_info_score(labels, adata_enh.obs["leiden"])

    results = {
        "dataset_shape": target.shape,
        "silhouette_vae": float(sil_vae),
        "silhouette_enhanced": float(sil_enhanced),
        "ari_vae": float(ari_vae),
        "ari_enhanced": float(ari_enhanced),
        "nmi_vae": float(nmi_vae),
        "nmi_enhanced": float(nmi_enhanced),
    }

    print("\nQuality Metrics Summary:")
    print(f"  Silhouette Score (VAE):      {sil_vae:.4f}")
    print(f"  Silhouette Score (Enhanced): {sil_enhanced:.4f}")
    print(f"  Adjusted Rand Index (VAE):   {ari_vae:.4f}")
    print(f"  Adjusted Rand Index (Enh):   {ari_enhanced:.4f}")
    print(f"  Normalized Mutual Info (VAE):{nmi_vae:.4f}")
    print(f"  Normalized Mutual Info (Enh):{nmi_enhanced:.4f}")

    # Save output
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nMetrics written to {args.output}")

    # 6. Try UMAP visualization export if possible
    try:
        import matplotlib.pyplot as plt

        print("\nGenerating UMAP comparison plot...")
        sc.tl.umap(adata_vae)
        sc.tl.umap(adata_enh)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot VAE UMAP
        adata_vae.obs["cell_type"] = labels
        sc.pl.umap(adata_vae, color="cell_type", ax=axes[0], show=False, title="VAE Latent UMAP")

        # Plot Enhanced UMAP
        adata_enh.obs["cell_type"] = labels
        sc.pl.umap(
            adata_enh, color="cell_type", ax=axes[1], show=False, title="Flow-Enhanced Latent UMAP"
        )

        plot_path = output_dir / "latent_umap_comparison.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"UMAP comparison plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of flow training epochs")
    parser.add_argument("--latent_dim", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--output", default="./results/latent_quality_metrics.json")
    args = parser.parse_args()
    main(args)
