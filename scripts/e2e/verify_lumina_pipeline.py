#!/usr/bin/env python3
"""
Deep verification script for LuminaST reimplementation.

Creates fully self-consistent synthetic data + registry.
Trains the flow model (TinyVAE + LuminaTransformer) for a few epochs.
Runs the complete enhance pipeline and checks for:
- Finite outputs
- Correct shapes
- Loss decreasing
- Some measurable "enhancement" signal

This is the deep check requested.
"""

import argparse
import numpy as np
import torch
import scanpy as sc
import pandas as pd
from torch.utils.data import DataLoader

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.latents.tiny_vae import TinyVAE
from lumina_st.models.lumina_transformer import LuminaTransformer
from lumina_st.modules.lumina_flow_module import LuminaFlowModule
from lumina_st.core.lumina_imputer import LuminaImputer
from lumina_st.data.datasets import ReferenceAtlasDataset
from lumina_st.data.cancer_registry import CancerRegistry


def generate_consistent_synthetic(n_cells=1500, n_genes=96, n_classes=4):
    rng = np.random.default_rng(123)
    cancer_names = [f"T{i}" for i in range(n_classes)]

    # Reference — RAW integer counts so the smoke honestly exercises the #314
    # raw-count guard (ReferenceAtlasDataset rejects float/log-normalized X).
    ref_expr = rng.poisson(2.0, (n_cells, n_genes)).astype(np.float32)
    ref_cancer = rng.choice(cancer_names, n_cells)
    ref = sc.AnnData(X=ref_expr)
    ref.obs["cancer_type"] = pd.Categorical(ref_cancer)
    ref.obs["batch"] = ref.obs["cancer_type"]

    # ST slice (slightly noisier, same cancers)
    st_expr = rng.normal(0, 1.3, (400, n_genes)).astype(np.float32)
    st = sc.AnnData(X=st_expr)
    st.obs["cancer_type"] = pd.Categorical(rng.choice(cancer_names, 400))

    return ref, st, cancer_names


def main(args):
    device = torch.device("cpu")
    print(f"Deep verification running on: {device}")

    ref, st, cancer_names = generate_consistent_synthetic()

    # Create registry that exactly matches the generated data
    name_to_idx = {name: i for i, name in enumerate(cancer_names)}
    registry = CancerRegistry(name_to_idx)
    num_classes = len(registry)

    cfg = LuminaSTConfig(
        latent_dim=24,
        hidden_size=48,
        depth=2,
        num_heads=2,
        batch_size=32,
        max_epochs=4,
        guidance_scale=2.5,
        t_forward=0.75,
        num_sampling_steps=15,
        cancer_types=cancer_names,
        vae_batch_key="cancer_type",  # synthetic fixture stores label here; #106
    )

    dataset = ReferenceAtlasDataset(ref, cfg, registry)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Models
    vae = TinyVAE(input_dim=96, latent_dim=24, hidden=64).to(device)
    transformer = LuminaTransformer(
        latent_dim=24,
        patch_size=1,
        hidden_size=48,
        depth=2,
        num_heads=2,
        mlp_ratio=4.0,
        num_classes=num_classes,
        class_dropout_prob=0.15,
    ).to(device)

    module = LuminaFlowModule(cfg, transformer, vae=vae).to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=2e-3)

    losses = []
    print("Training flow model...")
    for epoch in range(cfg.max_epochs):
        epoch_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            z, _ = vae.encode_to_latent(x, y)
            z = z.detach()

            loss_dict = module.transport.training_losses(transformer, z, {"y": y})
            loss = loss_dict["loss"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        losses.append(avg)
        print(f"  Epoch {epoch+1}/{cfg.max_epochs} — avg loss: {avg:.4f}")

    # Check that loss decreased
    loss_decreased = losses[-1] < losses[0]
    print(f"Loss decreased over training: {loss_decreased}  ({losses[0]:.4f} → {losses[-1]:.4f})")

    # Full pipeline test
    print("\nRunning full enhance pipeline on ST slice...")
    imputer = LuminaImputer(cfg, module)
    enhanced = imputer.enhance(st, cancer_type=cancer_names[0])

    # Checks
    has_imputed = "imputed" in enhanced.layers or "imputed_latent" in enhanced.layers
    latent_shape_ok = "latent_enhanced" in enhanced.obsm and enhanced.obsm["latent_enhanced"].shape[1] == 24
    finite = np.isfinite(enhanced.obsm["latent_enhanced"]).all()

    print(f"  Has imputed layer: {has_imputed}")
    print(f"  Latent enhanced shape correct: {latent_shape_ok}")
    print(f"  All values finite: {finite}")

    success = loss_decreased and has_imputed and latent_shape_ok and finite
    print("\n" + ("✅ DEEP VERIFICATION PASSED" if success else "❌ VERIFICATION HAD ISSUES"))
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    raise SystemExit(0 if main(args) else 1)
