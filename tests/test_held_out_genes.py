"""Test the LuminaImputer.enhance(held_out_genes=...) mask kwarg added in Wave 2."""
from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer
from lumina_st.data.cancer_registry import CancerRegistry
from lumina_st.latents.tiny_vae import TinyVAE
from lumina_st.models.lumina_transformer import LuminaTransformer
from lumina_st.modules.lumina_flow_module import LuminaFlowModule


def _build_imputer(n_genes: int = 30, n_classes: int = 1) -> LuminaImputer:
    cfg = LuminaSTConfig(
        latent_dim=8,
        hidden_size=16,
        depth=2,
        num_heads=2,
        batch_size=8,
        max_epochs=1,
        cancer_types=["T0"],
        apply_sparsity=False,
    )
    registry = CancerRegistry({"T0": 0})
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
    vae = TinyVAE(input_dim=n_genes, latent_dim=cfg.latent_dim)
    module = LuminaFlowModule(cfg, transformer, vae=vae)
    return LuminaImputer(cfg, module)


def test_held_out_genes_zeros_those_columns():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 20, 30
    X = rng.uniform(1.0, 5.0, (n_cells, n_genes)).astype(np.float32)
    gene_names = [f"G{i:02d}" for i in range(n_genes)]
    a = AnnData(X=X)
    a.var_names = gene_names
    a.obs["cancer_type"] = "T0"

    imputer = _build_imputer(n_genes=n_genes)
    captured = {}

    original_encode = imputer.module.vae.encode_to_latent

    def spying_encode(x, y):
        captured["x"] = x.detach().cpu().numpy().copy()
        return original_encode(x, y)

    imputer.module.vae.encode_to_latent = spying_encode  # type: ignore[assignment]

    held_out = ["G05", "G10", "G15"]
    imputer.enhance(a, cancer_type="T0", held_out_genes=held_out)

    assert "x" in captured, "encoder was not called"
    seen = captured["x"]
    held_idx = [gene_names.index(g) for g in held_out]
    # All cells must see zero at those columns
    assert np.all(seen[:, held_idx] == 0.0), (
        f"held-out columns leaked into encoder input: "
        f"max={float(np.abs(seen[:, held_idx]).max())}"
    )
    # Non-held-out columns must remain non-zero (we used uniform 1..5 input)
    keep_idx = [i for i in range(n_genes) if i not in held_idx]
    assert seen[:, keep_idx].mean() > 0.1, "non-held-out columns were unexpectedly zero"


def test_enhance_unchanged_without_held_out_genes():
    rng = np.random.default_rng(1)
    a = AnnData(X=rng.uniform(0.5, 2.0, (16, 12)).astype(np.float32))
    a.var_names = [f"G{i:02d}" for i in range(12)]
    a.obs["cancer_type"] = "T0"

    imputer = _build_imputer(n_genes=12)
    out_a = imputer.enhance(a, cancer_type="T0")
    out_b = imputer.enhance(a, cancer_type="T0", held_out_genes=None)
    assert "latent_enhanced" in out_a.obsm
    assert "latent_enhanced" in out_b.obsm
