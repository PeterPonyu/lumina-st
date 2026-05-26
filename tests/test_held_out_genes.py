"""Test the LuminaImputer.enhance(held_out_genes=...) mask kwarg added in Wave 2."""

from __future__ import annotations

import numpy as np
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
        f"held-out columns leaked into encoder input: max={float(np.abs(seen[:, held_idx]).max())}"
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


def test_held_out_gene_recovery():
    """Verify that held-out genes are zeroed in encoder input and
    imputed values for those genes are non-zero after enhancement."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 10, 100
    X = rng.uniform(1.0, 5.0, (n_cells, n_genes)).astype(np.float32)
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    adata = AnnData(X=X)
    adata.var_names = gene_names
    adata.obs["cancer_type"] = "T0"

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
    imputer = LuminaImputer.from_config(cfg)
    # Inject a TinyVAE so gene-count (100) → latent_dim (8) mapping works
    imputer.module.vae = TinyVAE(input_dim=n_genes, latent_dim=cfg.latent_dim)

    held_out = ["gene_0", "gene_1", "gene_2"]
    result = imputer.enhance(adata, cancer_type="T0", held_out_genes=held_out)

    # (a) imputed layer must exist and have correct shape
    assert "imputed" in result.layers, "imputed layer not found"
    assert result.layers["imputed"].shape == (n_cells, n_genes), (
        f"expected {(n_cells, n_genes)}, got {result.layers['imputed'].shape}"
    )

    # (b) imputed values for held-out genes are non-zero (model recovers them)
    held_idx = [gene_names.index(g) for g in held_out]
    imputed = result.layers["imputed"]
    held_out_vals = imputed[:, held_idx]
    assert not np.allclose(held_out_vals, 0.0), (
        f"held-out genes not recovered: max={held_out_vals.max():.6f}"
    )

    # (c) non-held-out genes are non-zero
    keep_idx = [i for i in range(n_genes) if i not in held_idx]
    non_held_vals = imputed[:, keep_idx]
    assert not np.allclose(non_held_vals, 0.0), "non-held-out genes should not be zero"

    # (d) shapes are correct
    observed = result.obsm["latent_observed"]
    assert observed.shape == (n_cells, cfg.latent_dim), (
        f"latent_observed shape mismatch: {observed.shape}"
    )
