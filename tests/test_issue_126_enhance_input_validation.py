"""Regression tests for issue #126 — enhance() input validation.

Before the fix, ``LuminaImputer.enhance`` passed the expression matrix straight
into encoding/sampling. NaN/inf cells silently propagated to NaN latents and
NaN imputations, and an empty matrix produced opaque downstream errors. These
tests pin that such inputs now raise a clear error up front.
"""

import numpy as np
import pytest
from anndata import AnnData

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer
from lumina_st.data.cancer_registry import CancerRegistry
from lumina_st.latents.tiny_vae import TinyVAE
from lumina_st.models.lumina_transformer import LuminaTransformer
from lumina_st.modules.lumina_flow_module import LuminaFlowModule


def _build_imputer(n_genes: int = 12) -> LuminaImputer:
    cfg = LuminaSTConfig(
        latent_dim=8, hidden_size=16, depth=2, num_heads=2,
        batch_size=8, max_epochs=1, cancer_types=["T0"], apply_sparsity=False,
    )
    registry = CancerRegistry({"T0": 0})
    transformer = LuminaTransformer(
        latent_dim=cfg.latent_dim, patch_size=1, hidden_size=cfg.hidden_size,
        depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=4.0,
        num_classes=len(registry), class_dropout_prob=0.1,
    )
    vae = TinyVAE(input_dim=n_genes, latent_dim=cfg.latent_dim)
    module = LuminaFlowModule(cfg, transformer, vae=vae)
    return LuminaImputer(cfg, module)


def _adata(X: np.ndarray, n_genes: int) -> AnnData:
    a = AnnData(X=X.astype(np.float32))
    a.var_names = [f"G{i:02d}" for i in range(n_genes)]
    a.obs["cancer_type"] = "T0"
    return a


def test_enhance_rejects_nan_input():
    n_genes = 12
    X = np.ones((6, n_genes), dtype=np.float32)
    X[2, 3] = np.nan
    imputer = _build_imputer(n_genes)
    with pytest.raises(ValueError, match="(?i)nan|finite"):
        imputer.enhance(_adata(X, n_genes), cancer_type="T0")


def test_enhance_rejects_inf_input():
    n_genes = 12
    X = np.ones((6, n_genes), dtype=np.float32)
    X[0, 0] = np.inf
    imputer = _build_imputer(n_genes)
    with pytest.raises(ValueError, match="(?i)inf|finite"):
        imputer.enhance(_adata(X, n_genes), cancer_type="T0")


def test_enhance_rejects_empty_input():
    n_genes = 12
    X = np.empty((0, n_genes), dtype=np.float32)
    imputer = _build_imputer(n_genes)
    with pytest.raises(ValueError, match="(?i)empty|no cells|no genes"):
        imputer.enhance(_adata(X, n_genes), cancer_type="T0")


def test_enhance_accepts_finite_input():
    n_genes = 12
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 5.0, (6, n_genes)).astype(np.float32)
    imputer = _build_imputer(n_genes)
    out = imputer.enhance(_adata(X, n_genes), cancer_type="T0")
    assert out.n_obs == 6
