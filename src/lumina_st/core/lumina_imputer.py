"""
High-level user-facing API for LuminaST.

LuminaImputer is the main class researchers will import and use for both
training and inference. It hides all the Lightning / flow / transformer details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import anndata as ad
import pytorch_lightning as pl
import torch

from ..config.lumina_config import LuminaSTConfig
from ..data.cancer_registry import CancerRegistry
from ..models.lumina_transformer import LuminaTransformer
from ..modules.lumina_flow_module import LuminaFlowModule


class LuminaImputer:
    """
    Main entry point for LuminaST.

    Example:
        imputer = LuminaImputer.from_checkpoint("checkpoints/lumina_50.ckpt")
        enhanced = imputer.enhance(st_adata)
    """

    def __init__(self, config: LuminaSTConfig, module: LuminaFlowModule):
        self.config = config
        self.module = module

    @classmethod
    def from_config(cls, config: LuminaSTConfig) -> "LuminaImputer":
        registry = CancerRegistry.default_pan_cancer()
        transformer = LuminaTransformer(
            latent_dim=config.latent_dim,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_classes=len(registry),
            class_dropout_prob=config.class_dropout_prob,
        )
        module = LuminaFlowModule(config, transformer)
        return cls(config, module)

    def enhance(
        self,
        st_adata: ad.AnnData,
        cancer_type: Optional[str] = None,
        layer: Optional[str] = None,
    ) -> ad.AnnData:
        """
        Full guided enhancement / imputation pipeline.

        Returns a copy of the input AnnData with:
            - .layers['imputed']          : enhanced gene expression
            - .obsm['latent_enhanced']    : improved latent embedding
            - .obsm['latent_observed']    : original encoded latent (for comparison)
        """
        import numpy as np
        import scanpy as sc

        cfg = self.config
        adata = st_adata.copy()

        # 1. Determine cancer label
        if cancer_type is None:
            # Try to infer from .obs
            if "cancer_type" in adata.obs:
                cancer_type = str(adata.obs["cancer_type"].iloc[0])
            else:
                cancer_type = cfg.cancer_types[0] if cfg.cancer_types else "UNKNOWN"

        # 2. Get expression matrix
        if layer is not None and layer in adata.layers:
            expr = adata.layers[layer]
        else:
            expr = adata.X

        if hasattr(expr, "toarray"):
            expr = expr.toarray()

        x = torch.from_numpy(expr).float()

        # 3. Encode to latent space
        if self.module.vae is not None:
            y = torch.full((x.shape[0],), self.module.transformer.y_embedder.num_classes - 1)  # placeholder
            z_obs, _ = self.module.vae.encode_to_latent(x, y)
        else:
            # Assume data is already in latent space or use identity
            z_obs = x

        # 4. Run guided enhancement in latent space
        y = torch.full((z_obs.shape[0],), 0, dtype=torch.long, device=z_obs.device)  # will be overridden by registry later
        z_enhanced = self.module.enhance_latent(z_obs, y)

        # 5. Decode back (if VAE present)
        if self.module.vae is not None:
            x_imputed = self.module.vae.decode_from_latent(z_enhanced, y)
        else:
            x_imputed = z_enhanced

        # 6. Store results
        adata.obsm["latent_observed"] = z_obs.detach().cpu().numpy()
        adata.obsm["latent_enhanced"] = z_enhanced.detach().cpu().numpy()

        if x_imputed.shape[1] == adata.n_vars:
            adata.layers["imputed"] = x_imputed.detach().cpu().numpy()
        else:
            # Latent space only
            adata.layers["imputed_latent"] = x_imputed.detach().cpu().numpy()

        # Optional: basic post-processing (sparsity)
        if cfg.apply_sparsity and "imputed" in adata.layers:
            # Simple top-percentile thresholding per cell (can be improved)
            mat = adata.layers["imputed"]
            thresh = np.percentile(mat, cfg.sparsity_percentile * 100, axis=1, keepdims=True)
            mat[mat < thresh] = 0
            adata.layers["imputed"] = mat

        return adata

    def fit(self, trainer: Optional[pl.Trainer] = None, **trainer_kwargs):
        """Train the flow model on the reference atlas."""
        # Will be wired in Phase 2
        pass
