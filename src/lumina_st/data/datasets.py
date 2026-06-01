"""
LuminaST Data handling — AnnData native datasets.

Replaces the original baseline data/dataset.py with a cleaner, more
extensible design that works directly with scanpy/anndata objects and
the new CancerRegistry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import scipy.sparse as sp
import torch
from anndata import AnnData
from torch.utils.data import Dataset

from ..config.lumina_config import LuminaSTConfig
from .cancer_registry import CancerRegistry


def _row_to_dense_array(X: Any, idx: int) -> np.ndarray:
    """Return ``X[idx]`` as a 1-D dense float ndarray.

    ``AnnData.X`` may be either a dense ``np.ndarray`` or a
    ``scipy.sparse`` matrix. ``np.asarray`` on a sparse row produces a 0-D
    object array, so we densify explicitly before casting.
    """

    row = X[idx]
    if sp.issparse(row):
        row = row.toarray()
    return np.asarray(row).squeeze()


class ReferenceAtlasDataset(Dataset):
    """
    Dataset for the pan-cancer scRNA atlas used to train the VAE + flow model.
    Expects an AnnData with:
        - .X or .layers['counts'] : raw counts
        - .obs['cancer_type'] or .obs[config.vae_batch_key]
    """

    def __init__(
        self,
        adata: AnnData,
        config: LuminaSTConfig,
        registry: Optional[CancerRegistry] = None,
    ):
        self.adata = adata
        self.config = config
        self.registry = registry or CancerRegistry.default_pan_cancer()

        # Pre-compute cancer indices.
        # ``vae_batch_key`` defaults to ``None`` (issue #106) so a forgotten
        # config can't silently miss the cancer-label column. Require the
        # caller to set it explicitly and surface a message that names the
        # field — not the literal ``None`` lookup that would otherwise be
        # rendered by ``adata.obs[None]``.
        key = config.vae_batch_key
        if key is None:
            raise ValueError(
                "vae_batch_key must be configured explicitly when building "
                "ReferenceAtlasDataset (typically 'cancer_type'). "
                "LuminaSTConfig.vae_batch_key defaults to None to prevent a "
                "silent conflict with the 'cancer_type' column used elsewhere."
            )
        if key not in adata.obs:
            raise KeyError(f"Cancer key '{key}' not found in .obs")

        self.cancer_labels = np.array(
            [self.registry[name] for name in adata.obs[key].astype(str)]
        )

    def __len__(self) -> int:
        return self.adata.n_obs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(_row_to_dense_array(self.adata.X, idx)).float()
        y = torch.tensor(self.cancer_labels[idx], dtype=torch.long)
        return x, y


class SpatialTranscriptomicsDataset(Dataset):
    """
    Dataset for ST slices that need enhancement / imputation.

    Expected AnnData:
        - .X : observed (possibly sparse) counts
        - .var['impute_mask'] : boolean mask of genes to impute (optional but recommended)
        - .obs['cancer_type']
    """

    def __init__(
        self,
        adata: AnnData,
        config: LuminaSTConfig,
        registry: Optional[CancerRegistry] = None,
        impute_mask_key: str = "impute_mask",
    ):
        self.adata = adata
        self.config = config
        self.registry = registry or CancerRegistry.default_pan_cancer()
        self.impute_mask_key = impute_mask_key

        key = "cancer_type"  # can be made configurable later
        self.cancer_labels = np.array(
            [self.registry[name] for name in adata.obs.get(key, ["UNKNOWN"] * adata.n_obs)]
        )

        # Impute mask
        if impute_mask_key in adata.var:
            self.impute_mask = torch.from_numpy(adata.var[impute_mask_key].values)
        else:
            self.impute_mask = torch.ones(adata.n_vars, dtype=torch.bool)

    def __len__(self) -> int:
        return self.adata.n_obs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(_row_to_dense_array(self.adata.X, idx)).float()
        y = torch.tensor(self.cancer_labels[idx], dtype=torch.long)
        mask = self.impute_mask

        return {
            "x": x,
            "y": y,
            "impute_mask": mask,
            "index": torch.tensor(idx),
        }
