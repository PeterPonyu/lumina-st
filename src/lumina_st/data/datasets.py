"""
LuminaST Data handling — AnnData native datasets.

Replaces the original baseline data/dataset.py with a cleaner, more
extensible design that works directly with scanpy/anndata objects and
the new CancerRegistry.
"""

from __future__ import annotations

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


def looks_log_normalized(X: Any) -> bool:
    """Heuristic (#314): ``True`` when ``X`` looks log1p-normalized, not raw counts.

    Raw UMI counts are integer-valued — even when stored as ``float`` — and
    routinely exceed 10. ``log1p`` of a normalized matrix is small (max typically
    single digits) AND carries non-integer fractional parts. We flag only when
    BOTH hold, so an integer count matrix that merely happens to be stored as
    ``float`` (e.g. raw Poisson draws cast to ``float32``) never trips the guard.
    """

    if sp.issparse(X):
        data = X.data
        dtype = X.dtype
    else:
        arr = np.asarray(X)
        data = arr.ravel()
        dtype = arr.dtype

    if not np.issubdtype(dtype, np.floating):
        return False

    finite = data[np.isfinite(data)] if data.size else data
    if finite.size == 0:
        return False
    if float(np.max(finite)) >= 10.0:
        return False

    frac = np.abs(finite - np.round(finite))
    return float(np.max(frac)) > 1e-3


def align_to_shared_panel(
    reference: AnnData,
    target: AnnData,
    *,
    min_shared_genes: int = 1,
) -> Tuple[AnnData, AnnData, Dict[str, int]]:
    """Restrict a reference atlas and a target ST slice to their shared gene panel.

    The VAE encoder (scVI / TinyVAE) is trained on the reference's gene space and
    the target must be encoded in that *exact* space, so both objects have to
    agree on a single, identically-ordered gene panel before encoding.

    This is robust to either direction of panel containment:

    * ``reference ⊆ target`` — legacy case (e.g. a ~9.9k-gene atlas vs a 10k
      target). The reference is unchanged; the target is subset + reordered.
    * ``target ⊆ reference`` — e.g. a whole-transcriptome ~33k scRNA atlas vs a
      ~5k Xenium panel. The reference is now subset *down* to the panel so the
      encoder's input dim matches what the target can supply.
    * partial overlap — both are restricted to the intersection.

    Genes are ordered by the reference's ``var_names`` for determinism. Raises
    ``ValueError`` when the shared panel is smaller than ``min_shared_genes``
    (no usable common space — usually a gene-id namespace mismatch).
    """

    target_set = set(map(str, target.var_names))
    shared = [g for g in reference.var_names if str(g) in target_set]
    if len(shared) < max(1, min_shared_genes):
        raise ValueError(
            f"Reference and target share only {len(shared)} gene(s) "
            f"(< required {min_shared_genes}); cannot align panels. "
            f"reference n_vars={reference.n_vars}, target n_vars={target.n_vars}. "
            "Check that both objects use the same gene-id namespace "
            "(e.g. symbols vs Ensembl IDs)."
        )

    stats = {
        "n_shared": len(shared),
        "n_reference_dropped": int(reference.n_vars - len(shared)),
        "n_target_dropped": int(target.n_vars - len(shared)),
    }
    ref_aligned = (
        reference if list(reference.var_names) == shared else reference[:, shared].copy()
    )
    tgt_aligned = (
        target if list(target.var_names) == shared else target[:, shared].copy()
    )
    return ref_aligned, tgt_aligned, stats


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
        validate_counts: bool = True,
    ):
        self.adata = adata
        self.config = config
        self.registry = registry or CancerRegistry.default_pan_cancer()

        # Raw-count guard (#314). The input contract requires RAW integer counts
        # in ``.X``; a log1p-normalized reference shifts the latent space and is
        # incompatible with raw-count ST targets. Reject the obvious case up
        # front rather than silently training on normalized input. The heuristic
        # only fires on float + max<10 + non-integer values, so a legitimate
        # integer count matrix stored as float is never rejected.
        if validate_counts and looks_log_normalized(adata.X):
            raise ValueError(
                "[LuminaST] ReferenceAtlasDataset received a reference whose .X "
                "looks log1p-normalized (float dtype, max < 10, non-integer "
                "values), but the input contract requires RAW integer counts "
                "(lumina-st #314). Supply a raw-count reference (e.g. restore "
                "adata.raw.to_adata() or layers['counts']), or pass "
                "validate_counts=False if this normalization is intended."
            )

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
