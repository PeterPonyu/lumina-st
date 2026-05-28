"""Regression test for lumina-st #95 and #109.

``ReferenceAtlasDataset`` and ``SpatialTranscriptomicsDataset`` must accept an
``AnnData`` whose ``.X`` is a ``scipy.sparse`` matrix (CSR/CSC). Real ST
datasets (Visium, Xenium, Slide-seq, etc.) ship counts as sparse matrices, so
the dataset has to densify per-row before tensor cast — otherwise
``torch.from_numpy(np.asarray(adata.X[idx]).squeeze())`` raises because
``np.asarray`` of a sparse row produces a 0-d object array, not a dense ndarray.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from anndata import AnnData

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.data.datasets import (
    ReferenceAtlasDataset,
    SpatialTranscriptomicsDataset,
)


def _make_anndata_sparse(n_obs: int = 4, n_vars: int = 6) -> AnnData:
    rng = np.random.default_rng(0)
    dense = rng.poisson(1.0, size=(n_obs, n_vars)).astype(np.float32)
    # Force a non-trivial sparsity pattern
    dense[dense < 1] = 0.0
    X = sp.csr_matrix(dense)
    obs = {"cancer_type": ["coad"] * n_obs}
    adata = AnnData(X=X, obs=obs)
    return adata


def test_sparse_X_no_crash() -> None:
    """Both datasets must yield dense float tensors from sparse ``.X``."""

    adata = _make_anndata_sparse()
    assert sp.issparse(adata.X), "fixture must be sparse to exercise the bug"

    config = LuminaSTConfig()

    # ReferenceAtlasDataset — uses vae_batch_key (defaults to 'cancer_type'
    # via the config). We re-key obs to match whatever the config expects.
    ref_key = config.vae_batch_key
    if ref_key not in adata.obs:
        adata.obs[ref_key] = adata.obs["cancer_type"].values

    ref_ds = ReferenceAtlasDataset(adata=adata, config=config)
    x, y = ref_ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert x.shape == (adata.n_vars,)
    assert torch.isfinite(x).all()

    # SpatialTranscriptomicsDataset
    st_ds = SpatialTranscriptomicsDataset(adata=adata, config=config)
    sample = st_ds[0]
    assert isinstance(sample["x"], torch.Tensor)
    assert sample["x"].dtype == torch.float32
    assert sample["x"].shape == (adata.n_vars,)
    assert torch.isfinite(sample["x"]).all()
