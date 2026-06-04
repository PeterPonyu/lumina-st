"""Smoke / sanity tests using *very small* ST-omics data in real format.

Mimics target ST slices from cards (raw integer counts in .X, obsm['spatial'],
optional cancer_type in obs). Not pure scRNA (no spatial, reference-style cell_type only,
or lognormed X).

Guards recent count_type (#186), validation, and dataset registry contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

import anndata as ad

from lumina_st.data.count_type import assert_raw_counts, infer_count_type, is_raw_counts
from lumina_st.data.validation import AnnDataSchemaValidator


def _tiny_st_adata(
    n_spots: int = 20,
    n_genes: int = 10,
    with_cancer: bool = True,
    seed: int = 123,
) -> ad.AnnData:
    """Tiny realistic ST target following real data format (poisson raw + spatial)."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(lam=3.1, size=(n_spots, n_genes)).astype(np.float32)
    obs = {}
    if with_cancer:
        obs["cancer_type"] = ["BRCA"] * n_spots
    a = ad.AnnData(X=X, obs=obs if obs else None)
    a.var_names = [f"G{i:04d}" for i in range(n_genes)]
    a.obsm["spatial"] = rng.uniform(0, 1000, size=(n_spots, 2)).astype(np.float32)
    return a


def test_tiny_st_raw_and_spatial():
    """Tiny maker + validators accept proper ST format and reject scRNA-like."""
    st = _tiny_st_adata()
    assert infer_count_type(st.X) == "raw"
    assert is_raw_counts(st.X) is True
    assert_raw_counts(st)  # must not raise

    # spatial validation (target ST contract)
    ok = AnnDataSchemaValidator.validate_spatial_data(st, required_obs=["cancer_type"])
    assert ok is True

    # also works without cancer (optional in some paths)
    st2 = _tiny_st_adata(with_cancer=False)
    ok2 = AnnDataSchemaValidator.validate_spatial_data(st2, required_obs=[])
    assert ok2 is True


def test_raw_assert_rejects_pure_scRNA_style():
    """lognorm or gaussian must be rejected by raw count gate (the #186 landmine)."""
    rng = np.random.default_rng(5)
    # lognorm style (small non-int)
    X_log = rng.uniform(0.1, 4.0, size=(15, 7)).astype(np.float32)
    ad_log = ad.AnnData(X=X_log)
    assert infer_count_type(ad_log.X) == "lognorm"
    with pytest.raises(ValueError, match="lognorm"):
        assert_raw_counts(ad_log)

    # gaussian / centered (negatives)
    X_g = rng.normal(0, 1, size=(12, 5)).astype(np.float32)
    ad_g = ad.AnnData(X=X_g)
    assert infer_count_type(ad_g.X) == "unknown"
    with pytest.raises(ValueError, match="unknown|raw"):
        assert_raw_counts(ad_g)


def test_spatial_missing_rejected_for_st_target():
    """ST target without spatial must fail validator (distinguishes from scRNA)."""
    rng = np.random.default_rng(9)
    X = rng.poisson(2, size=(8, 4)).astype(np.float32)
    no_spatial = ad.AnnData(X=X)
    with pytest.raises(ValueError, match="spatial"):
        AnnDataSchemaValidator.validate_spatial_data(no_spatial)
