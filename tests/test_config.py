"""Regression tests for ``LuminaSTConfig`` configuration semantics."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.data.datasets import ReferenceAtlasDataset


def test_vae_batch_key_no_silent_conflict() -> None:
    """``vae_batch_key`` must not silently default to a value that conflicts
    with the cancer-label column other call sites read (issue #106).

    Historically the default was the literal ``"batch"``. ``cancer_type`` is
    the canonical cancer-label column used elsewhere (e.g.
    ``SpatialTranscriptomicsDataset``), so a config that left
    ``vae_batch_key`` at the default while the AnnData stored labels under
    ``cancer_type`` would silently miss the label and either KeyError at
    dataset construction or label every cell as the same fallback class.

    The fix is to make the default ``None`` so users must opt in explicitly.
    A KeyError-on-missing remains the failure mode; the default no longer
    points at a stale column name that happens to exist in foreign atlases.
    """
    cfg = LuminaSTConfig()
    assert cfg.vae_batch_key is None, (
        "vae_batch_key must default to None so it cannot silently conflict "
        "with the 'cancer_type' column used by other call sites."
    )


def test_vae_batch_key_none_raises_when_dataset_consumes_it() -> None:
    """If ``vae_batch_key`` is ``None`` and a caller tries to read it, the
    failure must be loud (clear ``ValueError``) — not a confusing KeyError on
    a literal ``None`` lookup or silent fallback to a wrong column.
    """
    cfg = LuminaSTConfig()  # default vae_batch_key=None
    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        X=rng.poisson(2.0, size=(10, 8)).astype(np.float32),
        obs={"cancer_type": ["COAD"] * 10},
    )
    with pytest.raises((ValueError, KeyError)) as exc_info:
        ReferenceAtlasDataset(adata, cfg)
    # The error message must mention vae_batch_key explicitly so the user
    # knows what to configure.
    assert "vae_batch_key" in str(exc_info.value), (
        "Error message must name the missing configuration key."
    )


def test_vae_batch_key_explicit_value_still_works() -> None:
    """Explicitly setting ``vae_batch_key`` to a real ``.obs`` column must
    continue to work — the default change is opt-in, not a regression."""
    cfg = LuminaSTConfig(vae_batch_key="cancer_type", cancer_types=["COAD"])
    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        X=rng.poisson(2.0, size=(10, 8)).astype(np.float32),
        obs={"cancer_type": ["COAD"] * 10},
    )
    ds = ReferenceAtlasDataset(adata, cfg)
    assert len(ds) == 10


def test_get_cancer_index_per_type() -> None:
    """Distinct cancer types must map to distinct indices (issue #101).

    The previous implementation was a placeholder that returned 0 for every
    input, so distinct cancer types collapsed onto the same class index — a
    silent bug that would have made class-conditioned generation indistinguishable
    across cancer types.
    """
    cfg = LuminaSTConfig(cancer_types=["COAD", "OV", "LIHC"])
    idx_coad = cfg.get_cancer_index("COAD")
    idx_ov = cfg.get_cancer_index("OV")
    idx_lihc = cfg.get_cancer_index("LIHC")

    # Indices must be distinct ints.
    assert idx_coad != idx_ov
    assert idx_coad != idx_lihc
    assert idx_ov != idx_lihc
    assert isinstance(idx_coad, int)


def test_get_cancer_index_is_case_insensitive() -> None:
    """``cancer_types`` are upper-cased by the field validator, so lookups
    must accept any case for the same name."""
    cfg = LuminaSTConfig(cancer_types=["COAD", "OV"])
    assert cfg.get_cancer_index("coad") == cfg.get_cancer_index("COAD")
    assert cfg.get_cancer_index("Ov") == cfg.get_cancer_index("OV")


def test_get_cancer_index_unknown_raises() -> None:
    """Unknown cancer names must raise rather than silently return 0."""
    cfg = LuminaSTConfig(cancer_types=["COAD", "OV"])
    with pytest.raises((KeyError, ValueError)):
        cfg.get_cancer_index("NSCLC")
