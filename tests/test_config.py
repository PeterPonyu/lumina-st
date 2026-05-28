"""Regression tests for ``LuminaSTConfig`` configuration semantics."""

from __future__ import annotations

import pytest

from lumina_st.config.lumina_config import LuminaSTConfig


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
