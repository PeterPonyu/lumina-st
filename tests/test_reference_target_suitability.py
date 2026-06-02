"""Regression tests for the reference<->target suitability guard (issue #212).

On ``main`` (before this fix) ``check_reference_target_suitability`` does not
exist, so these tests error/fail; they pass with the fix. The core regression
is the real LuminaST landmine: a PBMC/Acute-Lymphocytic-Leukemia reference
(GSE132509) paired with a solid-tumor COAD target must be flagged as a
mismatch instead of silently producing negative-uplift "results".
"""

import logging

import anndata as ad
import numpy as np
import pytest

from lumina_st.data.validation import AnnDataSchemaValidator


def _adata(cancer_type: str, n: int = 6) -> ad.AnnData:
    return ad.AnnData(
        X=np.zeros((n, 3), dtype=np.float64),
        obs={"cancer_type": [cancer_type] * n},
    )


def test_pbmc_reference_vs_coad_target_is_mismatch():
    # The #212 landmine: GSE132509 PBMC/ALL reference vs COAD target.
    suitable = AnnDataSchemaValidator.check_reference_target_suitability(
        "AcuteLymphocyticLeukemia_PBMC", "COAD", on_mismatch="warn"
    )
    assert suitable is False


def test_mismatch_raises_when_requested():
    with pytest.raises(ValueError, match="MISMATCH"):
        AnnDataSchemaValidator.check_reference_target_suitability(
            _adata("AcuteLymphocyticLeukemia_PBMC"), "COAD", on_mismatch="raise"
        )


def test_matched_cancer_type_is_suitable():
    assert AnnDataSchemaValidator.check_reference_target_suitability(
        _adata("COAD"), _adata("COAD"), on_mismatch="raise"
    ) is True
    # Case/whitespace-insensitive.
    assert AnnDataSchemaValidator.check_reference_target_suitability(
        "coad", " COAD ", on_mismatch="raise"
    ) is True


def test_wildcard_reference_is_always_suitable():
    # A pan-cancer / UNKNOWN reference pool is compatible with any target.
    assert AnnDataSchemaValidator.check_reference_target_suitability(
        "UNKNOWN", "COAD", on_mismatch="raise"
    ) is True
    assert AnnDataSchemaValidator.check_reference_target_suitability(
        ["COAD", "BRCA", "MIXED"], "PRAD", on_mismatch="raise"
    ) is True


def test_warn_emits_log_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="lumina_st.data.validation"):
        AnnDataSchemaValidator.check_reference_target_suitability(
            "PBMC", "COAD", on_mismatch="warn"
        )
    assert any("MISMATCH" in rec.message for rec in caplog.records)


def test_missing_obs_key_raises_keyerror():
    no_label = ad.AnnData(X=np.zeros((4, 3), dtype=np.float64))
    with pytest.raises(KeyError, match="cancer_type"):
        AnnDataSchemaValidator.check_reference_target_suitability(no_label, "COAD")


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="on_mismatch"):
        AnnDataSchemaValidator.check_reference_target_suitability(
            "COAD", "COAD", on_mismatch="explode"
        )
