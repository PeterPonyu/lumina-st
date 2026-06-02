"""Regression tests for raw-count verification (issue #186).

On ``main`` (before this fix) ``lumina_st.data.count_type`` does not exist, so
importing it fails and the suite errors — i.e. these tests FAIL on main and
PASS with the fix. The core regression is that a ``log1p``-normalized matrix
must be flagged *not raw*, which previously flowed silently into the encoder.
"""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from lumina_st.data import (
    assert_raw_counts,
    infer_count_type,
    is_raw_counts,
)


def _raw_counts(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.poisson(2.0, size=(50, 8)).astype(np.float64)


def test_raw_integer_counts_classified_raw():
    X = _raw_counts()
    assert infer_count_type(X) == "raw"
    assert is_raw_counts(X) is True
    assert is_raw_counts(sp.csr_matrix(X)) is True


def test_log1p_matrix_flagged_not_raw():
    # The core #186 regression: a log-normalized matrix must NOT be called raw.
    X = _raw_counts()
    X_log = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)
    assert infer_count_type(X_log) == "lognorm"
    assert is_raw_counts(X_log) is False


def test_zscored_matrix_is_unknown():
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, size=(50, 8))  # has negatives
    assert infer_count_type(X) == "unknown"


def test_assert_raw_counts_raises_on_lognorm():
    X = _raw_counts()
    X_log = np.log1p(X)
    adata = ad.AnnData(X=X_log)
    with pytest.raises(ValueError, match="raw integer counts"):
        assert_raw_counts(adata)


def test_assert_raw_counts_passes_on_raw_and_checks_layer():
    X = _raw_counts()
    adata = ad.AnnData(X=np.log1p(X), layers={"counts": X})
    # .X is log-normalized -> must raise...
    with pytest.raises(ValueError):
        assert_raw_counts(adata)
    # ...but the raw `counts` layer passes.
    assert_raw_counts(adata, layer="counts")
