"""Regression test for lumina-st #103.

``AnnDataSchemaValidator.validate_spatial_data(check_sparsity=True)`` was
documented as "checks if ``adata.X`` has zero-variance or extreme values",
but the implementation only checked ``adata.X is None``. Schema
validation passed on degenerate inputs (all-zero ``.X``, constant genes,
NaN/Inf) that the docstring promised to catch — bad slices reached the
encoder behind a false sense of input validation.

The fix implements the promised behavior: when ``check_sparsity=True``,
reject matrices that

  * contain non-finite values (NaN / ±Inf), or
  * have zero per-gene variance across every gene (all-zero or
    everywhere-constant matrices).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from lumina_st.data.validation import AnnDataSchemaValidator


def _adata_with(X: np.ndarray) -> ad.AnnData:
    n_obs = X.shape[0]
    obsm = {"spatial": np.zeros((n_obs, 2), dtype=np.float32)}
    return ad.AnnData(X=X.astype(np.float32), obsm=obsm)


def test_check_sparsity_actually_checks() -> None:
    """An all-zero matrix must raise when ``check_sparsity=True``.

    Before the fix the docstring promised this behavior but the body
    only checked ``X is None``, so all-zero matrices silently passed.
    """

    adata = _adata_with(np.zeros((5, 5)))

    with pytest.raises(ValueError, match="zero per-gene variance"):
        AnnDataSchemaValidator.validate_spatial_data(adata, check_sparsity=True)


def test_check_sparsity_rejects_constant_matrix() -> None:
    """A non-zero but everywhere-constant matrix must also raise.

    The zero-variance branch is keyed on per-gene variance, not on the
    raw value, so all-fives is just as degenerate as all-zeros.
    """

    adata = _adata_with(np.full((5, 5), 5.0))

    with pytest.raises(ValueError, match="zero per-gene variance"):
        AnnDataSchemaValidator.validate_spatial_data(adata, check_sparsity=True)


def test_check_sparsity_rejects_non_finite_matrix() -> None:
    """NaN / Inf in ``.X`` must raise when ``check_sparsity=True``.

    The docstring promises "extreme values" rejection; previously NaN
    and Inf passed straight into the encoder.
    """

    X = np.ones((4, 3), dtype=np.float32)
    X[0, 0] = np.nan
    adata = _adata_with(X)

    with pytest.raises(ValueError, match="non-finite"):
        AnnDataSchemaValidator.validate_spatial_data(adata, check_sparsity=True)


def test_check_sparsity_accepts_normal_matrix() -> None:
    """A normal varying matrix must still validate (no false positives)."""

    rng = np.random.default_rng(0)
    X = rng.normal(loc=1.0, scale=0.5, size=(10, 4)).astype(np.float32)
    adata = _adata_with(X)

    assert AnnDataSchemaValidator.validate_spatial_data(
        adata, check_sparsity=True
    ) is True


def test_check_sparsity_off_disables_the_check() -> None:
    """Opting out (``check_sparsity=False``) must skip the new checks.

    The flag is a switch; users who deliberately pass degenerate input
    (e.g. mock fixtures) must still be able to bypass the validation.
    """

    adata = _adata_with(np.zeros((5, 5)))

    assert AnnDataSchemaValidator.validate_spatial_data(
        adata, check_sparsity=False
    ) is True
