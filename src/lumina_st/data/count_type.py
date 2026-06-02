"""Raw-count verification for LuminaST ingestion (issue #186).

The LuminaST loaders (:class:`~lumina_st.data.datasets.SpatialTranscriptomicsDataset`
and :class:`~lumina_st.data.datasets.ReferenceAtlasDataset`) and the dataset
registry (:mod:`lumina_st.data.dataset_registry`) all assume ``.X`` carries
**raw integer counts**. Nothing previously *checked* that assumption, so a
log-normalized matrix would silently flow into the encoder and corrupt every
downstream metric.

This module adds a tiny, dependency-light count-type classifier and an
assertion helper so loaders / cards / tests can fail loudly on preprocessed
input. The classifier samples a bounded number of rows (so it is cheap even on
the 344k-spot COAD target) and densifies only that slice.

Convention
----------
``count_type`` is one of:

* ``"raw"``     — finite, non-negative, integer-valued counts.
* ``"lognorm"`` — finite, non-negative, **non**-integer, small dynamic range
  (consistent with ``log1p`` of library-normalized expression).
* ``"unknown"`` — anything else (negative values, z-scored, NaN/Inf, or an
  ambiguous mix). Callers should treat this as "do not assume raw".
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

__all__ = [
    "CountType",
    "infer_count_type",
    "is_raw_counts",
    "assert_raw_counts",
]

CountType = Literal["raw", "lognorm", "unknown"]


def _sample_dense(X: Any, sample_rows: int) -> np.ndarray:
    """Return a bounded, dense ``float64`` slice of ``X`` for cheap inspection.

    Only the first ``sample_rows`` rows are materialised; on a sparse matrix
    this densifies a single bounded slice rather than the whole object.
    """

    if X is None:
        raise ValueError("expression matrix is None; cannot infer count type.")
    n = X.shape[0]
    rows = min(sample_rows, n)
    head = X[:rows]
    if sp.issparse(head):
        head = head.toarray()
    return np.asarray(head, dtype=np.float64)


def infer_count_type(
    X: Any,
    *,
    sample_rows: int = 2000,
    int_frac_threshold: float = 0.99,
    lognorm_max: float = 30.0,
) -> CountType:
    """Classify an expression matrix as ``"raw"``, ``"lognorm"`` or ``"unknown"``.

    Args:
        X: dense ``np.ndarray`` or ``scipy.sparse`` matrix (cells/spots x genes).
        sample_rows: number of leading rows to inspect (bounds the cost on
            large ST targets).
        int_frac_threshold: minimum fraction of finite entries that must be
            integer-valued for the matrix to be called ``"raw"``.
        lognorm_max: upper bound on the value range below which a
            non-negative, non-integer matrix is treated as log-normalized.

    Returns:
        The inferred :data:`CountType`. Empty matrices return ``"unknown"``.
    """

    head = _sample_dense(X, sample_rows)
    finite = head[np.isfinite(head)]
    if finite.size == 0:
        return "unknown"
    # Non-finite entries anywhere in the sample => not trustworthy raw counts.
    if finite.size != head.size:
        return "unknown"
    if float(finite.min()) < 0.0:
        return "unknown"

    int_frac = float(np.isclose(finite, np.round(finite)).mean())
    if int_frac >= int_frac_threshold:
        return "raw"
    if float(finite.max()) <= lognorm_max:
        return "lognorm"
    return "unknown"


def is_raw_counts(X: Any, **kwargs: Any) -> bool:
    """Return ``True`` iff :func:`infer_count_type` classifies ``X`` as raw."""

    return infer_count_type(X, **kwargs) == "raw"


def assert_raw_counts(adata: Any, *, layer: str | None = None, **kwargs: Any) -> None:
    """Raise ``ValueError`` unless ``adata``'s matrix is raw integer counts.

    Args:
        adata: an AnnData-like object exposing ``.X`` (and ``.layers`` when
            ``layer`` is given).
        layer: optional ``adata.layers`` key to check instead of ``.X``.
        **kwargs: forwarded to :func:`infer_count_type`.

    Raises:
        ValueError: if the inspected matrix is not classified ``"raw"``. The
            message names the inferred type so the caller knows whether the
            input was log-normalized or simply ambiguous.
    """

    X = adata.layers[layer] if layer is not None else adata.X
    kind = infer_count_type(X, **kwargs)
    if kind != "raw":
        where = f"layers[{layer!r}]" if layer is not None else ".X"
        raise ValueError(
            f"Expected raw integer counts in {where}, but inferred count_type="
            f"{kind!r}. LuminaST loaders require raw counts (see issue #186); "
            "pass the raw layer or restore counts before ingestion."
        )
