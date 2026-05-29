"""Regression tests for per-subtype DEG log2 fold-change.

Issue #99 — `per_subtype_degs` computed
`log2((in_mean + eps) / (out_mean + eps))`. On normalized / scaled
expression a group mean can be <= 0, so the ratio goes negative (or
explodes near zero) and `np.log2` returns NaN/-inf. That silently
corrupts the DEG table for exactly the enriched genes a user cares
about (positive in-group mean, negative out-group mean).
"""

from __future__ import annotations

import numpy as np

from lumina_st.benchmarks.subpopulation import per_subtype_degs


def test_log2fc_no_nan_on_negative_means() -> None:
    """log2fc stays finite when group means straddle zero (issue #99)."""
    rng = np.random.default_rng(0)
    n = 40
    half = n // 2
    labels = np.array([0] * half + [1] * half)

    # Gene 0: enriched in subtype 0 with a POSITIVE in-group mean and a
    # NEGATIVE out-group mean -> opposite-sign means -> negative raw ratio.
    # This is the case that produced log2(negative) = NaN on main.
    g0 = np.empty(n)
    g0[labels == 0] = rng.normal(2.0, 0.3, half)
    g0[labels == 1] = rng.normal(-2.0, 0.3, half)

    # Gene 1: both means negative (scaled data) -> denominator near/below 0.
    g1 = np.empty(n)
    g1[labels == 0] = rng.normal(-0.5, 0.3, half)
    g1[labels == 1] = rng.normal(-3.0, 0.3, half)

    X = np.column_stack([g0, g1])
    var_names = ["GENE0", "GENE1"]

    result = per_subtype_degs(X, var_names, labels.tolist(), n_top=10)

    log2fcs = [row["log2fc"] for rows in result.values() for row in rows]
    assert log2fcs, "expected at least one enriched gene to be reported"
    assert all(np.isfinite(v) for v in log2fcs), (
        f"log2fc must be finite even when group means are negative; got {log2fcs}"
    )
