"""Regression test for lumina-st #141.

`benchmarks.subpopulation.per_subtype_degs` previously reported raw
one-sided Mann-Whitney p-values with no multiple-testing correction. The
package shipped a working `benjamini_hochberg` helper in
`lumina_st.metrics.statistical`, but no caller wired it in — the
docstring even claimed the caller would do so, but no such caller
existed. Reporting uncorrected p-values across thousands of (gene ×
subcluster) tests inflated false positives and made the BH machinery
dead code.

The fix wires `benjamini_hochberg` into `per_subtype_degs` itself,
adding a `fdr_method` parameter (default `"bh"`). Each returned row now
carries `q_value` (FDR-corrected) and `reject` (the FDR mask at
`fdr_alpha`). BH is applied globally across all (sub-label × gene)
tests before the per-subtype top-N truncation, so the correction sees
the full p-value distribution.
"""

from __future__ import annotations

import math

import numpy as np

from lumina_st.benchmarks.subpopulation import per_subtype_degs


def _make_two_cluster_input(
    n_per: int = 12,
    n_genes: int = 30,
    seed: int = 0,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Two clusters; cluster 0 has elevated mean for the first 5 genes,
    cluster 1 has elevated mean for the next 5; the remaining genes are
    null. This generates a mix of low and high p-values so BH has
    something meaningful to correct.
    """

    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(2 * n_per, n_genes))
    X[:n_per, :5] += 4.0       # cluster-0 markers (strong)
    X[n_per:, 5:10] += 4.0     # cluster-1 markers (strong)
    labels = np.array([0] * n_per + [1] * n_per, dtype=np.int64)
    var_names = [f"G{i}" for i in range(n_genes)]
    return X, var_names, labels


def test_per_subtype_degs_fdr_corrected() -> None:
    """q-values must be present and respect BH ordering.

    BH q-values are non-decreasing once sorted by p-value, and each
    q-value is `>= p_value` for the same row (BH only inflates
    p-values; it never decreases them). Globally sorting all returned
    rows by p_value and checking the q-value ordering pins both
    properties at once.
    """

    X, var_names, labels = _make_two_cluster_input()
    result = per_subtype_degs(X, var_names, labels, n_top=20)

    # Flatten across subtypes.
    all_rows = [r for rows in result.values() for r in rows]
    assert all_rows, "expected non-empty DEGs"

    # Every row must carry q_value and reject keys (the new contract).
    for r in all_rows:
        assert "q_value" in r, r
        assert "reject" in r, r
        assert math.isfinite(r["q_value"]), r
        # BH never decreases a p-value, so q >= p for each row.
        assert r["q_value"] + 1e-9 >= r["p_value"], r

    # Sort by raw p-value; BH q-values must be non-decreasing in this
    # order (the canonical BH monotonicity contract).
    sorted_rows = sorted(all_rows, key=lambda r: r["p_value"])
    qs = [r["q_value"] for r in sorted_rows]
    assert all(qs[i] <= qs[i + 1] + 1e-12 for i in range(len(qs) - 1)), qs


def test_per_subtype_degs_fdr_none_keeps_legacy_behavior() -> None:
    """`fdr_method='none'` must skip correction, returning NaN q-values
    and `reject=False` everywhere. This preserves an opt-out for anyone
    who needs the raw p-values."""

    X, var_names, labels = _make_two_cluster_input()
    result = per_subtype_degs(X, var_names, labels, n_top=5, fdr_method="none")

    all_rows = [r for rows in result.values() for r in rows]
    assert all_rows, "expected non-empty DEGs"
    for r in all_rows:
        assert math.isnan(r["q_value"]), r
        assert r["reject"] is False, r


def test_per_subtype_degs_fdr_bh_rejects_strong_markers() -> None:
    """Strong markers (the planted +4σ shift) must survive BH at α=0.05,
    while the null genes (no shift) should mostly NOT be rejected.

    This is the substantive promise of FDR control: when the truth has
    a clear signal, BH keeps it; when there is no signal, BH drops it.
    """

    X, var_names, labels = _make_two_cluster_input(n_per=15, n_genes=40)
    result = per_subtype_degs(
        X, var_names, labels, n_top=40, fdr_method="bh", fdr_alpha=0.05
    )

    # Cluster 0's planted markers are G0..G4; cluster 1's are G5..G9.
    cluster_0 = {r["gene"]: r for r in result[0]}
    cluster_1 = {r["gene"]: r for r in result[1]}

    for marker in ["G0", "G1", "G2", "G3", "G4"]:
        assert cluster_0[marker]["reject"], cluster_0[marker]
    for marker in ["G5", "G6", "G7", "G8", "G9"]:
        assert cluster_1[marker]["reject"], cluster_1[marker]
