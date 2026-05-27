"""Tests for lumina_st.metrics.statistical (Round 12 W006)."""

from __future__ import annotations

import numpy as np
import pytest

from lumina_st.metrics.statistical import (
    benjamini_hochberg,
    empirical_pvalue,
    permutation_null,
)


class TestPermutationNull:
    def test_deterministic_with_seed(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([5.0, 6.0, 7.0, 8.0])

        def stat(a, b):
            return float(b.mean() - a.mean())

        a = permutation_null(stat, x, y, n_perm=50, random_state=42)
        b = permutation_null(stat, x, y, n_perm=50, random_state=42)
        np.testing.assert_array_equal(a, b)

    def test_null_mean_centered_near_zero(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        def stat(a, b):
            return float(b.mean() - a.mean())

        null = permutation_null(stat, x, y, n_perm=2000, random_state=1)
        # Under H0 (no group difference) the null mean should center near 0.
        assert abs(null.mean()) < 0.5

    def test_empty_group_returns_nans(self) -> None:
        null = permutation_null(lambda a, b: 0.0, np.array([]),
                                np.array([1.0, 2.0]), n_perm=10)
        assert np.isnan(null).all()


class TestEmpiricalPValue:
    def test_observed_at_null_median_returns_high_p(self) -> None:
        rng = np.random.default_rng(0)
        null = rng.normal(size=1000)
        p = empirical_pvalue(0.0, null, alternative="two-sided")
        assert 0.6 < p <= 1.0

    def test_observed_far_from_null_returns_low_p(self) -> None:
        rng = np.random.default_rng(0)
        null = rng.normal(size=1000)
        p = empirical_pvalue(10.0, null, alternative="greater")
        assert p < 0.01

    def test_one_over_n_plus_one_minimum(self) -> None:
        null = np.linspace(0, 1, 10)
        p = empirical_pvalue(100.0, null, alternative="greater")
        assert p == pytest.approx(1.0 / (10 + 1), abs=1e-9)

    def test_bad_alternative_raises(self) -> None:
        with pytest.raises(ValueError):
            empirical_pvalue(0.0, np.zeros(5), alternative="zzz")


class TestBenjaminiHochberg:
    def test_all_significant_at_low_pvalues(self) -> None:
        p = np.array([1e-6, 1e-5, 1e-4, 1e-3])
        out = benjamini_hochberg(p, alpha=0.05)
        assert out["reject"].all()
        assert out["n_rejected"] == 4

    def test_no_significant_at_high_pvalues(self) -> None:
        p = np.array([0.4, 0.5, 0.6, 0.7])
        out = benjamini_hochberg(p, alpha=0.05)
        assert not out["reject"].any()
        assert out["n_rejected"] == 0

    def test_qvalues_monotone(self) -> None:
        # BH q-values must be monotone non-decreasing in sorted-p order.
        p = np.array([0.001, 0.005, 0.01, 0.03, 0.07, 0.5])
        out = benjamini_hochberg(p)
        order = np.argsort(p)
        q_sorted = out["q_values"][order]
        diffs = np.diff(q_sorted)
        assert (diffs >= -1e-9).all()

    def test_nan_preservation(self) -> None:
        p = np.array([0.01, np.nan, 0.02])
        out = benjamini_hochberg(p)
        assert np.isnan(out["q_values"][1])
        assert not out["reject"][1]
