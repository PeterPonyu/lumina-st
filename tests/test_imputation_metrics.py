"""Unit tests for lumina_st.metrics.imputation_metrics (Round 11 W002/W003).

Each test pins a deterministic expected value on synthetic input so the
metric implementations stay honest under refactor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lumina_st.metrics.imputation_metrics import (
    cluster_concordance,
    jensen_shannon_divergence,
    ssim,
)


class TestSSIM:
    def test_identical_arrays_score_1(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(size=(64, 64))
        assert ssim(a, a) == pytest.approx(1.0, abs=1e-9)

    def test_noisy_recon_scores_below_identity(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(size=200)
        noisy = a + rng.normal(scale=0.5, size=200)
        # SSIM is monotone: a recon corrupted by noise must score
        # strictly below the identity comparison.
        s_identity = ssim(a, a)
        s_noisy = ssim(a, noisy)
        assert s_noisy < s_identity
        assert s_noisy < 0.95  # tighter sanity check on substantial noise

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            ssim(np.zeros(3), np.zeros(4))

    def test_empty_input_returns_nan(self) -> None:
        assert math.isnan(ssim(np.array([]), np.array([])))


class TestJensenShannon:
    def test_identical_distributions_zero(self) -> None:
        p = np.array([0.1, 0.2, 0.3, 0.4])
        assert jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_disjoint_supports_in_base_2_is_one(self) -> None:
        # Distributions with disjoint supports → JS in base 2 = 1.
        p = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0, 0.0])
        assert jensen_shannon_divergence(p, q, base=2.0) == pytest.approx(
            1.0, abs=1e-3
        )

    def test_uniform_vs_skewed_strictly_positive(self) -> None:
        uniform = np.full(4, 0.25)
        skewed = np.array([0.7, 0.1, 0.1, 0.1])
        assert jensen_shannon_divergence(uniform, skewed, base=2.0) > 0.05

    def test_zero_sum_returns_nan(self) -> None:
        assert math.isnan(jensen_shannon_divergence(np.zeros(4), np.zeros(4)))

    def test_unnormalized_inputs_get_normalized(self) -> None:
        # Multiplying both sides by a positive constant should not move JS.
        p = np.array([1.0, 2.0, 3.0])
        q = np.array([3.0, 2.0, 1.0])
        a = jensen_shannon_divergence(p, q)
        b = jensen_shannon_divergence(p * 5.0, q * 5.0)
        assert a == pytest.approx(b, abs=1e-9)


class TestClusterConcordance:
    def test_perfect_match_all_metrics_one(self) -> None:
        labels = np.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
        out = cluster_concordance(labels, labels)
        for key in ("ari", "ami", "homo", "nmi"):
            assert out[key] == pytest.approx(1.0, abs=1e-9), key

    def test_permuted_labels_score_one(self) -> None:
        truth = np.array([0, 0, 1, 1, 2, 2])
        # Permute label IDs (0->3, 1->7, 2->0) — clustering identity is
        # invariant to label permutation.
        permuted = np.array([3, 3, 7, 7, 0, 0])
        out = cluster_concordance(truth, permuted)
        assert out["ari"] == pytest.approx(1.0, abs=1e-9)
        assert out["nmi"] == pytest.approx(1.0, abs=1e-9)

    def test_random_labels_low_ari(self) -> None:
        rng = np.random.default_rng(7)
        truth = rng.integers(0, 4, size=200)
        pred = rng.integers(0, 4, size=200)
        out = cluster_concordance(truth, pred)
        # Random labels → ARI ≈ 0 (typically within ±0.1 for n=200).
        assert abs(out["ari"]) < 0.15

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            cluster_concordance([0, 1], [0, 1, 2])

    def test_empty_input_returns_nan_dict(self) -> None:
        out = cluster_concordance([], [])
        for key in ("ari", "ami", "homo", "nmi"):
            assert math.isnan(out[key]), key
