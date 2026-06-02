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

    def test_spatial_windowed_identical_scores_1(self) -> None:
        # Windowed SSIM must still report 1.0 for an exact reconstruction.
        xs, ys = np.meshgrid(np.arange(16), np.arange(16))
        coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
        rng = np.random.default_rng(3)
        a = rng.normal(size=coords.shape[0])
        assert ssim(a, a, spatial=coords) == pytest.approx(1.0, abs=1e-9)

    def test_spatial_windowed_validates_coord_shape(self) -> None:
        with pytest.raises(ValueError):
            ssim(np.zeros(9), np.zeros(9), spatial=np.zeros((9, 1)))  # need >=2 cols
        with pytest.raises(ValueError):
            ssim(np.zeros(9), np.zeros(9), spatial=np.zeros((8, 2)))  # length mismatch

    def test_spatial_vs_surrogate_diverge_on_local_texture(self) -> None:
        """Parity regression for issue #217: 1-D surrogate vs 2-D spatial SSIM.

        Construct a field whose *coarse* spatial structure is reproduced exactly
        but whose *fine local texture* is replaced by independent noise. The 1-D
        global surrogate (no spatial windowing) is dominated by the matching
        coarse structure and reports a near-perfect score, while the
        spatially-windowed SSIM sees that within each local window the texture is
        uncorrelated and scores substantially lower. This is exactly the
        spatial-texture sensitivity the surrogate lacks; the two definitions must
        therefore diverge. (On origin/main `ssim` has no `spatial=` kwarg, so this
        test fails there — proving the metric definition actually changed.)
        """
        grid = 24
        xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
        coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
        # Coarse blocky region means (shared by truth and recon -> fools global SSIM).
        coarse = (((xs // 6) + (ys // 6)).astype(float) * 10.0).ravel()
        rng = np.random.default_rng(7)
        truth = coarse + rng.normal(scale=1.0, size=coarse.size)
        # Same coarse structure, but the fine local texture is independent noise.
        recon = coarse + rng.normal(scale=1.0, size=coarse.size)

        s_global = ssim(truth, recon)                       # 1-D surrogate
        s_spatial = ssim(truth, recon, spatial=coords, n_windows=8)

        # Global surrogate is fooled high by the matching coarse structure.
        assert s_global > 0.9
        # Spatial SSIM detects the destroyed local texture and scores well below.
        assert s_spatial < 0.8
        # The two definitions provably diverge (not an apples-to-apples metric).
        assert (s_global - s_spatial) > 0.1


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
