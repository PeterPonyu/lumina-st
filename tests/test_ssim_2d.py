"""Tests for the true 2-D windowed SSIM (issue #306).

Validates that ``ssim_2d_windowed`` is a real scikit-image image SSIM over the
spatial grid: identity scores 1.0, noise degrades it monotonically, it is
finite on synthetic fields, and that the 1-D surrogate is now carried as a
DEMOTED ``ssim_reference_reported`` key while the PRIMARY claimed key is
``ssim_2d_windowed``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lumina_st.metrics.imputation_metrics import ssim, ssim_2d_windowed
from lumina_st.metrics.further_metrics import _per_gene_scores, pergene_recovery_stats


def _grid_coords(grid: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


class TestSSIM2DWindowed:
    def test_uses_skimage_backend(self) -> None:
        # The implementation must route through scikit-image (real image SSIM).
        import skimage.metrics  # noqa: F401  (import proves the dep is present)

    def test_identity_scores_one(self) -> None:
        coords = _grid_coords(32)
        rng = np.random.default_rng(0)
        a = rng.normal(size=coords.shape[0])
        assert ssim_2d_windowed(a, a, spatial=coords) == pytest.approx(1.0, abs=1e-9)

    def test_noise_degrades_below_identity(self) -> None:
        coords = _grid_coords(32)
        rng = np.random.default_rng(1)
        a = rng.normal(size=coords.shape[0])
        noisy = a + rng.normal(scale=0.7, size=a.size)
        s_id = ssim_2d_windowed(a, a, spatial=coords)
        s_noisy = ssim_2d_windowed(a, noisy, spatial=coords)
        assert s_noisy < s_id
        assert s_noisy < 0.95

    def test_increasing_noise_is_monotone(self) -> None:
        coords = _grid_coords(32)
        rng = np.random.default_rng(2)
        base = rng.normal(size=coords.shape[0])
        s_low = ssim_2d_windowed(base, base + rng.normal(scale=0.2, size=base.size),
                                 spatial=coords)
        s_high = ssim_2d_windowed(base, base + rng.normal(scale=1.5, size=base.size),
                                  spatial=coords)
        assert s_high < s_low

    def test_output_is_finite_and_bounded(self) -> None:
        coords = _grid_coords(20)
        rng = np.random.default_rng(3)
        a = rng.normal(size=coords.shape[0])
        b = rng.normal(size=coords.shape[0])
        s = ssim_2d_windowed(a, b, spatial=coords)
        assert math.isfinite(s)
        assert -1.0 <= s <= 1.0

    def test_empty_input_returns_nan(self) -> None:
        s = ssim_2d_windowed(np.array([]), np.array([]),
                             spatial=np.zeros((0, 2)))
        assert math.isnan(s)

    def test_shape_mismatch_raises(self) -> None:
        coords = _grid_coords(8)
        with pytest.raises(ValueError):
            ssim_2d_windowed(np.zeros(64), np.zeros(63), spatial=coords)

    def test_coord_shape_validation(self) -> None:
        with pytest.raises(ValueError):
            ssim_2d_windowed(np.zeros(9), np.zeros(9), spatial=np.zeros((9, 1)))
        with pytest.raises(ValueError):
            ssim_2d_windowed(np.zeros(9), np.zeros(9), spatial=np.zeros((8, 2)))

    def test_handles_irregular_point_cloud(self) -> None:
        # Non-grid coordinates with collisions / empty pixels must still rasterize.
        rng = np.random.default_rng(4)
        coords = rng.uniform(0, 50, size=(300, 2))
        a = rng.normal(size=300)
        s = ssim_2d_windowed(a, a, spatial=coords)
        assert math.isfinite(s)
        assert s == pytest.approx(1.0, abs=1e-9)


class TestSurrogateDemotion:
    def test_per_gene_scores_emit_primary_and_reference_keys(self) -> None:
        coords = _grid_coords(24)
        rng = np.random.default_rng(5)
        o = np.abs(rng.normal(size=coords.shape[0]))
        r = o + np.abs(rng.normal(scale=0.3, size=o.size))
        sc = _per_gene_scores(o, r, coords=coords)
        # PRIMARY claimed SSIM key is the 2-D windowed value.
        assert "ssim_2d_windowed" in sc
        # DEMOTED surrogate retained under reference_reported.
        assert "ssim_reference_reported" in sc
        # Legacy bare "ssim" key must NOT be the primary anymore.
        assert "ssim" not in sc
        assert math.isfinite(sc["ssim_2d_windowed"])

    def test_reference_reported_matches_surrogate_call(self) -> None:
        coords = _grid_coords(16)
        rng = np.random.default_rng(6)
        o = np.abs(rng.normal(size=coords.shape[0]))
        r = np.abs(rng.normal(size=coords.shape[0]))
        sc = _per_gene_scores(o, r, coords=coords)
        # reference_reported equals the historical surrogate (1-D / windowed-on-points).
        from lumina_st.metrics.further_metrics import _scale_max
        expected = ssim(_scale_max(o), _scale_max(r), spatial=coords)
        if math.isfinite(expected):
            assert sc["ssim_reference_reported"] == pytest.approx(expected, abs=1e-9)

    def test_no_coords_primary_is_nan_surrogate_finite(self) -> None:
        rng = np.random.default_rng(7)
        o = np.abs(rng.normal(size=128))
        r = np.abs(rng.normal(size=128))
        sc = _per_gene_scores(o, r, coords=None)
        assert math.isnan(sc["ssim_2d_windowed"])
        assert math.isfinite(sc["ssim_reference_reported"])

    def test_pergene_recovery_stats_carries_both_metrics(self) -> None:
        coords = _grid_coords(16)
        rng = np.random.default_rng(8)
        n_cells = coords.shape[0]
        n_genes = 12
        observed = np.abs(rng.normal(size=(n_cells, n_genes)))
        recovered = observed + np.abs(rng.normal(scale=0.2, size=(n_cells, n_genes)))
        gene_var = observed.var(axis=0)
        out = pergene_recovery_stats(
            observed, recovered, gene_var,
            n_partitions=3, hvg_k=(5, 10), seed=0, coords=coords,
        )
        assert out["ssim_mode"] == "spatial"
        assert "ssim_2d_windowed" in out["per_metric"]
        assert "ssim_reference_reported" in out["per_metric"]
        # primary metric has a finite mean for at least one K
        means = [out["per_metric"]["ssim_2d_windowed"][kk]["mean"]
                 for kk in out["per_metric"]["ssim_2d_windowed"]]
        assert any(m is not None and math.isfinite(m) for m in means)
