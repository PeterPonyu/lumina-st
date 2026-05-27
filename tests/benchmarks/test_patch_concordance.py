"""Tests for lumina_st.benchmarks.patch_concordance (Round 12 W003)."""

from __future__ import annotations

import numpy as np
import pytest

from lumina_st.benchmarks.patch_concordance import (
    patch_concordance,
    per_patch_celltype_proportions,
    segment_into_patches,
)


class TestSegmentIntoPatches:
    def test_unit_square_2x2_grid(self) -> None:
        # 4 cells in 4 quadrants of [0,1]^2.
        coords = np.array([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]])
        pid = segment_into_patches(coords, n_x=2, n_y=2)
        # Patch index = bx * n_y + by; expect 0, 2, 1, 3.
        assert pid.tolist() == [0, 2, 1, 3]

    def test_out_of_bbox_gets_minus_one(self) -> None:
        coords = np.array([[0.5, 0.5], [10.0, 0.5]])
        pid = segment_into_patches(coords, n_x=2, n_y=2,
                                   bbox=(0.0, 1.0, 0.0, 1.0))
        assert pid[0] >= 0
        assert pid[1] == -1

    def test_degenerate_bbox_raises(self) -> None:
        coords = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError):
            segment_into_patches(coords, n_x=2, n_y=2, bbox=(0.0, 0.0, 0.0, 0.0))


class TestPerPatchProportions:
    def test_row_sums_to_one_in_populated_patches(self) -> None:
        pid = np.array([0, 0, 0, 1])
        labels = ["A", "B", "A", "B"]
        prop, cts = per_patch_celltype_proportions(pid, labels, n_patches=2)
        assert cts == ["A", "B"]
        assert prop[0, 0] == pytest.approx(2 / 3)
        assert prop[0, 1] == pytest.approx(1 / 3)
        assert prop[1, 1] == pytest.approx(1.0)

    def test_empty_patch_stays_zero(self) -> None:
        pid = np.array([0])
        labels = ["A"]
        prop, _ = per_patch_celltype_proportions(pid, labels, n_patches=3)
        assert (prop[1] == 0).all()
        assert (prop[2] == 0).all()


class TestPatchConcordance:
    def test_identical_labels_score_one(self) -> None:
        coords = np.random.default_rng(0).uniform(size=(40, 2))
        labels = ["A"] * 20 + ["B"] * 20
        out = patch_concordance(coords, labels, labels, n_x=2, n_y=2)
        # All non-degenerate cell types reach Pearson 1.
        nonnan = [v for v in out["per_celltype_pearson"].values() if not np.isnan(v)]
        assert len(nonnan) >= 1
        assert all(v == pytest.approx(1.0, abs=1e-9) for v in nonnan)

    def test_shuffled_labels_lower_pearson(self) -> None:
        rng = np.random.default_rng(0)
        coords = rng.uniform(size=(100, 2))
        truth = ["A" if c[0] < 0.5 else "B" for c in coords]
        rng.shuffle(truth)
        recon = list(truth)
        rng.shuffle(recon)
        out = patch_concordance(coords, truth, recon, n_x=5, n_y=5)
        # Random shuffle should yield a mean Pearson well below 1.
        assert out["mean_pearson"] < 0.5
