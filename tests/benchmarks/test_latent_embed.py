"""Tests for lumina_st.benchmarks.latent_embed (Round 13 W003)."""

from __future__ import annotations

import numpy as np
import pytest

from lumina_st.benchmarks.latent_embed import latent_pca, latent_umap


class TestLatentPCA:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 8))
        out = latent_pca(x, n_components=2)
        assert out.shape == (50, 2)

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 5))
        a = latent_pca(x, n_components=2)
        b = latent_pca(x, n_components=2)
        np.testing.assert_allclose(a, b)

    def test_three_clusters_separate_in_pca_space(self) -> None:
        # Three well-separated Gaussian blobs → first PC should split them.
        rng = np.random.default_rng(1)
        a = rng.normal(loc=[0, 0, 0, 0], scale=0.1, size=(30, 4))
        b = rng.normal(loc=[10, 0, 0, 0], scale=0.1, size=(30, 4))
        c = rng.normal(loc=[-10, 0, 0, 0], scale=0.1, size=(30, 4))
        x = np.vstack([a, b, c])
        out = latent_pca(x, n_components=2)
        means = np.array([out[:30].mean(axis=0),
                          out[30:60].mean(axis=0),
                          out[60:].mean(axis=0)])
        # The 3 cluster centroids should be visually distinct.
        d_ab = np.linalg.norm(means[0] - means[1])
        assert d_ab > 5.0

    def test_empty_input_returns_empty(self) -> None:
        out = latent_pca(np.zeros((0, 5)), n_components=2)
        assert out.shape == (0, 2)

    def test_pad_when_n_components_exceeds_rank(self) -> None:
        # Rank-1 matrix → only 1 non-trivial SVD component; pad to 3.
        x = np.ones((5, 3))
        out = latent_pca(x, n_components=3)
        assert out.shape == (5, 3)


class TestLatentUMAP:
    def test_shape_and_method_field(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 6))
        out = latent_umap(x, n_components=2, n_neighbors=5, random_state=0)
        assert out["embedding"].shape == (40, 2)
        assert out["method"] in {"umap", "pca-fallback"}

    def test_random_state_is_passed_through(self) -> None:
        x = np.random.default_rng(0).normal(size=(20, 4))
        out = latent_umap(x, random_state=7)
        assert out["random_state"] == 7

    def test_validates_n_neighbors(self) -> None:
        with pytest.raises(ValueError):
            latent_umap(np.zeros((10, 3)), n_neighbors=1)

    def test_validates_min_dist(self) -> None:
        with pytest.raises(ValueError):
            latent_umap(np.zeros((10, 3)), min_dist=-0.1)

    def test_bad_input_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            latent_umap(np.zeros(10))
