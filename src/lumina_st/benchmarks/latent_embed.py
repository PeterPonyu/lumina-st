"""Latent embedding helpers for visualization (UMAP / PCA fallback).

Round 13 W003 — closes the stPainter §2.4 (latent-driven UMAP) helper
gap that R12 mislabeled gap-D. UMAP/PCA over a latent matrix is a pure
function and deterministic given a seed; the on-real-data figure waits
for Day 5+, but the helper does not.

UMAP is preferred when ``umap-learn`` is importable; otherwise we fall
back to PCA on a centered matrix. Both paths are deterministic given
``random_state``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


__all__ = [
    "latent_umap",
    "latent_pca",
]


def latent_pca(
    latent_matrix: np.ndarray,
    n_components: int = 2,
    eps: float = 1e-12,
) -> np.ndarray:
    """Deterministic PCA via SVD, used as a UMAP fallback.

    Args:
        latent_matrix: (N, D) input latent representation.
        n_components: output dimensionality (default 2 for plots).
        eps: numerical floor for centering.

    Returns:
        (N, n_components) array of projected coordinates.
    """
    x = np.asarray(latent_matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"latent_matrix must be 2-D, got {x.shape}")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if x.shape[0] == 0:
        return np.zeros((0, n_components), dtype=np.float64)

    centered = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered + eps * 0.0, full_matrices=False)
    k = min(n_components, s.shape[0])
    out = (u[:, :k] * s[:k][None, :])
    if k < n_components:
        pad = np.zeros((x.shape[0], n_components - k), dtype=np.float64)
        out = np.concatenate([out, pad], axis=1)
    return out


def latent_umap(
    latent_matrix: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> dict[str, Any]:
    """UMAP projection of a latent matrix, with deterministic PCA fallback.

    Args:
        latent_matrix: (N, D) input latent representation.
        n_components: output dimensionality.
        n_neighbors: UMAP local-neighborhood size; ignored under fallback.
        min_dist: UMAP min-distance hyperparameter; ignored under fallback.
        random_state: integer seed; controls both UMAP and (trivially) PCA.

    Returns:
        ``{"embedding": (N, n_components) array,
           "method": "umap" or "pca-fallback",
           "n_neighbors": ..., "min_dist": ..., "random_state": ...}``.
    """
    x = np.asarray(latent_matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"latent_matrix must be 2-D, got {x.shape}")
    if n_neighbors < 2:
        raise ValueError(f"n_neighbors must be >= 2, got {n_neighbors}")
    if min_dist < 0:
        raise ValueError(f"min_dist must be >= 0, got {min_dist}")

    try:
        import umap  # type: ignore[import-untyped]
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, max(2, x.shape[0] - 1)),
            min_dist=min_dist,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(x)
        return {
            "embedding": np.asarray(embedding, dtype=np.float64),
            "method": "umap",
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "random_state": random_state,
        }
    except (ImportError, Exception):
        return {
            "embedding": latent_pca(x, n_components=n_components),
            "method": "pca-fallback",
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "random_state": random_state,
        }
