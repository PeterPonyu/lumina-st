"""Latent embedding helpers for visualization (UMAP / PCA fallback).

Round 13 W003 — closes the §2.4 content gap (latent-driven UMAP) helper
gap that R12 mislabeled gap-D. UMAP/PCA over a latent matrix is a pure
function and deterministic given a seed; the on-real-data figure waits
for Day 5+, but the helper does not.

UMAP is preferred when ``umap-learn`` is importable; otherwise we fall
back to PCA on a centered matrix. Both paths are deterministic given
``random_state``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


__all__ = [
    "latent_umap",
    "latent_pca",
]


logger = logging.getLogger(__name__)


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
    return np.asarray(out, dtype=np.float64)


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
           "fallback_reason": str or None,  # set when UMAP failed at runtime
           "n_neighbors": ..., "min_dist": ..., "random_state": ...}``.

    Fallback policy:
        * ``ImportError`` (``umap-learn`` not installed) → silent PCA
          fallback. This is the documented "umap not available" path.
        * Any other exception raised by UMAP at construction or fit time
          → log a ``WARNING`` with the exception type+message and fall
          back to PCA. ``fallback_reason`` records the same string so
          callers (and any AnnData ``.uns["latent_umap_method"]``
          provenance recorded by them) can tell "umap missing" apart
          from "umap crashed".
    """
    x = np.asarray(latent_matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"latent_matrix must be 2-D, got {x.shape}")
    if n_neighbors < 2:
        raise ValueError(f"n_neighbors must be >= 2, got {n_neighbors}")
    if min_dist < 0:
        raise ValueError(f"min_dist must be >= 0, got {min_dist}")

    fallback_reason: str | None = None
    try:
        import umap  # type: ignore[import-untyped]
    except ImportError:
        fallback_reason = None  # silent: umap-learn not installed
    else:
        try:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(n_neighbors, max(2, x.shape[0] - 1)),
                min_dist=min_dist,
                random_state=random_state,
            )
            embedding = reducer.fit_transform(x)
        except Exception as exc:
            # Real UMAP failure (not a missing dependency): record the
            # exception type+message in both the log and the returned
            # dict so provenance is preserved instead of silently lying
            # that everything was fine.
            fallback_reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "latent_umap: UMAP failed (%s); falling back to deterministic PCA.",
                fallback_reason,
            )
        else:
            return {
                "embedding": np.asarray(embedding, dtype=np.float64),
                "method": "umap",
                "fallback_reason": None,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "random_state": random_state,
            }

    return {
        "embedding": latent_pca(x, n_components=n_components),
        "method": "pca-fallback",
        "fallback_reason": fallback_reason,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "random_state": random_state,
    }
