"""Regression test for lumina-st #132.

``EnhancementEvaluator.run_clustering_metrics()`` previously built a single
neighbor graph from the ENHANCED latent and then ran Leiden twice on that
same graph — once as ``leiden_enhanced`` and once as the supposed baseline
``leiden``. The two clusterings were identical by construction, so
``ari_enhanced_vs_original`` and ``nmi_enhanced_vs_original`` were ≡ 1.0
regardless of what enhancement actually did.

The fix builds an INDEPENDENT neighbor graph from ``latent_observed`` (or
from the raw ``.X`` PCA fallback) and runs the baseline Leiden on that graph
so the comparison reflects real partition change.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from lumina_st.metrics.enhancement_evaluator import EnhancementEvaluator


def _make_two_latent_adata(seed: int = 0, n_obs: int = 80, n_latent: int = 10):
    """Build an AnnData with ``latent_observed`` and a SHUFFLED ``latent_enhanced``.

    Cells are drawn around three well-separated centers so Leiden on the
    observed graph yields a non-trivial partition. The enhanced latent is the
    same vectors with rows permuted, so any honest comparison must produce a
    partition disagreement and ARI well below 1.0.
    """

    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5.0, size=(3, n_latent))
    assignments = rng.integers(0, 3, size=n_obs)
    latent_observed = centers[assignments] + rng.normal(scale=0.5, size=(n_obs, n_latent))
    perm = rng.permutation(n_obs)
    latent_enhanced = latent_observed[perm].copy()

    X = rng.normal(loc=5.0, scale=2.0, size=(n_obs, 12)).astype(np.float32)
    adata = AnnData(X=X)
    adata.obsm["latent_observed"] = latent_observed.astype(np.float32)
    adata.obsm["latent_enhanced"] = latent_enhanced.astype(np.float32)
    return adata


def test_ari_not_identity() -> None:
    """Shuffled enhancement must give ARI/NMI clearly below 1.0.

    Pre-fix: ``run_clustering_metrics`` built one neighbor graph from
    ``latent_enhanced`` and ran Leiden on it twice, so ARI was ≡ 1.0 even
    though ``latent_observed`` and ``latent_enhanced`` encoded different
    cluster assignments.

    Post-fix: the baseline Leiden runs on an independent graph built from
    ``latent_observed``, so the two partitions disagree and ARI/NMI fall
    well below 1.0.
    """

    pytest.importorskip("scanpy")
    pytest.importorskip("leidenalg")

    adata = _make_two_latent_adata()
    metrics = EnhancementEvaluator(adata).run_clustering_metrics(n_neighbors=10)

    assert metrics["ari_enhanced_vs_original"] < 0.9, metrics
    assert metrics["nmi_enhanced_vs_original"] < 0.9, metrics
