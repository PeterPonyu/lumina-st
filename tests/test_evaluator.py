"""Regression test for lumina-st #142.

``LuminaSTConfig`` defaulted ``apply_sparsity=True`` with
``sparsity_percentile=0.95``. When no per-gene sparsity-ratio file was
configured (synthetic data, default benchmark sweep), ``enhance()`` fell back
to a per-cell threshold that zeroed every gene below the 95th percentile of
that cell — i.e. it kept only ~5% of genes per cell — and that zeroed
imputation was what the held-out-gene Pearson was computed against.

Every published recovery metric at defaults was therefore measuring the
sparsifier, not the model.

The fix turns sparsity post-processing OFF by default, so held-out gene
Pearson is computed on the raw imputation. Users who want sparsity matching
can still opt in explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.metrics.enhancement_evaluator import EnhancementEvaluator


def test_default_apply_sparsity_is_off() -> None:
    """Default config must NOT zero the bottom of every cell's imputation."""

    config = LuminaSTConfig()
    assert config.apply_sparsity is False, (
        "apply_sparsity must default to False so held-out gene Pearson is "
        "computed on the raw model imputation, not on a 95%-zeroed version."
    )


def test_identity_imputation_pearson_one() -> None:
    """Identity imputation (imputed == observed) must yield mean Pearson 1.0.

    This pins the metric path: when the model 'predicts' the held-out genes
    perfectly, the evaluator must report perfect correlation. Any silent
    post-processing that zeroes part of the imputation before scoring would
    drop this number below 1.0.
    """

    rng = np.random.default_rng(0)
    # Use enough cells per gene that pearsonr has well-defined variance
    # and enough variation across genes that no gene is degenerate.
    n_obs, n_vars = 30, 8
    X = rng.normal(loc=5.0, scale=2.0, size=(n_obs, n_vars)).astype(np.float32)

    adata = AnnData(X=X)
    adata.layers["imputed"] = X.copy()  # identity imputation

    evaluator = EnhancementEvaluator(adata)
    metrics = evaluator.run_gene_metrics()

    assert "mean_pearson" in metrics, metrics
    assert metrics["mean_pearson"] == pytest.approx(1.0, abs=1e-6), metrics
