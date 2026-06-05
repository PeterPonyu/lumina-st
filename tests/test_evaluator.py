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


def test_gt_layer_used() -> None:
    """Regression for #97+#127.

    ``EnhancementEvaluator(gt_layer=...)`` previously stored ``gt_layer`` on
    the instance and then silently ignored it — ``run_gene_metrics`` always
    correlated the ``"imputed"`` layer against ``adata.X``. For held-out gene
    evaluation this produced a number measuring imputed-vs-observed
    self-consistency, not imputed-vs-truth accuracy.

    With the fix, ``gt_layer`` is honored: when set and present, the
    correlation is computed against ``adata.layers[gt_layer]``, and the
    resulting metric differs from the ``.X``-based baseline.
    """

    rng = np.random.default_rng(0)
    n_obs, n_vars = 30, 8

    # Three matrices: imputed must correlate near-perfectly with truth and
    # be ~uncorrelated with X so the two reference paths give clearly
    # different numbers.
    truth = rng.normal(loc=5.0, scale=2.0, size=(n_obs, n_vars)).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=2.0, size=(n_obs, n_vars)).astype(np.float32)
    observed_X = rng.normal(loc=5.0, scale=2.0, size=(n_obs, n_vars)).astype(np.float32)
    imputed = truth + 0.01 * noise

    adata = AnnData(X=observed_X)
    adata.layers["imputed"] = imputed
    adata.layers["truth"] = truth

    baseline = EnhancementEvaluator(adata).run_gene_metrics()
    grounded = EnhancementEvaluator(adata, gt_layer="truth").run_gene_metrics()

    # Honoring gt_layer must change the reported numbers.
    assert grounded["mean_pearson"] != pytest.approx(
        baseline["mean_pearson"], abs=1e-3
    ), (baseline, grounded)
    # And the grounded number must reflect imputed-vs-truth (~1.0), while
    # the .X-baseline must be far below it (uncorrelated).
    assert grounded["mean_pearson"] > 0.9, grounded
    assert baseline["mean_pearson"] < 0.5, baseline


def test_gt_layer_missing_raises() -> None:
    """Regression for #97+#127.

    Passing a ``gt_layer`` that is not present in ``adata.layers`` must
    raise rather than silently falling back to ``.X`` — the silent fallback
    was the original bug.
    """

    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(20, 5)).astype(np.float32)
    adata = AnnData(X=X)
    adata.layers["imputed"] = X.copy()

    evaluator = EnhancementEvaluator(adata, gt_layer="not_there")
    with pytest.raises(KeyError, match="gt_layer"):
        evaluator.run_gene_metrics()


def test_summary_clustering_uplift_ungated_without_gt() -> None:
    """Regression for #308.

    ``summary()`` previously surfaced a self-derived clustering AGREEMENT
    number (``leiden_enhanced`` vs ``leiden``, both computed FROM latents),
    which both hid regressions and could be faked positive. With the gate,
    ``summary()`` runs the GT-anchored uplift path: with no class-A ``gt_key``
    the uplift metrics are reported as ``None`` (ungated), never a circular
    agreement score.
    """

    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(20, 6)).astype(np.float32)
    adata = AnnData(X=X)
    adata.layers["imputed"] = X.copy()

    metrics = EnhancementEvaluator(adata).summary()

    # Gene metrics still computed.
    assert "mean_pearson" in metrics, metrics
    # Uplift gated OFF without a class-A GT.
    assert str(metrics["clustering_uplift_gate"]).startswith("ungated"), metrics
    for key in ("ari_delta_over_raw", "ari_enhanced_vs_gt", "nmi_delta_over_raw"):
        assert metrics[key] is None, (key, metrics)
    # The old self-agreement keys must NOT be surfaced as uplift.
    assert "ari_enhanced_vs_original" not in metrics, metrics
