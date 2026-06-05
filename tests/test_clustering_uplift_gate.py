"""Gate for clustering-uplift metrics — lumina-st #308.

A prior real COAD run produced NEGATIVE clustering uplift (ARI Δ −0.085,
NMI Δ −0.340). The previous evaluator could only report a self-derived
agreement number (``leiden_enhanced`` vs ``leiden`` — both computed FROM
latents), which both hid regressions and could be faked positive.

These tests pin three guarantees of ``run_clustering_uplift_metrics``:

1. Class-A GT gating — without a class-A ground-truth column, every uplift
   metric is ``None`` (ungated), not a silently-computed number.
2. Signed Δ-over-raw — enhanced and raw latents are EACH scored against the
   independent GT; a regression shows a NEGATIVE ``*_delta_over_raw``.
3. No circular agreement — scoring against a clustering derived from a latent
   raises rather than returning a fake-perfect score.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from lumina_st.metrics.enhancement_evaluator import EnhancementEvaluator

_METRICS = ("ari", "ami", "nmi", "homogeneity")
_GATED_KEYS = tuple(
    f"{m}_{suffix}"
    for m in _METRICS
    for suffix in ("enhanced_vs_gt", "raw_vs_gt", "delta_over_raw")
)


def _num(out: dict, key: str) -> float:
    """Pull a populated (gated-ON) metric, asserting it is a real number."""
    value = out[key]
    assert isinstance(value, float), (key, value, out)
    return value


def _labeled_adata(
    seed: int = 0,
    n_per: int = 30,
    n_latent: int = 10,
    enhanced_quality: str = "good",
):
    """Build an AnnData with an independent 3-class GT and two latents.

    ``gt`` labels three well-separated groups. ``latent_observed`` (raw)
    recovers them moderately. ``latent_enhanced`` recovers them well when
    ``enhanced_quality='good'`` (positive uplift) or worse-than-raw when
    ``'bad'`` (negative uplift — the COAD failure mode).
    """
    rng = np.random.default_rng(seed)
    n_groups = 3
    labels = np.repeat(np.arange(n_groups), n_per)
    n_obs = labels.size

    centers = rng.normal(scale=8.0, size=(n_groups, n_latent))

    # Raw latent: moderately separated.
    latent_observed = centers[labels] + rng.normal(scale=3.0, size=(n_obs, n_latent))

    if enhanced_quality == "good":
        # Enhanced: tighter clusters -> recovers GT better than raw.
        latent_enhanced = centers[labels] + rng.normal(scale=0.5, size=(n_obs, n_latent))
    elif enhanced_quality == "bad":
        # Enhanced: nearly pure noise -> recovers GT WORSE than raw.
        latent_enhanced = rng.normal(scale=8.0, size=(n_obs, n_latent))
    else:  # pragma: no cover - defensive
        raise ValueError(enhanced_quality)

    X = rng.normal(loc=5.0, scale=2.0, size=(n_obs, 12)).astype(np.float32)
    adata = AnnData(X=X)
    adata.obs["gt"] = labels.astype(str)
    adata.obsm["latent_observed"] = latent_observed.astype(np.float32)
    adata.obsm["latent_enhanced"] = latent_enhanced.astype(np.float32)
    return adata


# --- Rule 1: class-A GT gating ------------------------------------------------


def test_missing_gt_key_yields_none() -> None:
    """No gt_key -> every uplift metric is None, with a reason recorded."""
    adata = _labeled_adata()
    out = EnhancementEvaluator(adata).run_clustering_uplift_metrics()
    for k in _GATED_KEYS:
        assert out[k] is None, (k, out)
    assert str(out["clustering_uplift_gate"]).startswith("ungated")


def test_gt_present_but_not_class_a_yields_none() -> None:
    """GT exists but gt_class_a=False -> ungated None, no silent number."""
    pytest.importorskip("scanpy")
    pytest.importorskip("leidenalg")
    adata = _labeled_adata()
    out = EnhancementEvaluator(adata).run_clustering_uplift_metrics(
        gt_key="gt", gt_class_a=False
    )
    for k in _GATED_KEYS:
        assert out[k] is None, (k, out)
    assert "not asserted class-A" in str(out["clustering_uplift_gate"])


def test_absent_gt_column_yields_none() -> None:
    """gt_key naming a non-existent column -> ungated None, never raises."""
    adata = _labeled_adata()
    out = EnhancementEvaluator(adata).run_clustering_uplift_metrics(
        gt_key="not_there", gt_class_a=True
    )
    for k in _GATED_KEYS:
        assert out[k] is None, (k, out)
    assert "absent" in str(out["clustering_uplift_gate"])


# --- Rule 2: signed delta-over-raw -------------------------------------------


def test_positive_uplift_signed_delta() -> None:
    """Class-A GT + good enhancement -> populated metrics, positive ARI delta."""
    pytest.importorskip("scanpy")
    pytest.importorskip("leidenalg")
    adata = _labeled_adata(enhanced_quality="good")
    out = EnhancementEvaluator(adata).run_clustering_uplift_metrics(
        gt_key="gt", gt_class_a=True, n_neighbors=10
    )
    assert out["clustering_uplift_gate"] == "class-A"
    for k in _GATED_KEYS:
        assert out[k] is not None, (k, out)
    enh = _num(out, "ari_enhanced_vs_gt")
    raw = _num(out, "ari_raw_vs_gt")
    delta = _num(out, "ari_delta_over_raw")
    # Enhanced recovers GT better than raw -> non-negative uplift.
    assert delta == pytest.approx(enh - raw, abs=1e-9)
    assert delta >= 0.0, out
    assert enh > 0.5, out


def test_negative_uplift_shows_negative_delta() -> None:
    """The COAD failure mode: worse-than-raw enhancement -> NEGATIVE delta.

    This is the regression that must be impossible to hide.
    """
    pytest.importorskip("scanpy")
    pytest.importorskip("leidenalg")
    adata = _labeled_adata(enhanced_quality="bad")
    out = EnhancementEvaluator(adata).run_clustering_uplift_metrics(
        gt_key="gt", gt_class_a=True, n_neighbors=10
    )
    assert out["clustering_uplift_gate"] == "class-A"
    # Raw recovers GT; noise-enhanced does not -> delta is clearly negative.
    assert _num(out, "ari_delta_over_raw") < 0.0, out
    assert _num(out, "nmi_delta_over_raw") < 0.0, out
    assert _num(out, "ari_enhanced_vs_gt") < _num(out, "ari_raw_vs_gt"), out


# --- Rule 3: no circular agreement -------------------------------------------


@pytest.mark.parametrize("derived", ["leiden", "leiden_enhanced", "leiden_raw"])
def test_circular_comparison_raises(derived: str) -> None:
    """Scoring against a latent-derived clustering must raise, not fake-pass."""
    adata = _labeled_adata()
    with pytest.raises(ValueError, match="[Cc]ircular"):
        EnhancementEvaluator(adata).run_clustering_uplift_metrics(
            gt_key=derived, gt_class_a=True
        )
