"""Tests for the selective-prediction risk-coverage curve (#303)."""

from __future__ import annotations

import numpy as np
import pytest

from lumina_st.benchmarks import risk_coverage_curve


def _correlated_uncertainty(seed: int = 0, n: int = 500):
    """Build a case where uncertainty correlates with absolute error:
    error magnitude grows with the uncertainty score, so restricting to the
    most-confident subset should reduce risk."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(0.0, 1.0, size=n)
    # uncertainty in [0, 1]; noise std scales with uncertainty.
    uncertainty = rng.uniform(0.0, 1.0, size=n)
    noise = rng.normal(0.0, 1.0, size=n) * uncertainty
    pred = truth + noise
    return truth, pred, uncertainty


def test_output_shapes_and_keys():
    truth, pred, unc = _correlated_uncertainty()
    out = risk_coverage_curve(truth, pred, unc, risk="rmse", n_points=20)
    assert set(out) == {"coverage", "risk", "aurc", "risk_kind", "n"}
    assert out["coverage"].shape == (20,)
    assert out["risk"].shape == (20,)
    assert out["n"] == truth.size
    assert out["risk_kind"] == "rmse"
    # coverage grid spans (0, 1], ending at exactly 1.0
    assert out["coverage"][-1] == pytest.approx(1.0)
    assert out["coverage"][0] > 0.0


def test_confident_subset_has_lower_risk_than_full_set():
    truth, pred, unc = _correlated_uncertainty()
    out = risk_coverage_curve(truth, pred, unc, risk="rmse")
    full_risk = out["risk"][-1]  # coverage == 1.0 == full set
    # The most-confident 25% should have risk <= the full-set risk.
    quarter_idx = int(np.searchsorted(out["coverage"], 0.25))
    assert out["risk"][quarter_idx] <= full_risk
    # And generally the curve should rise: first point <= last point.
    assert out["risk"][0] <= full_risk


def test_aurc_in_zero_to_full_risk_band():
    truth, pred, unc = _correlated_uncertainty()
    out = risk_coverage_curve(truth, pred, unc, risk="rmse")
    full_risk = out["risk"][-1]
    assert 0.0 <= out["aurc"] <= full_risk + 1e-9


def test_deterministic_under_fixed_seed():
    truth, pred, unc = _correlated_uncertainty(seed=7)
    a = risk_coverage_curve(truth, pred, unc, risk="rmse")
    b = risk_coverage_curve(truth, pred, unc, risk="rmse")
    np.testing.assert_array_equal(a["coverage"], b["coverage"])
    np.testing.assert_array_equal(a["risk"], b["risk"])
    assert a["aurc"] == b["aurc"]


def test_random_uncertainty_gives_flatter_curve_higher_aurc():
    """Informative uncertainty should yield a lower AURC than a random,
    uninformative score on the same predictions."""
    truth, pred, unc = _correlated_uncertainty(seed=3)
    informative = risk_coverage_curve(truth, pred, unc, risk="rmse")
    rng = np.random.default_rng(99)
    random_unc = rng.uniform(0.0, 1.0, size=truth.size)
    uninformative = risk_coverage_curve(truth, pred, random_unc, risk="rmse")
    assert informative["aurc"] < uninformative["aurc"]


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        risk_coverage_curve(np.zeros(5), np.zeros(4), np.zeros(5))


def test_invalid_risk_kind_raises():
    truth, pred, unc = _correlated_uncertainty()
    with pytest.raises(ValueError):
        risk_coverage_curve(truth, pred, unc, risk="not_a_risk")


def test_higher_dim_inputs_are_flattened():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(20, 4))
    pred = truth + rng.normal(scale=0.1, size=(20, 4))
    unc = rng.uniform(size=(20, 4))
    out = risk_coverage_curve(truth, pred, unc, risk="rmse")
    assert out["n"] == 80


def test_one_minus_pearson_risk_runs():
    truth, pred, unc = _correlated_uncertainty(seed=1)
    out = risk_coverage_curve(truth, pred, unc, risk="one_minus_pearson")
    assert out["risk_kind"] == "one_minus_pearson"
    assert np.isfinite(out["aurc"])
