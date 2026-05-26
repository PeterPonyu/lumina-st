"""Tests for the native conformal uncertainty calibrator."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from lumina_st.benchmarks import AdapterInput, ConformalCalibrator
from lumina_st.benchmarks.adapters import KNNAdapter, MeanAdapter
from lumina_st.benchmarks.contract import BaseAdapter


def _make_adata(n_cells: int = 200, n_genes: int = 30, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    return adata


class _IdentityAdapter(BaseAdapter):
    """Returns the truth as imputation. Should yield coverage=1.0 width=0."""

    name = "identity"

    def __init__(self, truth_X: np.ndarray):
        super().__init__()
        self._truth = truth_X

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        out = masked.copy()
        out.layers["imputed"] = self._truth.copy()
        return out


class _NoisyAdapter(BaseAdapter):
    """Imputes truth + Gaussian noise σ. Conformal coverage should ≈ 1-α."""

    name = "noisy"

    def __init__(self, truth_X: np.ndarray, sigma: float = 1.0, seed: int = 0):
        super().__init__()
        self._truth = truth_X
        self._sigma = sigma
        self._seed = seed

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        rng = np.random.default_rng(self._seed + inp.seed)
        out = masked.copy()
        out.layers["imputed"] = (self._truth + rng.normal(0, self._sigma, size=self._truth.shape)).astype(np.float32)
        return out


# -- Constructor validation ----------------------------------------------


def test_conformal_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        ConformalCalibrator(base=MeanAdapter(), alpha=0.0)
    with pytest.raises(ValueError):
        ConformalCalibrator(base=MeanAdapter(), alpha=1.0)


def test_conformal_rejects_invalid_cal_fraction():
    with pytest.raises(ValueError):
        ConformalCalibrator(base=MeanAdapter(), cal_fraction=0.0)
    with pytest.raises(ValueError):
        ConformalCalibrator(base=MeanAdapter(), cal_fraction=1.5)


# -- Perfect adapter → zero-width, full coverage --------------------------


def test_identity_adapter_yields_perfect_coverage_zero_width():
    adata = _make_adata(n_cells=80, seed=0)
    held_out = ["GENE_005", "GENE_010", "GENE_015"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=42)

    truth_X = np.asarray(adata.X, dtype=np.float32)
    calibrator = ConformalCalibrator(base=_IdentityAdapter(truth_X), alpha=0.1)
    result = calibrator.run(inp)

    assert result.status == "ok", result.status
    assert result.metrics_json["empirical_coverage"] == pytest.approx(1.0)
    assert result.metrics_json["mean_interval_width"] == pytest.approx(0.0, abs=1e-6)
    # Layers populated
    assert "imputed_lower" in result.imputed_h5ad.layers
    assert "imputed_upper" in result.imputed_h5ad.layers
    assert "imputed_width" in result.imputed_h5ad.layers


# -- Noisy adapter → coverage near nominal ---------------------------------


def test_noisy_adapter_achieves_near_nominal_coverage():
    adata = _make_adata(n_cells=400, n_genes=20, seed=0)
    held_out = ["GENE_002", "GENE_007", "GENE_012", "GENE_017"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=42)

    truth_X = np.asarray(adata.X, dtype=np.float32)
    calibrator = ConformalCalibrator(
        base=_NoisyAdapter(truth_X, sigma=1.0, seed=1), alpha=0.1,
    )
    result = calibrator.run(inp)

    coverage = result.metrics_json["empirical_coverage"]
    # Split-conformal guarantee: coverage ≈ 1 - α = 0.9; allow ±0.07 slack
    # for the moderate sample size and Gaussian residual tail.
    assert 0.83 <= coverage <= 0.97, f"coverage {coverage} outside expected band"
    # Width must be positive (noisy adapter is not perfect)
    assert result.metrics_json["mean_interval_width"] > 0.0


# -- Schema -------------------------------------------------------------


def test_conformal_result_schema_includes_uncertainty_fields():
    adata = _make_adata(n_cells=120, seed=0)
    inp = AdapterInput(input_h5ad=adata, held_out_genes=["GENE_004"], seed=0)
    result = ConformalCalibrator(base=MeanAdapter(), alpha=0.2).run(inp)

    assert result.status == "ok"
    m = result.metrics_json
    for k in ("empirical_coverage", "mean_interval_width", "alpha",
              "nominal_coverage", "n_calibration_cells", "n_test_cells"):
        assert k in m, f"missing field {k}"
    assert m["alpha"] == 0.2
    assert m["nominal_coverage"] == pytest.approx(0.8)
    assert m["n_calibration_cells"] + m["n_test_cells"] == adata.n_obs


# -- Audit boundary -----------------------------------------------------


def test_conformal_does_not_leak_held_out_truth_to_base_adapter():
    """The wrapped base adapter must see zeros at held-out columns, just
    like an unwrapped adapter would."""
    captured: dict = {}

    class _Snoop(BaseAdapter):
        name = "snoop"

        def _impute(self, masked, inp):
            var_names = list(masked.var_names)
            cols = [var_names.index(g) for g in inp.held_out_genes if g in var_names]
            captured["seen"] = np.asarray(masked.X)[:, cols].copy()
            out = masked.copy()
            out.layers["imputed"] = np.asarray(masked.X).copy()
            return out

    adata = _make_adata(n_cells=60, seed=0)
    inp = AdapterInput(input_h5ad=adata, held_out_genes=["GENE_007"], seed=0)
    _ = ConformalCalibrator(base=_Snoop(), alpha=0.1).run(inp)

    assert np.all(captured["seen"] == 0.0), \
        "AUDIT FAILURE: conformal wrapper leaked held-out truth to base adapter"


# -- Unavailable propagation --------------------------------------------


def test_conformal_propagates_base_adapter_unavailable():
    class _Never(BaseAdapter):
        name = "never"

        def _check_available(self):
            return False, "external-tool-missing"

        def _impute(self, masked, inp):
            raise RuntimeError("unreachable")

    adata = _make_adata(seed=0)
    inp = AdapterInput(input_h5ad=adata, held_out_genes=["GENE_005"], seed=0)
    result = ConformalCalibrator(base=_Never(), alpha=0.1).run(inp)

    assert result.status.startswith("unavailable:")
    assert "external-tool-missing" in result.status


def test_conformal_works_with_real_baseline_adapter():
    """End-to-end smoke: kNN baseline wrapped in conformal."""
    adata = _make_adata(n_cells=160, n_genes=25, seed=0)
    held_out = ["GENE_005", "GENE_010"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=42)

    result = ConformalCalibrator(base=KNNAdapter(k=5), alpha=0.1).run(inp)
    assert result.status == "ok"
    assert "empirical_coverage" in result.metrics_json
    assert 0.0 <= result.metrics_json["empirical_coverage"] <= 1.0
    assert result.metrics_json["mean_interval_width"] >= 0.0
