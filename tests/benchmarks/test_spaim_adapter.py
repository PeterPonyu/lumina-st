"""Tests for the SpaIM adapter.

Covers both paths:
1. SpaIM is not installed → adapter returns status='unavailable:...'
2. SpaIM is mocked into sys.modules → adapter returns a valid result, and
   the audit boundary still holds (the mocked SpaIM only sees zeroed held-out
   columns).
"""

from __future__ import annotations

import sys
import types

import anndata as ad
import numpy as np

from lumina_st.benchmarks import AdapterInput
from lumina_st.benchmarks.adapters import SpaIMAdapter


def _make_synthetic_adata(n_cells: int = 50, n_genes: int = 20, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    obs = {"cancer_type": ["COAD"] * n_cells}
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    return adata


def test_spaim_adapter_reports_unavailable_when_not_installed():
    # Ensure SpaIM is not in sys.modules from a previous test.
    for name in ("SpaIM", "spaim"):
        sys.modules.pop(name, None)

    adata = _make_synthetic_adata()
    inp = AdapterInput(input_h5ad=adata, held_out_genes=["GENE_005"], seed=0)
    ref = _make_synthetic_adata(seed=1)

    result = SpaIMAdapter(reference_adata=ref).run(inp)

    assert result.status.startswith("unavailable:"), result.status
    assert "spaim-not-installed" in result.status
    assert result.imputed_h5ad is None


def test_spaim_adapter_unavailable_when_reference_missing():
    """Even if SpaIM is importable, missing reference data is fatal."""
    fake_module = types.ModuleType("SpaIM")

    class _FakeSpaIM:
        def __init__(self, device: str = "cpu"):
            self.device = device

        def fit(self, reference, spatial):
            return self

        def predict(self, spatial):
            return spatial

    fake_module.SpaIM = _FakeSpaIM  # type: ignore[attr-defined]
    sys.modules["SpaIM"] = fake_module
    try:
        adata = _make_synthetic_adata()
        inp = AdapterInput(input_h5ad=adata, held_out_genes=["GENE_005"], seed=0)
        result = SpaIMAdapter(reference_adata=None).run(inp)

        assert result.status.startswith("unavailable:"), result.status
        assert "spaim-requires-reference" in result.status
    finally:
        sys.modules.pop("SpaIM", None)


def test_spaim_adapter_runs_when_module_is_mocked_and_audit_holds():
    """End-to-end run with a mocked SpaIM module. Verifies result schema and audit."""
    saw_held_out_values = {}

    fake_module = types.ModuleType("SpaIM")

    class _FakeSpaIM:
        def __init__(self, device: str = "cpu"):
            self.device = device

        def fit(self, reference: ad.AnnData, spatial: ad.AnnData):
            # Audit check: when SpaIM sees the spatial matrix, the held-out
            # columns must already be zero.
            saw_held_out_values["fit_spatial_X"] = np.asarray(spatial.X).copy()
            return self

        def predict(self, spatial: ad.AnnData):
            # Return per-cell mean over observed columns as a trivial prediction.
            X = np.asarray(spatial.X, dtype=np.float32)
            per_cell = X.mean(axis=1, keepdims=True)
            out = spatial.copy()
            out.layers["imputed"] = np.broadcast_to(per_cell, X.shape).copy()
            return out

    fake_module.SpaIM = _FakeSpaIM  # type: ignore[attr-defined]
    sys.modules["SpaIM"] = fake_module
    try:
        adata = _make_synthetic_adata(n_cells=40, n_genes=15)
        held_out = ["GENE_003", "GENE_007"]
        inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=42)
        ref = _make_synthetic_adata(seed=99)

        result = SpaIMAdapter(reference_adata=ref).run(inp)

        assert result.status == "ok", result.status
        assert result.method == "spaim"
        assert result.imputed_h5ad is not None
        assert "imputed" in result.imputed_h5ad.layers
        assert result.imputed_h5ad.layers["imputed"].shape == adata.X.shape

        # Audit boundary: the held-out columns in the matrix SpaIM saw must be zero
        seen = saw_held_out_values["fit_spatial_X"]
        var_names = list(adata.var_names)
        idx = [var_names.index(g) for g in held_out]
        assert np.all(seen[:, idx] == 0.0), \
            "AUDIT FAILURE: SpaIM observed non-zero values at held-out positions"

        # Metrics populated
        assert result.metrics_json["n_genes_scored"] == len(held_out)
        assert result.metrics_json["n_cells"] == adata.n_obs
    finally:
        sys.modules.pop("SpaIM", None)
