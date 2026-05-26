"""Tests for the TISSUE, stMCDI, and CellT competitor adapters.

Each adapter is verified on two paths:
1. External library not installed → status='unavailable:...'
2. Mocked import into sys.modules → adapter executes and audit boundary holds
"""

from __future__ import annotations

import sys
import types

import anndata as ad
import numpy as np

from lumina_st.benchmarks import AdapterInput
from lumina_st.benchmarks.adapters import (
    CellTAdapter,
    STMCDIAdapter,
    TISSUEAdapter,
)


def _make_synthetic_adata(n_cells: int = 40, n_genes: int = 15, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    return adata


def _clear_modules(*names: str) -> None:
    for n in names:
        sys.modules.pop(n, None)


# -- TISSUE -----------------------------------------------------------------


def test_tissue_unavailable_when_not_installed():
    _clear_modules("tissue", "TISSUE", "tissue_sc")
    adata = _make_synthetic_adata()
    inp = AdapterInput(input_h5ad=adata, held_out_genes=["GENE_005"], seed=0)
    res = TISSUEAdapter(base_imputer=lambda m, i: m).run(inp)
    assert res.status.startswith("unavailable:"), res.status
    assert "tissue-not-installed" in res.status


def test_tissue_unavailable_when_base_imputer_missing():
    fake = types.ModuleType("tissue")
    fake.calibrate = lambda **kw: types.SimpleNamespace(
        lower=np.zeros((1, 1)), upper=np.zeros((1, 1))
    )
    sys.modules["tissue"] = fake
    try:
        adata = _make_synthetic_adata()
        inp = AdapterInput(input_h5ad=adata, held_out_genes=["GENE_005"], seed=0)
        res = TISSUEAdapter(base_imputer=None).run(inp)
        assert res.status.startswith("unavailable:"), res.status
        assert "tissue-requires-base-imputer" in res.status
    finally:
        _clear_modules("tissue")


def test_tissue_runs_with_mocked_module_and_records_intervals():
    snooped: dict = {}

    fake = types.ModuleType("tissue")

    def fake_calibrate(point, reference, alpha):
        snooped["alpha"] = alpha
        n_c, n_g = point.X.shape
        return types.SimpleNamespace(
            lower=np.full((n_c, n_g), -1.0, dtype=np.float32),
            upper=np.full((n_c, n_g), 1.0, dtype=np.float32),
        )

    fake.calibrate = fake_calibrate
    sys.modules["tissue"] = fake
    try:
        adata = _make_synthetic_adata()
        inp = AdapterInput(input_h5ad=adata, held_out_genes=["GENE_005"], seed=0)

        def base(m, i):
            out = m.copy()
            out.layers["imputed"] = np.asarray(m.X, dtype=np.float32).copy()
            return out

        res = TISSUEAdapter(base_imputer=base, alpha=0.1).run(inp)

        assert res.status == "ok", res.status
        assert res.imputed_h5ad is not None
        assert "imputed_lower" in res.imputed_h5ad.layers
        assert "imputed_upper" in res.imputed_h5ad.layers
        assert "imputed_width" in res.imputed_h5ad.layers
        assert snooped["alpha"] == 0.1
    finally:
        _clear_modules("tissue")


# -- stMCDI -----------------------------------------------------------------


def test_stmcdi_unavailable_when_not_installed():
    _clear_modules("stmcdi", "stMCDI", "st_mcdi")
    adata = _make_synthetic_adata()
    inp = AdapterInput(input_h5ad=adata, held_out_genes=["GENE_003"], seed=0)
    res = STMCDIAdapter().run(inp)
    assert res.status.startswith("unavailable:"), res.status
    assert "stmcdi-not-installed" in res.status


def test_stmcdi_runs_with_mocked_module_and_audit_holds():
    seen: dict = {}
    fake = types.ModuleType("stmcdi")

    class _FakeSTMCDI:
        def __init__(self, n_steps: int = 1000, device: str = "cpu"):
            self.n_steps = n_steps

        def fit(self, adata: ad.AnnData):
            seen["fit_X"] = np.asarray(adata.X).copy()

        def impute(self, adata: ad.AnnData) -> ad.AnnData:
            out = adata.copy()
            out.layers["imputed"] = np.asarray(adata.X, dtype=np.float32).copy()
            return out

    fake.stMCDI = _FakeSTMCDI  # type: ignore[attr-defined]
    sys.modules["stmcdi"] = fake
    try:
        adata = _make_synthetic_adata()
        held_out = ["GENE_003", "GENE_007"]
        inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)
        res = STMCDIAdapter().run(inp)

        assert res.status == "ok", res.status
        assert res.imputed_h5ad is not None
        assert "imputed" in res.imputed_h5ad.layers

        var_names = list(adata.var_names)
        idx = [var_names.index(g) for g in held_out]
        assert np.all(seen["fit_X"][:, idx] == 0.0), \
            "AUDIT FAILURE: stMCDI saw non-zero values at held-out positions"
    finally:
        _clear_modules("stmcdi")


# -- CellT ------------------------------------------------------------------


def test_cellt_unavailable_when_not_installed():
    _clear_modules("cellt", "CellT", "cell_t")
    adata = _make_synthetic_adata()
    inp = AdapterInput(input_h5ad=adata, held_out_genes=["GENE_004"], seed=0)
    res = CellTAdapter().run(inp)
    assert res.status.startswith("unavailable:"), res.status
    assert "cellt-not-installed" in res.status


def test_cellt_runs_with_mocked_module_and_audit_holds():
    seen: dict = {}
    fake = types.ModuleType("cellt")

    class _FakeCellT:
        def __init__(self, device: str = "cpu"):
            self.device = device

        def fit(self, adata: ad.AnnData):
            seen["fit_X"] = np.asarray(adata.X).copy()

        def predict(self, adata: ad.AnnData) -> ad.AnnData:
            out = adata.copy()
            out.layers["imputed"] = np.asarray(adata.X, dtype=np.float32).copy()
            return out

    fake.CellT = _FakeCellT  # type: ignore[attr-defined]
    sys.modules["cellt"] = fake
    try:
        adata = _make_synthetic_adata()
        held_out = ["GENE_004", "GENE_009"]
        inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)
        res = CellTAdapter().run(inp)

        assert res.status == "ok", res.status
        assert res.imputed_h5ad is not None
        assert "imputed" in res.imputed_h5ad.layers

        var_names = list(adata.var_names)
        idx = [var_names.index(g) for g in held_out]
        assert np.all(seen["fit_X"][:, idx] == 0.0), \
            "AUDIT FAILURE: CellT saw non-zero values at held-out positions"
    finally:
        _clear_modules("cellt")
