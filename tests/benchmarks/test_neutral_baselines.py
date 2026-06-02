"""Synthetic ranking tests for the two clean-room neutral baselines.

These pin the behaviour the harness needs: on *planted, recoverable* structure
each signal-carrying baseline must beat a trivial constant baseline under the
shared held-out-gene metric loop, and each must emit the metric contract keys.

- ``SpatialNeighborAvgAdapter`` is given a spatially smooth held-out gene over
  ``obsm['spatial']``; its neighbour average should track the truth while a
  constant fill cannot.
- ``ReferenceRegressionAdapter`` is given a held-out gene that is a linear map
  of the observed panel; its ridge fit should recover it while a constant fill
  cannot.

Both adapters are new on this branch, so this whole module fails to import (and
therefore fails) on origin/main — the required fail-to-exist guard.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from lumina_st.benchmarks import AdapterInput
from lumina_st.benchmarks.adapters import (
    ReferenceRegressionAdapter,
    SpatialNeighborAvgAdapter,
)
from lumina_st.benchmarks.contract import BaseAdapter

CONTRACT_KEYS = {
    "per_gene_pearson",
    "per_gene_spearman",
    "per_gene_rmse",
    "mean_pearson",
    "mean_spearman",
    "mean_rmse",
    "n_genes_scored",
}


class _ConstantAdapter(BaseAdapter):
    """Trivial baseline: fill every held-out gene with one global constant.

    A constant prediction has zero variance, so its per-gene correlation is
    undefined (NaN) — the canonical 'no information' floor the real baselines
    must beat.
    """

    name = "constant"

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        X = np.asarray(masked.X, dtype=np.float32).copy()
        var_names = list(masked.var_names)
        cols = [var_names.index(g) for g in inp.held_out_genes if g in var_names]
        fill = float(X.mean())
        for c in cols:
            X[:, c] = fill
        out = masked.copy()
        out.layers["imputed"] = X
        return out


def _score_floor(value: float) -> float:
    """Treat an undefined (NaN) mean correlation as the worst possible score."""
    return -1.0 if np.isnan(value) else value


def _spatial_adata(n_side: int = 12, n_observed: int = 10, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side))
    coords = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    coords += rng.normal(0.0, 0.02, size=coords.shape)
    n_cells = coords.shape[0]

    observed = rng.poisson(2.0, size=(n_cells, n_observed)).astype(np.float32)
    # Two held-out genes, each a *different* smooth function of position so the
    # predicted columns must differ per gene (issue #137 discipline).
    x = coords[:, 0]
    y = coords[:, 1]
    held = np.stack(
        [
            np.sin(x / 2.0) + 0.5 * y,
            np.cos(y / 2.0) + 0.3 * x,
        ],
        axis=1,
    ).astype(np.float32)

    X = np.concatenate([observed, held], axis=1)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"OBS_{i:03d}" for i in range(n_observed)] + ["HELD_A", "HELD_B"]
    adata.obsm["spatial"] = coords
    return adata


def _linear_adata(n_cells: int = 120, n_observed: int = 8, seed: int = 1) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    observed = rng.normal(0.0, 1.0, size=(n_cells, n_observed)).astype(np.float32)
    w_a = rng.normal(0.0, 1.0, size=n_observed)
    w_b = rng.normal(0.0, 1.0, size=n_observed)
    held_a = observed @ w_a + 0.05 * rng.normal(0.0, 1.0, size=n_cells)
    held_b = observed @ w_b + 0.05 * rng.normal(0.0, 1.0, size=n_cells)
    held = np.stack([held_a, held_b], axis=1).astype(np.float32)

    X = np.concatenate([observed, held], axis=1)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"OBS_{i:03d}" for i in range(n_observed)] + ["HELD_A", "HELD_B"]
    return adata


def _held_out_columns(imputed: ad.AnnData, held_out: list[str]) -> np.ndarray:
    var_names = list(imputed.var_names)
    cols = [var_names.index(g) for g in held_out]
    X = np.asarray(imputed.layers["imputed"])
    return X[:, cols]


def test_spatial_neighbor_avg_beats_constant_on_spatial_structure() -> None:
    adata = _spatial_adata()
    held_out = ["HELD_A", "HELD_B"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)

    spatial = SpatialNeighborAvgAdapter(k=8).run(inp)
    constant = _ConstantAdapter().run(inp)

    assert spatial.status == "ok", spatial.status
    assert constant.status == "ok", constant.status
    assert CONTRACT_KEYS.issubset(spatial.metrics_json.keys())

    spatial_mp = spatial.metrics_json["mean_pearson"]
    constant_mp = constant.metrics_json["mean_pearson"]

    # The spatial baseline recovers planted spatial autocorrelation...
    assert spatial_mp > 0.5, f"spatial mean_pearson too low: {spatial_mp}"
    # ...and strictly beats the trivial constant floor.
    assert spatial_mp > _score_floor(constant_mp)

    # Distinct per held-out gene (no per-cell-scalar collapse, issue #137).
    cols = _held_out_columns(spatial.imputed_h5ad, held_out)
    assert not np.allclose(cols[:, 0], cols[:, 1])


def test_reference_regression_beats_constant_on_linear_structure() -> None:
    adata = _linear_adata()
    held_out = ["HELD_A", "HELD_B"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)

    regression = ReferenceRegressionAdapter(alpha=1.0).run(inp)
    constant = _ConstantAdapter().run(inp)

    assert regression.status == "ok", regression.status
    assert constant.status == "ok", constant.status
    assert CONTRACT_KEYS.issubset(regression.metrics_json.keys())

    reg_mp = regression.metrics_json["mean_pearson"]
    constant_mp = constant.metrics_json["mean_pearson"]

    # The ridge map recovers the planted linear relationship...
    assert reg_mp > 0.8, f"regression mean_pearson too low: {reg_mp}"
    # ...and strictly beats the trivial constant floor.
    assert reg_mp > _score_floor(constant_mp)

    # Distinct per held-out gene.
    cols = _held_out_columns(regression.imputed_h5ad, held_out)
    assert not np.allclose(cols[:, 0], cols[:, 1])


def test_neutral_baselines_handle_empty_held_out() -> None:
    """Empty held-out set is a no-op that still returns an imputed layer."""
    adata = _linear_adata()
    inp = AdapterInput(input_h5ad=adata, held_out_genes=[], seed=0)
    for adapter in (SpatialNeighborAvgAdapter(), ReferenceRegressionAdapter()):
        res = adapter.run(inp)
        assert res.status == "ok", res.status
        assert res.imputed_h5ad is not None
        assert "imputed" in res.imputed_h5ad.layers


def test_spatial_neighbor_avg_falls_back_without_coordinates() -> None:
    """No obsm['spatial'] -> single-neighbourhood reference per-gene mean."""
    adata = _linear_adata()  # has no spatial coordinates
    held_out = ["HELD_A", "HELD_B"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=0)
    res = SpatialNeighborAvgAdapter(k=8).run(inp)
    assert res.status == "ok", res.status
    cols = _held_out_columns(res.imputed_h5ad, held_out)
    # Each gene filled with its own reference mean: constant within a gene,
    # distinct across genes.
    assert np.allclose(cols[:, 0], cols[0, 0])
    assert not np.isclose(cols[0, 0], cols[0, 1])
