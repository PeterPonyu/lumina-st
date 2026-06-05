"""Protocol gates for held-out parity, encoder leakage, and task boundaries.

Covers GitHub issues #307 (held-out protocol parity + encoder-leakage guard,
enforced centrally in the runner/contract layer) and #309 (task-boundary
separation: denoising / pathway-aggregate results can never be silently scored
or reported as gene recovery).

These gates are validatable on synthetic data now, before real data, so the
real-data numbers are trustworthy and comparable the moment they land.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from lumina_st.benchmarks.contract import (
    AdapterInput,
    BaseAdapter,
    EncoderLeakageError,
    ProtocolParityError,
    TaskBoundaryError,
    TaskType,
    compute_imputation_metrics,
)
from lumina_st.benchmarks.panels import get_panel
from lumina_st.benchmarks.runner import enforce_protocol_parity, run_panel
from lumina_st.benchmarks.cross_validation import run_cv, stateless_factory


# -- fixtures ---------------------------------------------------------


def _make_adata(n_cells: int = 40, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    base = [f"GENE_{i:03d}" for i in range(20)]
    markers = list(get_panel("tme-immune-stromal").genes)
    all_genes = base + markers
    X = rng.poisson(2.0, size=(n_cells, len(all_genes))).astype(np.float32) + 1.0
    adata = ad.AnnData(X=X)
    adata.var_names = all_genes
    adata.obs["cancer_type"] = ["COAD"] * n_cells
    return adata


class _IdentityImputer(BaseAdapter):
    """Echoes the masked input back as the imputed layer (gene-recovery task)."""

    name = "identity"

    def _impute(self, masked, inp):
        out = masked.copy()
        X = np.asarray(masked.X if not hasattr(masked.X, "toarray") else masked.X.toarray())
        out.layers["imputed"] = X.copy()
        return out


class _DenoisingAdapter(_IdentityImputer):
    name = "denoiser"
    task_type = TaskType.DENOISING


class _PathwayAdapter(_IdentityImputer):
    name = "pathway"
    task_type = TaskType.PATHWAY_AGGREGATE


# -- #309 task-boundary separation ------------------------------------


def test_default_task_type_is_gene_recovery():
    assert AdapterInput(input_h5ad=_make_adata()).task_type is TaskType.GENE_RECOVERY
    assert _IdentityImputer.task_type is TaskType.GENE_RECOVERY


def test_scorer_refuses_non_gene_recovery_task():
    """compute_imputation_metrics is the single gene-recovery chokepoint (#309)."""
    adata = _make_adata()
    out = adata.copy()
    out.layers["imputed"] = np.asarray(adata.X).copy()
    held = list(get_panel("tme-immune-stromal").genes)
    truth = np.asarray(adata.X)

    for task in (TaskType.DENOISING, TaskType.PATHWAY_AGGREGATE):
        with pytest.raises(TaskBoundaryError, match="NOT gene recovery"):
            compute_imputation_metrics(truth, out, held, task_type=task)


def test_adapter_run_rejects_task_mismatch_between_input_and_adapter():
    adata = _make_adata()
    inp = AdapterInput(
        input_h5ad=adata,
        held_out_genes=list(get_panel("tme-immune-stromal").genes),
        task_type=TaskType.DENOISING,
    )
    # Adapter declares GENE_RECOVERY, input asks for DENOISING -> fail closed.
    with pytest.raises(TaskBoundaryError):
        _IdentityImputer().run(inp)


def test_run_panel_rejects_denoising_adapter():
    adata = _make_adata()
    panel = get_panel("tme-immune-stromal")
    with pytest.raises(TaskBoundaryError, match="do not produce gene-recovery"):
        run_panel([_IdentityImputer(), _DenoisingAdapter()], adata, panel)


def test_run_cv_rejects_pathway_aggregate_factory():
    contexts = {"A": _make_adata(seed=0), "B": _make_adata(seed=1)}
    panel = get_panel("tme-immune-stromal")
    with pytest.raises(TaskBoundaryError):
        run_cv(contexts, stateless_factory(_PathwayAdapter), panel)


def test_gene_recovery_result_is_tagged_in_metrics():
    """A scored gene-recovery row carries its task tag so it can't be misread."""
    adata = _make_adata()
    panel = get_panel("tme-immune-stromal")
    results = run_panel([_IdentityImputer()], adata, panel)
    assert results[0].metrics_json["task_type"] == TaskType.GENE_RECOVERY.value


# -- #307 encoder-leakage guard ---------------------------------------


def test_leakage_guard_passes_when_held_out_genes_present_and_maskable():
    adata = _make_adata()
    inp = AdapterInput(
        input_h5ad=adata,
        held_out_genes=list(get_panel("tme-immune-stromal").genes),
    )
    inp.assert_no_encoder_leakage()  # must not raise


def test_leakage_guard_fails_when_held_out_gene_absent_from_var_names():
    adata = _make_adata()
    inp = AdapterInput(input_h5ad=adata, held_out_genes=["NOT_A_GENE"])
    with pytest.raises(EncoderLeakageError, match="var_names"):
        inp.assert_no_encoder_leakage()


def test_leakage_guard_fails_when_observed_layer_missing():
    """If observed_layer is named but absent, the adapter would read an unmasked
    layer -> structural leakage path -> fail closed (#307)."""
    adata = _make_adata()
    inp = AdapterInput(
        input_h5ad=adata,
        held_out_genes=list(get_panel("tme-immune-stromal").genes),
        observed_layer="counts_that_do_not_exist",
    )
    with pytest.raises(EncoderLeakageError):
        inp.assert_no_encoder_leakage()


def test_leakage_guard_runs_inside_run_panel():
    """The runner fails closed before any adapter runs when a held-out gene is
    structurally unmaskable."""
    adata = _make_adata()
    # Build a panel-like input by hand via run_panel with a bad held-out gene is
    # not possible (run_panel derives genes from the panel), so exercise the
    # central enforcer directly with a leaky input.
    inp = AdapterInput(input_h5ad=adata, held_out_genes=["GHOST_GENE"])
    with pytest.raises(EncoderLeakageError):
        enforce_protocol_parity([_IdentityImputer()], inp)


# -- #307 held-out protocol parity ------------------------------------


def test_protocol_parity_requires_non_empty_held_out_set():
    adata = _make_adata()
    inp = AdapterInput(input_h5ad=adata, held_out_genes=[])
    with pytest.raises(ProtocolParityError):
        enforce_protocol_parity([_IdentityImputer()], inp)


def test_protocol_signature_is_order_insensitive_and_identical_across_adapters():
    """Parity is centralized: all adapters see one signature derived from the
    single shared input (#307)."""
    adata = _make_adata()
    genes = list(get_panel("tme-immune-stromal").genes)
    a = AdapterInput(input_h5ad=adata, held_out_genes=genes)
    b = AdapterInput(input_h5ad=adata, held_out_genes=list(reversed(genes)))
    assert a.protocol_signature() == b.protocol_signature()


def test_run_panel_compares_all_adapters_on_identical_protocol():
    """Every adapter is scored on the same held-out split / scoring genes (#307)."""
    adata = _make_adata()
    panel = get_panel("tme-immune-stromal")
    results = run_panel([_IdentityImputer(), _make_second_identity()], adata, panel)
    assert {r.method for r in results} == {"identity", "identity2"}
    scored = {r.metrics_json["n_genes_scored"] for r in results}
    assert scored == {len(panel.genes)}


def _make_second_identity() -> BaseAdapter:
    class _Identity2(_IdentityImputer):
        name = "identity2"

    return _Identity2()
