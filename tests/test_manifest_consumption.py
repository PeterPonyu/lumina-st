"""Tests: manifest emission from benchmark runners and sweep scripts (#131 / #179).

Verifies that each runner / sweep script writes a valid ``run_manifest.json``
beside its results JSON and that the manifest round-trips via
``RunManifest.from_json``.  All tests use tiny synthetic inputs and run fully
on CPU.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np

from lumina_st.benchmarks import (
    aggregate_results,
    get_panel,
    run_panel,
    write_results_json,
)
from lumina_st.benchmarks.adapters import KNNAdapter, MeanAdapter
from lumina_st.experiment import RunManifest

# Repository root (two levels up from this file which is at tests/)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_CI = _REPO_ROOT / "scripts" / "ci"


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _load_script(name: str) -> types.ModuleType:
    """Load a scripts/ci/<name>.py module by file path, bypassing package import."""
    path = _SCRIPTS_CI / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _make_tiny_adata(n_cells: int = 30, seed: int = 0) -> ad.AnnData:
    """Minimal AnnData with the marker genes the panel expects."""
    rng = np.random.default_rng(seed)
    base_genes = [f"GENE_{i:03d}" for i in range(20)]
    marker_genes = list(get_panel("tme-immune-stromal").genes)
    all_genes = base_genes + marker_genes
    X = rng.poisson(2.0, size=(n_cells, len(all_genes))).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = all_genes
    adata.obs["cancer_type"] = ["COAD"] * n_cells
    return adata


def _assert_valid_manifest(manifest_path: Path) -> RunManifest:
    """Assert file exists, parses cleanly, and round-trips; return the object."""
    assert manifest_path.exists(), f"run_manifest.json not found at {manifest_path}"
    m = RunManifest.from_json(manifest_path)
    # required identity fields
    assert isinstance(m.run_id, str) and m.run_id
    assert isinstance(m.timestamp, str) and m.timestamp
    assert isinstance(m.git_commit, str) and m.git_commit
    assert isinstance(m.manifest_version, str)
    # round-trip: serialise → re-parse → equal
    raw = json.loads(manifest_path.read_text())
    assert RunManifest(**raw) == m
    return m


# ---------------------------------------------------------------------------
# write_results_json emits manifest (runner path)
# ---------------------------------------------------------------------------

def test_write_results_json_emits_manifest(tmp_path: Path) -> None:
    """write_results_json writes run_manifest.json beside the results file."""
    adata = _make_tiny_adata()
    panel = get_panel("tme-immune-stromal")
    results = run_panel([MeanAdapter(), KNNAdapter(k=5)], adata, panel, seed=7)
    aggregated = aggregate_results({("synthetic-coad", panel.name): results})

    out = write_results_json(
        aggregated,
        tmp_path / "results.json",
        seed=7,
        dataset_id="synthetic-coad",
        sweep_params={"panel": panel.name},
    )

    assert out.exists()
    manifest_path = tmp_path / "run_manifest.json"
    m = _assert_valid_manifest(manifest_path)
    assert m.seed == 7
    assert m.dataset_id == "synthetic-coad"
    assert m.sweep_params.get("panel") == panel.name


def test_write_results_json_emit_manifest_false(tmp_path: Path) -> None:
    """Opt-out via emit_manifest=False must not produce run_manifest.json."""
    adata = _make_tiny_adata()
    panel = get_panel("tme-immune-stromal")
    results = run_panel([MeanAdapter()], adata, panel, seed=0)
    aggregated = aggregate_results({("synthetic-coad", panel.name): results})

    write_results_json(
        aggregated,
        tmp_path / "results.json",
        emit_manifest=False,
    )

    assert not (tmp_path / "run_manifest.json").exists()


# ---------------------------------------------------------------------------
# run_synthetic_benchmark main() path
# ---------------------------------------------------------------------------

def test_synthetic_benchmark_main_emits_manifest(tmp_path: Path) -> None:
    """run_synthetic_benchmark.main() writes run_manifest.json beside results."""
    out_json = tmp_path / "synthetic_smoke.json"
    old_argv = sys.argv
    sys.argv = [
        "run_synthetic_benchmark",
        "--out", str(out_json),
        "--seed", "3",
        "--n-cells", "30",
    ]
    try:
        mod = _load_script("run_synthetic_benchmark")
        rc = mod.main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    assert out_json.exists()
    manifest_path = out_json.parent / "run_manifest.json"
    m = _assert_valid_manifest(manifest_path)
    assert m.seed == 3
    assert m.dataset_id == "synthetic-coad-smoke"


# ---------------------------------------------------------------------------
# run_sparsity_sweep main() path
# ---------------------------------------------------------------------------

def test_sparsity_sweep_main_emits_manifest(tmp_path: Path) -> None:
    """run_sparsity_sweep.main() writes run_manifest.json beside results."""
    out_json = tmp_path / "sparsity_sweep.json"
    old_argv = sys.argv
    sys.argv = [
        "run_sparsity_sweep",
        "--out", str(out_json),
        "--seed", "5",
        "--n-cells", "40",
    ]
    try:
        mod = _load_script("run_sparsity_sweep")
        rc = mod.main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    assert out_json.exists()
    manifest_path = out_json.parent / "run_manifest.json"
    m = _assert_valid_manifest(manifest_path)
    assert m.seed == 5
    assert m.dataset_id == "synthetic-coad-sparsity"
    assert "fractions" in m.sweep_params


# ---------------------------------------------------------------------------
# run_leave_one_context main() path
# ---------------------------------------------------------------------------

def test_leave_one_context_main_emits_manifest(tmp_path: Path) -> None:
    """run_leave_one_context.main() writes run_manifest.json beside results."""
    out_json = tmp_path / "leave_one_context.json"
    old_argv = sys.argv
    sys.argv = [
        "run_leave_one_context",
        "--out", str(out_json),
        "--seed", "9",
        "--n-cells", "30",
        "--n-boot", "50",
        "--contexts", "COAD", "OV",
    ]
    try:
        mod = _load_script("run_leave_one_context")
        rc = mod.main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    assert out_json.exists()
    manifest_path = out_json.parent / "run_manifest.json"
    m = _assert_valid_manifest(manifest_path)
    assert m.seed == 9
    assert m.dataset_id == "synthetic-loo-cv"
    assert "contexts" in m.sweep_params
