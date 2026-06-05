"""Tests for the prediction-parameterization ablation harness (issue #139).

Runs on a TINY synthetic problem with an untrained tiny model on CPU.  The
contract under test is the *harness* (it runs every grid point and writes a
schema-valid JSON with NaN-incidence fields), not the metric values.  The script
lives under ``scripts/ci`` (not an importable package), so it is loaded by file
path following the same pattern as ``test_ablation_scripts.py``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_CI_DIR = Path(__file__).resolve().parents[1] / "scripts" / "ci"


def _load_script(name: str) -> ModuleType:
    path = _CI_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prediction_mod() -> ModuleType:
    return _load_script("run_prediction_ablation.py")


def test_prediction_ablation_writes_schema_valid_json(prediction_mod, tmp_path) -> None:
    written = tmp_path / "prediction.json"
    # Tiny grid: all three prediction targets, small latent dim and cell count.
    rc = prediction_mod.main(
        [
            "--out",
            str(written),
            "--seed",
            "0",
            "--latent-dim",
            "8",
            "--n-cells",
            "12",
            "--predictions",
            "velocity",
            "noise",
            "score",
        ]
    )
    assert rc == 0
    assert written.is_file()

    payload = json.loads(written.read_text())
    assert payload["schema_version"] == "1"
    assert payload["ablation"] == "prediction_param"
    assert payload["n_rows"] == 3  # one row per prediction target
    rows = payload["rows"]
    assert len(rows) == 3

    predictions_seen = set()
    for row in rows:
        assert "prediction" in row
        assert "path_type" in row
        assert row["path_type"] == "linear"
        assert "status" in row
        assert "runtime_s" in row
        assert isinstance(row["runtime_s"], float)
        assert row["runtime_s"] >= 0.0
        # NaN-incidence fields must be present on every row (issue #139).
        assert "n_nonfinite" in row, "n_nonfinite field required by issue #139"
        assert "nan_incidence" in row, "nan_incidence field required by issue #139"
        assert "metrics_json" in row
        if row["status"] == "ok":
            assert "mean_pearson" in row["metrics_json"]
            assert "mean_spearman" in row["metrics_json"]
            assert "mean_rmse" in row["metrics_json"]
            # n_nonfinite must be a non-negative integer for successful rows.
            assert isinstance(row["n_nonfinite"], int)
            assert row["n_nonfinite"] >= 0
        predictions_seen.add(row["prediction"])

    assert predictions_seen == {"velocity", "noise", "score"}, (
        "All three prediction targets must appear as rows"
    )

    # Manifest with sweep_params is emitted alongside the JSON.
    manifest = written.with_name("prediction.manifest.json")
    assert manifest.is_file()
    mdata = json.loads(manifest.read_text())
    assert "prediction" in mdata["sweep_params"]
    assert set(mdata["sweep_params"]["prediction"]) == {"velocity", "noise", "score"}


def test_prediction_ablation_subset_grid(prediction_mod, tmp_path) -> None:
    """Passing a single prediction target produces exactly one row."""
    written = tmp_path / "pred_single.json"
    rc = prediction_mod.main(
        [
            "--out",
            str(written),
            "--seed",
            "1",
            "--latent-dim",
            "8",
            "--n-cells",
            "8",
            "--predictions",
            "velocity",
        ]
    )
    assert rc == 0
    payload = json.loads(written.read_text())
    assert payload["n_rows"] == 1
    row = payload["rows"][0]
    assert row["prediction"] == "velocity"
    assert "n_nonfinite" in row
    assert "nan_incidence" in row


def test_run_ablation_direct(prediction_mod) -> None:
    """Call run_ablation() directly; verify schema without writing to disk."""
    rows = prediction_mod.run_ablation(
        predictions=("velocity", "noise"),
        latent_dim=8,
        n_cells=8,
        seed=0,
    )
    assert len(rows) == 2
    for row in rows:
        assert "prediction" in row
        assert "n_nonfinite" in row, "n_nonfinite must be present (issue #139)"
        assert "nan_incidence" in row, "nan_incidence must be present (issue #139)"
        assert "status" in row
        assert "metrics_json" in row
