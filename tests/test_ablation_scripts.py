"""Tests for the CFG/guidance (#143) and sampling-time (#144) ablation harnesses.

These run on a TINY synthetic problem with an untrained tiny model on CPU; the
contract under test is the *harness* (it runs every grid point and writes a
schema-valid JSON), not the metric values. The scripts live under ``scripts/ci``
(not an importable package), so they are loaded by file path.
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
def guidance_mod() -> ModuleType:
    return _load_script("run_guidance_ablation.py")


@pytest.fixture(scope="module")
def sampling_mod() -> ModuleType:
    return _load_script("run_sampling_ablation.py")


def test_guidance_ablation_writes_schema_valid_json(guidance_mod, tmp_path) -> None:
    written = tmp_path / "guidance.json"
    # Tiny 2x2 grid (incl. the guidance_scale=1.0 CFG-off baseline).
    rc = guidance_mod.main(
        [
            "--out",
            str(written),
            "--seed",
            "0",
            "--latent-dim",
            "8",
            "--n-cells",
            "12",
            "--guidance-scales",
            "1.0",
            "2.0",
            "--cfg-decays",
            "none",
            "linear",
        ]
    )
    assert rc == 0
    assert written.is_file()

    payload = json.loads(written.read_text())
    assert payload["schema_version"] == "1"
    assert payload["ablation"] == "guidance_cfg"
    assert payload["n_rows"] == 4  # 2 scales x 2 decays
    rows = payload["rows"]
    assert len(rows) == 4

    seen_baseline = False
    for row in rows:
        assert "guidance_scale" in row
        assert "cfg_decay" in row
        assert "status" in row
        assert "metrics_json" in row
        if row["guidance_scale"] == 1.0:
            seen_baseline = True
            assert row["cfg_off_baseline"] is True
        if row["status"] == "ok":
            assert "mean_pearson" in row["metrics_json"]
            assert "mean_rmse" in row["metrics_json"]
    assert seen_baseline, "guidance_scale=1.0 CFG-off baseline row must be present"

    # Manifest with sweep_params is emitted alongside the JSON.
    manifest = written.with_name("guidance.manifest.json")
    assert manifest.is_file()
    mdata = json.loads(manifest.read_text())
    assert "guidance_scale" in mdata["sweep_params"]
    assert "cfg_decay" in mdata["sweep_params"]


def test_sampling_ablation_writes_schema_valid_json(sampling_mod, tmp_path) -> None:
    written = tmp_path / "sampling.json"
    rc = sampling_mod.main(
        [
            "--out",
            str(written),
            "--seed",
            "0",
            "--latent-dim",
            "8",
            "--n-cells",
            "12",
            "--sampling-methods",
            "euler",
            "heun",
            "--num-steps",
            "4",
            "--t-forwards",
            "0.8",
            "0.9",
        ]
    )
    assert rc == 0
    assert written.is_file()

    payload = json.loads(written.read_text())
    assert payload["schema_version"] == "1"
    assert payload["ablation"] == "sampling_time"
    assert payload["n_rows"] == 4  # 2 methods x 1 step x 2 t_forwards
    rows = payload["rows"]
    assert len(rows) == 4

    for row in rows:
        assert "sampling_method" in row
        assert "num_sampling_steps" in row
        assert "t_forward" in row
        assert "status" in row
        # Per-call runtime is the defining deliverable of the sampling ablation.
        assert "runtime_s" in row
        assert isinstance(row["runtime_s"], float)
        assert row["runtime_s"] >= 0.0
        if row["status"] == "ok":
            assert "mean_pearson" in row["metrics_json"]
            assert "mean_rmse" in row["metrics_json"]

    manifest = written.with_name("sampling.manifest.json")
    assert manifest.is_file()
    mdata = json.loads(manifest.read_text())
    assert "sampling_method" in mdata["sweep_params"]
    assert "num_sampling_steps" in mdata["sweep_params"]
    assert "t_forward" in mdata["sweep_params"]
