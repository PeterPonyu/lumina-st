"""Tests for the experiment-harness RunManifest contract (#179 / #181)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from lumina_st.config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer
from lumina_st.experiment import RunManifest

_FIXED_TS = "2026-01-02T03:04:05+00:00"

# Required manifest fields and their expected (JSON-friendly) types.
_REQUIRED_FIELDS = {
    "run_id": str,
    "timestamp": str,
    "git_commit": str,
    "seed": int,
    "seeds": dict,
    "python_version": str,
    "torch_version": str,
    "numpy_version": str,
    "device": str,
    "config": dict,
    "sweep_params": dict,
    "logging_config": dict,
    "eval_cadence": dict,
    "manifest_version": str,
}


def _tiny_st_adata(n_cells: int = 3, n_genes: int = 5) -> ad.AnnData:
    """Minimal in-memory AnnData matching the no-VAE enhance fixtures."""
    st = ad.AnnData(
        np.ones((n_cells, n_genes), dtype="float32"),
        obs=pd.DataFrame({"cancer_type": ["COAD"] * n_cells}),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    st.obsm["spatial"] = np.zeros((n_cells, 2), dtype="float32")
    return st


def test_create_honors_injected_timestamp_and_seed() -> None:
    m = RunManifest.create(run_id="r1", seed=123, timestamp=_FIXED_TS)
    assert m.timestamp == _FIXED_TS
    assert m.seed == 123
    assert m.seeds == {"global": 123}


def test_create_defaults_seed_from_config() -> None:
    cfg = LuminaSTConfig(seed=77)
    m = RunManifest.create(run_id="r", config=cfg, timestamp=_FIXED_TS)
    assert m.seed == 77
    assert m.config["seed"] == 77
    # config snapshot is JSON-serializable (Paths coerced to str)
    json.dumps(m.config)


def test_required_fields_present_and_typed() -> None:
    cfg = LuminaSTConfig(seed=5)
    m = RunManifest.create(run_id="r", config=cfg, seed=5, timestamp=_FIXED_TS)
    d = m.to_dict()
    for field_name, expected_type in _REQUIRED_FIELDS.items():
        assert field_name in d, f"missing required field {field_name!r}"
        assert isinstance(d[field_name], expected_type), (
            f"{field_name} is {type(d[field_name])}, expected {expected_type}"
        )


def test_to_json_from_json_round_trip(tmp_path: Path) -> None:
    cfg = LuminaSTConfig(seed=11, experiment_name="rt_exp")
    m = RunManifest.create(
        run_id="rt",
        config=cfg,
        seed=11,
        timestamp=_FIXED_TS,
        sweep_params={"lr": 1e-4},
        logging_config={"use_wandb": False},
        checkpoint_path="ckpts/best.ckpt",
        resume_from=None,
        eval_cadence={"every_n_epochs": 2},
        dataset_id="coad_v1",
        dataset_hash=None,
    )
    path = m.to_json(tmp_path / "run_manifest.json")
    assert path.exists()

    loaded = RunManifest.from_json(path)
    assert loaded == m
    assert loaded.to_dict() == m.to_dict()


def test_git_commit_is_best_effort_string() -> None:
    m = RunManifest.create(run_id="g", timestamp=_FIXED_TS)
    # Either a resolved sha or one of the tolerated sentinels; never raises.
    assert isinstance(m.git_commit, str) and m.git_commit


def test_enhance_emits_run_manifest(tmp_path: Path) -> None:
    cfg = LuminaSTConfig(latent_dim=5, cancer_types=["COAD"], num_sampling_steps=1)
    imputer = LuminaImputer.from_config(cfg)
    st = _tiny_st_adata(n_cells=3, n_genes=5)

    out = imputer.enhance(
        st, run_dir=tmp_path, run_manifest_timestamp=_FIXED_TS
    )
    # Existing behaviour preserved.
    assert "latent_enhanced" in out.obsm

    manifest_path = tmp_path / "run_manifest.json"
    assert manifest_path.exists()

    m = RunManifest.from_json(manifest_path)
    assert m.timestamp == _FIXED_TS
    assert m.seed == cfg.seed
    assert m.config["latent_dim"] == 5
    assert m.sweep_params["run_kind"] == "enhance"
    assert m.sweep_params["n_obs"] == 3
    for field_name in _REQUIRED_FIELDS:
        assert field_name in m.to_dict()


def test_enhance_without_run_dir_emits_nothing(tmp_path: Path) -> None:
    cfg = LuminaSTConfig(latent_dim=5, cancer_types=["COAD"], num_sampling_steps=1)
    imputer = LuminaImputer.from_config(cfg)
    st = _tiny_st_adata(n_cells=3, n_genes=5)

    out = imputer.enhance(st)
    assert "latent_enhanced" in out.obsm
    assert not (tmp_path / "run_manifest.json").exists()
