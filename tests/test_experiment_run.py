"""Tests for the unified experiment driver (#179 capstone smoke).

Runs one full experiment through ``run_experiment`` and asserts the emitted
manifest is valid (exists, round-trips, carries seed/config/sweep/eval), the
metrics sidecar is written, and that ``set_seed`` makes RNG draws deterministic
across two context entries with the same seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lumina_st.config import LuminaSTConfig
from lumina_st.experiment import ExperimentRun, RunManifest, run_experiment

_FIXED_TS = "2026-01-02T03:04:05+00:00"


def _tiny_config(seed: int = 123) -> LuminaSTConfig:
    return LuminaSTConfig(
        latent_dim=5,
        cancer_types=["COAD"],
        num_sampling_steps=1,
        seed=seed,
    )


def test_run_experiment_emits_valid_manifest(tmp_path: Path) -> None:
    cfg = _tiny_config(seed=123)
    sweep = {"lr": 1e-4, "depth": 2}
    cadence = {"every_n_epochs": 3}

    with run_experiment(
        config=cfg,
        run_dir=tmp_path / "run0",
        sweep_params=sweep,
        eval_cadence=cadence,
        dataset_id="coad_v1",
        logging_config={"use_wandb": False},
        timestamp=_FIXED_TS,
    ) as run:
        assert isinstance(run, ExperimentRun)
        assert run.run_dir == tmp_path / "run0"
        assert run.seed == 123
        # Manifest is written on ENTRY (crash-safe), before the body finishes.
        assert run.manifest_path.exists()
        # Trivial body, then finalize with metrics.
        run.record_metrics({"loss": 0.5})
        run.finalize(metrics={"ari": 0.42})

    # Manifest exists and round-trips.
    manifest_path = run.run_dir / "run_manifest.json"
    assert manifest_path.exists()
    loaded = RunManifest.from_json(manifest_path)
    assert loaded.seed == 123
    assert loaded.timestamp == _FIXED_TS
    assert loaded.config["seed"] == 123
    assert loaded.config["latent_dim"] == 5
    assert loaded.sweep_params == sweep
    assert loaded.eval_cadence == cadence
    assert loaded.dataset_id == "coad_v1"

    # Metrics sidecar (RunManifest schema stays frozen) carries metrics+wallclock.
    metrics_path = run.run_dir / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["metrics"] == {"loss": 0.5, "ari": 0.42}
    assert payload["seed"] == 123
    assert isinstance(payload["wallclock_s"], float) and payload["wallclock_s"] >= 0.0


def test_run_experiment_finalizes_on_context_exit(tmp_path: Path) -> None:
    cfg = _tiny_config(seed=7)
    with run_experiment(config=cfg, run_dir=tmp_path / "auto", timestamp=_FIXED_TS) as run:
        run.record_metrics({"acc": 1.0})
        # No explicit finalize() — context exit must flush the sidecar.
    payload = json.loads((tmp_path / "auto" / "metrics.json").read_text())
    assert payload["metrics"] == {"acc": 1.0}


def test_seed_makes_rng_draws_deterministic(tmp_path: Path) -> None:
    cfg = _tiny_config(seed=2024)

    with run_experiment(config=cfg, run_dir=tmp_path / "a", timestamp=_FIXED_TS):
        torch_a = torch.rand(4)
        np_a = np.random.rand(4)

    with run_experiment(config=cfg, run_dir=tmp_path / "b", timestamp=_FIXED_TS):
        torch_b = torch.rand(4)
        np_b = np.random.rand(4)

    assert torch.equal(torch_a, torch_b)
    assert np.array_equal(np_a, np_b)


def test_default_run_dir_under_config_output_dir(tmp_path: Path) -> None:
    cfg = _tiny_config(seed=1)
    cfg.output_dir = tmp_path / "out"
    cfg.experiment_name = "smoke179"
    with run_experiment(config=cfg, timestamp=_FIXED_TS) as run:
        assert run.run_dir == tmp_path / "out" / "smoke179"
        assert run.manifest_path.exists()
