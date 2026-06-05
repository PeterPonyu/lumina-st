"""Tests for the latent vs ambient (gene-space) diffusion ablation (issue #138).

These run on CPU with tiny dims in well under a second of model work. They
assert the harness wires both encoder seams end-to-end through ``enhance`` and
emits a benchmark-schema JSON + RunManifest sidecar with one row per mode.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

_CI_DIR = Path(__file__).resolve().parents[1] / "scripts" / "ci"


def _load_script(name: str) -> ModuleType:
    """Load a ``scripts/ci`` harness by file path (not an importable package)."""
    path = _CI_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lva = _load_script("run_latent_vs_ambient_ablation.py")


def test_run_ablation_both_modes_produce_metric_rows():
    rows = lva.run_ablation(
        n_genes=12, latent_dim=4, n_cells=16, vae_pretrain_steps=2, seed=0
    )
    assert len(rows) == 2
    by_mode = {r["mode"]: r for r in rows}
    assert set(by_mode) == {"latent", "ambient"}

    for mode, row in by_mode.items():
        # Both modes are expected to run end-to-end on synthetic data.
        assert row["status"] == "ok", f"{mode} did not run: {row['status']}"
        # Wall-clock fields present and non-negative.
        assert row["inference_s"] >= 0.0
        assert row["train_time_s"] >= 0.0
        assert row["runtime_s"] >= 0.0
        # Memory proxy.
        assert isinstance(row["param_count"], int) and row["param_count"] > 0
        # Held-out gene metric keys present.
        m = row["metrics_json"]
        for key in ("mean_pearson", "mean_spearman", "mean_rmse", "n_genes_scored"):
            assert key in m, f"{mode} missing metric {key}"
        assert m["n_genes_scored"] == 3

    # The two modes differ in where the flow lives: latent compresses to a
    # smaller working dimension, ambient runs at full gene width.
    assert by_mode["latent"]["flow_dim"] == 4
    assert by_mode["ambient"]["flow_dim"] == 12


def test_main_emits_schema_valid_json_and_manifest(tmp_path):
    out = tmp_path / "lva.json"
    rc = lva.main(
        [
            "--out",
            str(out),
            "--n-genes",
            "12",
            "--latent-dim",
            "4",
            "--n-cells",
            "16",
            "--vae-pretrain-steps",
            "2",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    assert out.exists()

    data = json.loads(out.read_text())
    assert data["ablation"] == "latent_vs_ambient"
    assert data["n_rows"] == 2
    assert data["held_out_genes"] == ["CD8A", "EPCAM", "POSTN"]
    assert {r["mode"] for r in data["rows"]} == {"latent", "ambient"}
    for r in data["rows"]:
        assert "metrics_json" in r
        assert "inference_s" in r
        assert "param_count" in r

    manifest_path = out.with_name(out.stem + ".manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sweep_params"]["mode"] == ["latent", "ambient"]
    assert manifest["seed"] == 0


def test_unavailable_status_path_is_recorded_not_faked(monkeypatch):
    """If a mode genuinely cannot run, its row records a clear unavailable
    status with empty metrics rather than fabricated numbers."""

    class _Boom(Exception):
        pass

    def _explode(self, *args, **kwargs):
        raise _Boom("synthetic blocker")

    monkeypatch.setattr(lva.LuminaImputer, "enhance", _explode)
    rows = lva.run_ablation(
        n_genes=12, latent_dim=4, n_cells=16, vae_pretrain_steps=0, modes=("latent",), seed=0
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["status"].startswith("unavailable:")
    assert row["metrics_json"] == {}
    assert row["inference_s"] >= 0.0
