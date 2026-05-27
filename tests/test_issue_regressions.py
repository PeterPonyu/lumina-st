from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_script(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_latent_flow_builds_config_from_allowed_cli_fields(tmp_path: Path) -> None:
    train_latent_flow = _load_script("scripts/e2e/train_latent_flow.py")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"latent_dim": 12, "cancer_types": ["coad"]}))

    cfg = train_latent_flow._build_config(
        argparse.Namespace(
            data="/not/part/of/config.h5ad",
            config=str(config_path),
            seed=123,
            max_epochs=2,
        )
    )

    assert cfg.latent_dim == 12
    assert cfg.cancer_types == ["COAD"]
    assert cfg.seed == 123
    assert cfg.max_epochs == 2


def test_baseline_imputation_missing_inputs_exit_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    baseline = (
        _load_script(str(Path.cwd().parent / "test_baseline_imputation.py"))
        if False
        else _load_script(
            str(Path(__file__).resolve().parents[1] / "scripts/e2e/test_baseline_imputation.py")
        )
    )

    assert baseline.main() == 1


def test_compare_outputs_missing_external_baseline_is_clean_precondition() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/e2e/compare_outputs.py"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "baselines/stPainter-original" in result.stdout
    assert "Traceback" not in result.stderr


def test_download_file_failure_does_not_write_mock_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    download = _load_script("src/lumina_st/cli/download.py")

    def fail_urlopen(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(download.urllib.request, "urlopen", fail_urlopen)
    dest = tmp_path / "fake.ckpt"

    with pytest.raises(OSError, match="network unavailable"):
        download.download_file("https://example.invalid/fake.ckpt", str(dest), timeout=0.01)

    assert not dest.exists()
    assert not (tmp_path / "fake.ckpt.tmp").exists()


def test_checkpoint_loaders_request_weights_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lumina_st.core import lumina_imputer

    calls: list[dict[str, object]] = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        return {"state_dict": {}}

    monkeypatch.setattr(lumina_imputer.torch, "load", fake_load)

    assert lumina_imputer._safe_torch_load(tmp_path / "model.ckpt") == {"state_dict": {}}
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_fit_seeds_lightning_and_dataloader(monkeypatch: pytest.MonkeyPatch) -> None:
    from lumina_st.config import LuminaSTConfig
    from lumina_st.core import lumina_imputer
    from lumina_st.core.lumina_imputer import LuminaImputer

    seen: dict[str, object] = {}

    monkeypatch.setattr(
        lumina_imputer.pl,
        "seed_everything",
        lambda seed, workers: seen.update(seed=seed, workers=workers),
    )

    class DummyDataset:
        def __init__(self, *args, **kwargs):
            pass

    class DummyLoader:
        def __init__(self, dataset, **kwargs):
            seen["loader_kwargs"] = kwargs

    class DummyTrainer:
        def fit(self, module, train_dataloaders):
            seen["fit_called"] = True

    monkeypatch.setattr(lumina_imputer, "ReferenceAtlasDataset", DummyDataset)
    monkeypatch.setattr(lumina_imputer, "DataLoader", DummyLoader)

    imputer = LuminaImputer(LuminaSTConfig(seed=77, num_workers=0), module=SimpleNamespace())
    returned = imputer.fit(trainer=DummyTrainer(), reference_adata=object())

    assert isinstance(returned, DummyTrainer)
    assert seen["seed"] == 77
    assert seen["workers"] is True
    loader_kwargs = seen["loader_kwargs"]
    assert loader_kwargs["worker_init_fn"] is lumina_imputer._seed_worker
    assert isinstance(loader_kwargs["generator"], torch.Generator)
    assert seen["fit_called"] is True


def test_repo_root_defaults_stay_inside_lumina_checkout() -> None:
    repo = Path(__file__).resolve().parents[1]
    eval_root = (repo / "scripts/e2e/eval_latent_quality.py").resolve().parents[2]
    enhance_root = (repo / "scripts/e2e/enhance_real_st.py").resolve().parents[2]
    fig_root = (repo / "scripts/visualize/fig_composed_main_claim.py").resolve().parents[2]

    assert eval_root == repo
    assert enhance_root == repo
    assert fig_root == repo


def test_readme_install_and_quickstart_match_current_api() -> None:
    readme = Path("README.md").read_text()
    install_section = readme.split("## Quick Start", 1)[0]

    assert "not published on PyPI yet" in install_section
    assert 'pip install -e ".[dev,scvi]"' in install_section
    assert "pip install lumina-st" not in install_section
    assert 'cancer_types=["COAD"]' in readme
    assert "LuminaImputer.from_config(cfg)" in readme
    assert 'cancer_type="COAD"' not in readme
    assert "checkpoint=" not in readme


def test_contract_pinned_benchmark_result_is_versionable() -> None:
    required = Path("results/benchmark/lumina_sweep_latest.json")
    assert required.exists()

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(required)],
        check=False,
    )
    assert ignored.returncode == 1


def test_all_torch_load_calls_use_weights_only() -> None:
    import ast

    paths = [
        Path("src/lumina_st/core/lumina_imputer.py"),
        Path("src/lumina_st/latents/scvi_vae.py"),
        Path("scripts/e2e/run_enhancement.py"),
    ]
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_torch_load = (
                isinstance(func, ast.Attribute)
                and func.attr == "load"
                and isinstance(func.value, ast.Name)
                and func.value.id == "torch"
            )
            if is_torch_load:
                assert any(
                    kw.arg == "weights_only"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                ), f"{path} has torch.load without weights_only=True"
