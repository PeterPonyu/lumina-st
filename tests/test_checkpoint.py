"""Regression tests for lumina-st checkpoint persistence.

#118 — ``LuminaImputer.from_checkpoint`` previously called
``module.load_state_dict(..., strict=False)``, silently leaving missing
parameters at random init. Now routed through ``_strict_load_state_dict``
which raises ``RuntimeError`` with the missing/unexpected key list.

#147 — ``scripts/e2e/train_latent_flow.py`` previously persisted only
``module.transformer.state_dict()``. Inference samples exclusively from
``self.ema_model`` (``lumina_flow_module.py:134/148/185``), so the EMA
weights used at enhancement time were not in the checkpoint at all —
reloads silently regenerated EMA from random init. The fix saves the
full ``module.state_dict()``, which contains both ``transformer.*`` and
``ema_model.*`` prefixes that ``from_checkpoint`` already understands.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import _strict_load_state_dict, save_flow_checkpoint
from lumina_st.models.lumina_transformer import LuminaTransformer
from lumina_st.modules.lumina_flow_module import LuminaFlowModule


def test_renamed_key_raises_with_key_list() -> None:
    """A renamed key must trigger ``RuntimeError`` and the message must list
    both the missing model-side key and the unexpected checkpoint-side key.
    """

    module = nn.Linear(3, 4)
    real_state = module.state_dict()

    # Simulate a checkpoint where ``weight`` was renamed to ``weight_renamed``.
    bad_state = {
        "weight_renamed": real_state["weight"].clone(),
        "bias": real_state["bias"].clone(),
    }

    with pytest.raises(RuntimeError) as excinfo:
        _strict_load_state_dict(module, bad_state)

    msg = str(excinfo.value)
    # The unexpected (checkpoint-only) key must appear in the error.
    assert "weight_renamed" in msg, msg
    # The missing (model-only) key must appear in the error.
    assert "weight" in msg, msg
    # The message must make it clear loading was refused.
    assert "strict=False" in msg or "random init" in msg or "Refusing" in msg, msg


def test_matching_keys_load_silently() -> None:
    """A state_dict whose keys exactly match must load without raising."""

    module = nn.Linear(3, 4)
    state = module.state_dict()

    # Mutate the values so we can verify the load actually happened.
    new_state = {k: torch.zeros_like(v) for k, v in state.items()}

    _strict_load_state_dict(module, new_state)

    for name, param in module.state_dict().items():
        assert torch.equal(param, torch.zeros_like(param)), name


# ---------------------------------------------------------------------------
# #147 — EMA persistence
# ---------------------------------------------------------------------------


def _build_minimal_module() -> LuminaFlowModule:
    cfg = LuminaSTConfig(
        latent_dim=8,
        hidden_size=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        cancer_types=["A", "B"],
    )
    transformer = LuminaTransformer(
        latent_dim=cfg.latent_dim,
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        num_classes=len(cfg.cancer_types),
        class_dropout_prob=cfg.class_dropout_prob,
    )
    return LuminaFlowModule(cfg, transformer)


def test_ema_state_persisted_and_restored(tmp_path) -> None:
    """The checkpoint format used by ``train_latent_flow.py`` must round-trip
    the EMA branch param-by-param.

    Pre-fix: ``torch.save({"state_dict": module.transformer.state_dict()})``
    dropped every ``ema_model.*`` key, so reloading regenerated EMA from
    random init even though guided sampling reads exclusively from EMA.

    Post-fix: ``torch.save({"state_dict": module.state_dict()})`` carries
    both ``transformer.*`` and ``ema_model.*``, and ``_strict_load_state_dict``
    refuses any silent mismatch.
    """

    torch.manual_seed(0)
    original = _build_minimal_module()

    # Simulate the post-training EMA having drifted off the transformer weights
    # (the whole point of EMA — and the part the pre-fix save threw away).
    with torch.no_grad():
        for p in original.ema_model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    # Snapshot EMA params before save so we can compare bit-for-bit on reload.
    ema_before = {n: p.detach().clone() for n, p in original.ema_model.state_dict().items()}

    # Persist using the SAME schema train_latent_flow.py emits after the fix.
    ckpt_path = tmp_path / "lumina_flow.ckpt"
    torch.save({"state_dict": original.state_dict()}, ckpt_path)

    # Saved checkpoint must literally contain ema_model.* keys — otherwise the
    # fix regressed back to transformer-only persistence.
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    saved_state = saved["state_dict"]
    assert any(k.startswith("ema_model.") for k in saved_state), (
        "Checkpoint missing ema_model.* keys — EMA weights would not survive reload."
    )

    # Reload into a freshly-initialized module and verify EMA equality.
    torch.manual_seed(1)  # different seed: any leaked random init would diverge
    restored = _build_minimal_module()
    _strict_load_state_dict(restored, saved_state)

    for name, param in restored.ema_model.state_dict().items():
        ref = ema_before[name]
        assert torch.equal(param, ref), (
            f"EMA parameter '{name}' did not round-trip through save/reload."
        )


def test_legacy_transformer_only_save_fails_loudly() -> None:
    """The pre-fix save shape — ``module.transformer.state_dict()`` — must
    now be rejected by ``_strict_load_state_dict``.

    Before #118 + #147, loading this shape silently left both
    ``transformer.*`` and ``ema_model.*`` keys at random init. The combined
    fix turns that into a loud ``RuntimeError`` so the user notices the
    broken artifact instead of publishing metrics off random weights.
    """

    module = _build_minimal_module()
    bad_state = module.transformer.state_dict()  # missing 'transformer.' / 'ema_model.' prefixes

    with pytest.raises(RuntimeError):
        _strict_load_state_dict(module, bad_state)


# ---------------------------------------------------------------------------
# #130 — fit() checkpoint save / resume contract
# ---------------------------------------------------------------------------


def _tiny_fit_config(tmp_path, *, max_epochs: int = 1, checkpoint_dir=None) -> LuminaSTConfig:
    """A minimal CPU-friendly config whose latent_dim == reference n_vars so the
    VAE-free flow module trains directly on ``.X`` (training_step sets z = x)."""

    return LuminaSTConfig(
        latent_dim=8,
        hidden_size=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        cancer_types=["A", "B"],
        vae_batch_key="cancer_type",
        batch_size=8,
        num_workers=0,
        max_epochs=max_epochs,
        output_dir=tmp_path / "run",
        checkpoint_dir=checkpoint_dir,
    )


def _tiny_reference_adata(n_obs: int = 16, n_vars: int = 8):
    """Synthetic reference atlas: n_vars must equal config.latent_dim (no VAE)."""

    import numpy as np
    from anndata import AnnData

    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n_obs, n_vars)).astype("float32")
    obs = {"cancer_type": (["A", "B"] * (n_obs // 2 + 1))[:n_obs]}
    return AnnData(X=X, obs=obs)


def _cpu_trainer_kwargs() -> dict:
    return {"accelerator": "cpu", "devices": 1, "logger": False, "enable_progress_bar": False}


def _resolved_ckpt_dir(cfg: LuminaSTConfig):
    from pathlib import Path

    return (
        Path(cfg.checkpoint_dir)
        if cfg.checkpoint_dir is not None
        else Path(cfg.output_dir) / "checkpoints"
    )


def test_fit_train_only_writes_resumable_checkpoint(tmp_path) -> None:
    """A train-only run (no val_loader) must still write a resumable ``last.ckpt``
    in the resolved checkpoint dir (#130). Pre-fix, ModelCheckpoint was only
    registered when val_loader was given, so train-only runs saved nothing."""

    from lumina_st.core.lumina_imputer import LuminaImputer

    cfg = _tiny_fit_config(tmp_path, max_epochs=1)
    imputer = LuminaImputer.from_config(cfg)

    imputer.fit(
        reference_adata=_tiny_reference_adata(),
        run_dir=None,
        **_cpu_trainer_kwargs(),
    )

    ckpt_dir = _resolved_ckpt_dir(cfg)
    last = ckpt_dir / "last.ckpt"
    assert last.exists(), f"expected resumable checkpoint at {last}; dir contents: " + (
        str(list(ckpt_dir.iterdir())) if ckpt_dir.exists() else "<missing>"
    )


def test_fit_respects_explicit_checkpoint_dir(tmp_path) -> None:
    """``config.checkpoint_dir`` overrides the default output_dir/checkpoints (#130)."""

    from lumina_st.core.lumina_imputer import LuminaImputer

    explicit = tmp_path / "custom_ckpts"
    cfg = _tiny_fit_config(tmp_path, max_epochs=1, checkpoint_dir=explicit)
    imputer = LuminaImputer.from_config(cfg)

    imputer.fit(
        reference_adata=_tiny_reference_adata(),
        run_dir=None,
        **_cpu_trainer_kwargs(),
    )

    assert (explicit / "last.ckpt").exists()
    # The default location must NOT be used when an explicit dir is set.
    assert not (cfg.output_dir / "checkpoints" / "last.ckpt").exists()


def test_fit_resume_continues_training(tmp_path) -> None:
    """Resuming from a checkpoint must continue training rather than restart:
    ``trainer.global_step`` / ``current_epoch`` must advance beyond the resume
    point (#130). Pre-fix, resume_from was only recorded in the manifest and
    never passed to trainer.fit(ckpt_path=...), so training restarted at 0."""

    from lumina_st.core.lumina_imputer import LuminaImputer

    cfg = _tiny_fit_config(tmp_path, max_epochs=1)
    imputer = LuminaImputer.from_config(cfg)
    trainer1 = imputer.fit(
        reference_adata=_tiny_reference_adata(),
        run_dir=None,
        **_cpu_trainer_kwargs(),
    )
    step_after_first = trainer1.global_step
    epoch_after_first = trainer1.current_epoch
    assert step_after_first > 0

    ckpt = _resolved_ckpt_dir(cfg) / "last.ckpt"
    assert ckpt.exists()

    # Fresh imputer + larger budget, resume from the saved checkpoint.
    cfg2 = _tiny_fit_config(tmp_path, max_epochs=3)
    imputer2 = LuminaImputer.from_config(cfg2)
    trainer2 = imputer2.fit(
        reference_adata=_tiny_reference_adata(),
        run_dir=None,
        resume_from=ckpt,
        **_cpu_trainer_kwargs(),
    )

    # Resume restored epoch/global_step state, then trained further: the second
    # run must end strictly past where the first stopped (no restart from 0).
    assert trainer2.current_epoch > epoch_after_first, (
        f"resume restarted: current_epoch {trainer2.current_epoch} "
        f"did not advance past {epoch_after_first}"
    )
    assert trainer2.global_step > step_after_first, (
        f"resume restarted: global_step {trainer2.global_step} "
        f"did not advance past {step_after_first}"
    )


def test_fit_checkpoint_loads_for_inference(tmp_path) -> None:
    """A fit-produced checkpoint round-trips back into a flow module via the
    strict loader without manual key surgery (#130 inference path)."""

    from lumina_st.core.lumina_imputer import LuminaImputer

    cfg = _tiny_fit_config(tmp_path, max_epochs=1)
    imputer = LuminaImputer.from_config(cfg)
    imputer.fit(
        reference_adata=_tiny_reference_adata(),
        run_dir=None,
        **_cpu_trainer_kwargs(),
    )

    ckpt = _resolved_ckpt_dir(cfg) / "last.ckpt"
    assert ckpt.exists()

    # Lightning's last.ckpt nests weights under "state_dict" with the SAME
    # module-relative keys (transformer.* / ema_model.*) the strict loader
    # expects — no manual remapping required.
    saved = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = saved["state_dict"]
    assert any(k.startswith("ema_model.") for k in state)
    assert any(k.startswith("transformer.") for k in state)

    fresh = _build_minimal_module()
    _strict_load_state_dict(fresh, state)
