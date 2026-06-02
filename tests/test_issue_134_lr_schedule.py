"""Unit tests for the LR warmup + scheduler (issue #134)."""

from __future__ import annotations

import math

import torch

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.models.lumina_transformer import LuminaTransformer
from lumina_st.modules.lumina_flow_module import LuminaFlowModule, build_lr_lambda


def _module(**cfg_kwargs) -> LuminaFlowModule:
    cfg = LuminaSTConfig(
        latent_dim=8, hidden_size=16, depth=1, num_heads=2, **cfg_kwargs
    )
    transformer = LuminaTransformer(
        latent_dim=8, patch_size=1, hidden_size=16, depth=1,
        num_heads=2, mlp_ratio=4.0, num_classes=2, class_dropout_prob=0.1,
    )
    return LuminaFlowModule(cfg, transformer)


def test_linear_warmup_ramp():
    lam = build_lr_lambda(warmup_steps=10, total_steps=100, schedule="cosine",
                          base_lr=1e-3, min_lr=0.0)
    assert lam(0) == 0.1
    assert math.isclose(lam(4), 0.5)
    assert math.isclose(lam(9), 1.0)


def test_cosine_decays_to_min_after_warmup():
    lam = build_lr_lambda(warmup_steps=10, total_steps=110, schedule="cosine",
                          base_lr=1.0, min_lr=0.0)
    # First post-warmup step ~ full LR, end of horizon ~ min.
    assert math.isclose(lam(10), 1.0, abs_tol=1e-6)
    assert math.isclose(lam(110), 0.0, abs_tol=1e-6)
    # Monotone non-increasing through the decay window.
    vals = [lam(s) for s in range(10, 111, 10)]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))


def test_min_lr_floor_respected():
    lam = build_lr_lambda(warmup_steps=0, total_steps=100, schedule="linear",
                          base_lr=1.0, min_lr=0.25)
    assert math.isclose(lam(100), 0.25, abs_tol=1e-6)


def test_constant_schedule_is_flat():
    lam = build_lr_lambda(warmup_steps=0, total_steps=100, schedule="constant",
                          base_lr=1.0, min_lr=0.0)
    assert lam(0) == 1.0 and lam(50) == 1.0 and lam(100) == 1.0


def test_default_config_returns_bare_optimizer():
    """Defaults (constant, no warmup) must reproduce the prior bare-AdamW path."""
    module = _module()
    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)


def test_scheduled_config_returns_step_interval_scheduler():
    module = _module(lr_schedule="cosine", warmup_steps=5, min_lr=1e-6, max_epochs=10)
    out = module.configure_optimizers()
    assert isinstance(out, dict)
    assert isinstance(out["optimizer"], torch.optim.AdamW)
    assert out["lr_scheduler"]["interval"] == "step"
    assert isinstance(
        out["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.LambdaLR
    )


def test_schedule_choice_is_checkpointed():
    cfg = LuminaSTConfig(lr_schedule="cosine", warmup_steps=100, min_lr=1e-6)
    dumped = cfg.model_dump_for_checkpoint()
    assert dumped["lr_schedule"] == "cosine"
    assert dumped["warmup_steps"] == 100
    assert dumped["min_lr"] == 1e-6
