"""Validation step + model-selection callbacks (issue #146)."""

from __future__ import annotations

from pathlib import Path

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer
from lumina_st.models.lumina_transformer import LuminaTransformer
from lumina_st.modules.lumina_flow_module import LuminaFlowModule


def test_module_defines_validation_step():
    assert hasattr(LuminaFlowModule, "validation_step")


def _tiny_setup(tmp_path: Path):
    cfg = LuminaSTConfig(
        latent_dim=8, hidden_size=16, depth=1, num_heads=2,
        max_epochs=1, batch_size=4, output_dir=tmp_path,
    )
    transformer = LuminaTransformer(
        latent_dim=8, patch_size=1, hidden_size=16, depth=1,
        num_heads=2, mlp_ratio=4.0, num_classes=2, class_dropout_prob=0.1,
    )
    module = LuminaFlowModule(cfg, transformer)  # no VAE: z = x
    x = torch.randn(16, 8)
    y = torch.zeros(16, dtype=torch.long)
    train = DataLoader(TensorDataset(x, y), batch_size=4)
    val = DataLoader(TensorDataset(x[:8], y[:8]), batch_size=4)
    return cfg, module, train, val


def test_fit_logs_val_loss_and_writes_best_checkpoint(tmp_path):
    cfg, module, train, val = _tiny_setup(tmp_path)
    imputer = LuminaImputer(cfg, module)

    trainer = imputer.fit(
        train_loader=train,
        val_loader=val,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
    )

    # val_loss was logged during validation_step.
    assert "val_loss" in trainer.callback_metrics

    # A monitored best-checkpoint was written.
    ckpts = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
    assert ckpts, "ModelCheckpoint(monitor='val_loss') should be wired when a val_loader is passed"
    best = ckpts[0].best_model_path
    assert best and Path(best).exists()


def test_fit_without_val_loader_has_no_monitored_checkpoint(tmp_path):
    """Backward compatibility: no val loader -> no monitored callbacks added."""
    cfg, module, train, _ = _tiny_setup(tmp_path)
    imputer = LuminaImputer(cfg, module)

    trainer = imputer.fit(
        train_loader=train,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
    )
    monitored = [
        c for c in trainer.callbacks
        if isinstance(c, ModelCheckpoint) and c.monitor == "val_loss"
    ]
    assert not monitored
