#!/usr/bin/env python3
"""
Training script for the LuminaST latent flow model (Phase 2).

Usage example:
    python scripts/train_latent_flow.py \
        --config configs/lumina_coad.yaml \
        --data data/processed/sc_train.h5ad
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import scanpy as sc
import torch
from torch.utils.data import DataLoader, random_split

from lumina_st.config import LuminaSTConfig
from lumina_st.core.lumina_imputer import save_flow_checkpoint
from lumina_st.data.datasets import ReferenceAtlasDataset
from lumina_st.data.cancer_registry import CancerRegistry
from lumina_st.models.lumina_transformer import LuminaTransformer
from lumina_st.modules.lumina_flow_module import LuminaFlowModule


def _seed_worker(worker_id: int) -> None:
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _load_config_overrides(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "YAML config files require PyYAML; use JSON or install PyYAML."
            ) from exc
        raw = yaml.safe_load(path.read_text())

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return raw


def _build_config(args: argparse.Namespace) -> LuminaSTConfig:
    config_kwargs = _load_config_overrides(args.config)

    # LuminaSTConfig uses extra="forbid" (so a misspelled key is caught rather
    # than silently ignored), but splatting a raw YAML dict straight in surfaces
    # that as an opaque Pydantic ValidationError. Check up front and raise an
    # actionable message naming the unknown key(s) and the valid field set so a
    # config typo is obvious instead of cryptic (#280).
    known = set(LuminaSTConfig.model_fields)
    unknown = sorted(set(config_kwargs) - known)
    if unknown:
        raise ValueError(
            f"Unknown config field(s) {unknown} in {args.config!r}. "
            f"Valid LuminaSTConfig fields are: {sorted(known)}"
        )

    # CLI flags override file values for the two knobs the CLI exposes.
    cli_kwargs = {"seed": args.seed, "max_epochs": args.max_epochs}
    return LuminaSTConfig(**{**config_kwargs, **cli_kwargs})


def main(args):
    pl.seed_everything(args.seed, workers=True)

    cfg = _build_config(args)
    print("Loaded config:", cfg.model_dump_for_checkpoint())

    # Load reference atlas
    adata = sc.read_h5ad(args.data)
    registry = CancerRegistry.default_pan_cancer()

    dataset = ReferenceAtlasDataset(adata, cfg, registry)
    train_len = int(0.9 * len(dataset))
    split_generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(
        dataset, [train_len, len(dataset) - train_len], generator=split_generator
    )

    loader_generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=loader_generator,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        worker_init_fn=_seed_worker,
    )

    # Model
    transformer = LuminaTransformer(
        latent_dim=cfg.latent_dim,
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        num_classes=len(registry),
        class_dropout_prob=cfg.class_dropout_prob,
    )

    module = LuminaFlowModule(cfg, transformer)

    # Validation-driven model selection (issue #146): now that the module logs
    # `val_loss` in validation_step, monitor it to save the best epoch (not just
    # the last) and optionally stop early when it plateaus.
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    ckpt_dir = Path(cfg.output_dir) / cfg.experiment_name / "checkpoints"
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}-{val_loss:.4f}",
        )
    ]
    if cfg.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", mode="min", patience=cfg.early_stopping_patience
            )
        )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=cfg.gradient_clip_val,
        logger=True,
        callbacks=callbacks,
    )

    trainer.fit(module, train_loader, val_loader)

    # Save
    out_dir = Path(cfg.output_dir) / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "lumina_flow.ckpt"
    # Persist the full LightningModule state — including the EMA branch
    # that ``enhance()`` actually samples from. See ``save_flow_checkpoint``
    # docstring (lumina-st #147).
    save_flow_checkpoint(module, cfg, ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=50)
    args = parser.parse_args()
    main(args)
