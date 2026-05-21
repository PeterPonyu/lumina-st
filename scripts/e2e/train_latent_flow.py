#!/usr/bin/env python3
"""
Training script for the LuminaST latent flow model (Phase 2).

Usage example:
    python scripts/train_latent_flow.py \
        --config configs/lumina_coad.yaml \
        --data data/processed/sc_train.h5ad
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import scanpy as sc
import torch
from torch.utils.data import DataLoader, random_split

from lumina_st.config import LuminaSTConfig
from lumina_st.data.datasets import ReferenceAtlasDataset
from lumina_st.data.cancer_registry import CancerRegistry
from lumina_st.models.lumina_transformer import LuminaTransformer
from lumina_st.modules.lumina_flow_module import LuminaFlowModule


def main(args):
    pl.seed_everything(args.seed, workers=True)

    cfg = LuminaSTConfig(**vars(args)) if args.config else LuminaSTConfig()
    print("Loaded config:", cfg.model_dump_for_checkpoint())

    # Load reference atlas
    adata = sc.read_h5ad(args.data)
    registry = CancerRegistry.default_pan_cancer()

    dataset = ReferenceAtlasDataset(adata, cfg, registry)
    train_len = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

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

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=cfg.gradient_clip_val,
        logger=True,
    )

    trainer.fit(module, train_loader, val_loader)

    # Save
    out_dir = Path(cfg.output_dir) / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": module.transformer.state_dict(),
        "config": cfg.model_dump_for_checkpoint(),
    }, out_dir / "lumina_flow.ckpt")
    print(f"Model saved to {out_dir / 'lumina_flow.ckpt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=50)
    args = parser.parse_args()
    main(args)
