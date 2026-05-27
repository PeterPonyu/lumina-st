#!/usr/bin/env python3
"""Smoke-train LuminaST on the synthetic AnnData produced by
`scripts/data_flow/generate_synthetic_st.py`.

This is the minimum-viable proof that the training pipeline works end-to-end
on this machine. Sidesteps the `--config` Pydantic-extra bug in
`train_latent_flow.py` by calling `imputer.fit()` directly with a tiny
config.

Usage:
    python scripts/e2e/train_smoke.py            # 2 epochs, CPU or GPU auto
    python scripts/e2e/train_smoke.py --epochs 5 # longer

The script prints runtime, final loss, device used, and the checkpoint path.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import anndata as ad
import pytorch_lightning as pl
import torch

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "synthetic_st_target.h5ad",
        help="Path to synthetic AnnData (defaults to data/processed/synthetic_st_target.h5ad inside the lumina-st repo).",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Force CPU or GPU; auto picks GPU when available.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/training/smoke"),
        help="Where to save the checkpoint (relative to lumina-st/).",
    )
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    if not args.data.exists():
        print(f"FAIL: synthetic AnnData not found at {args.data}")
        print(
            "Run `python scripts/data_flow/generate_synthetic_st.py` from the repo root"
            " to generate the fixture, then re-run this script."
        )
        return 1

    adata = ad.read_h5ad(args.data)
    print(f"[train-smoke] loaded {args.data.name}: n_cells={adata.n_obs}, n_genes={adata.n_vars}")
    print(f"[train-smoke] cancer types: {sorted(adata.obs['cancer_type'].unique().tolist())}")

    # Tiny config for fast smoke. The no-VAE training path treats X as
    # already-latent (see LuminaFlowModule.training_step), so latent_dim
    # must equal n_genes for the positional embedding to match the input.
    # When a real VAE is wired in, latent_dim becomes independent of n_genes.
    cfg = LuminaSTConfig(
        latent_dim=adata.n_vars,
        hidden_size=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        patch_size=1,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        num_workers=0,
        cancer_types=sorted(adata.obs["cancer_type"].unique().tolist()),
        vae_batch_key="cancer_type",
    )
    print(
        f"[train-smoke] tiny config: latent_dim={cfg.latent_dim}, "
        f"hidden={cfg.hidden_size}, depth={cfg.depth}, batch={cfg.batch_size}, epochs={cfg.max_epochs}"
    )

    imputer = LuminaImputer.from_config(cfg)

    # Manual trainer so we get explicit accelerator + per-epoch loss visibility
    if args.accelerator == "auto":
        accel = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        accel = args.accelerator
    if accel == "gpu":
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            pass
    print(f"[train-smoke] accelerator: {accel}")
    if accel == "gpu":
        print(f"[train-smoke] CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"[train-smoke] torch: {torch.__version__}, CUDA: {torch.version.cuda}")
        print(f"[train-smoke] arch_list: {torch.cuda.get_arch_list()}")

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=accel,
        devices=1,
        gradient_clip_val=cfg.gradient_clip_val,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
    )

    t0 = time.perf_counter()
    imputer.fit(trainer=trainer, reference_adata=adata)
    runtime = time.perf_counter() - t0

    out_dir = Path(__file__).resolve().parents[2] / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"smoke_{accel}.ckpt"
    torch.save(
        {
            "state_dict": imputer.module.transformer.state_dict(),
            "config": cfg.model_dump(),
            "epochs_trained": cfg.max_epochs,
            "global_step": trainer.global_step,
        },
        ckpt_path,
    )

    print()
    print("=" * 60)
    print("[train-smoke] DONE")
    print(f"  device:    {accel}")
    print(f"  epochs:    {cfg.max_epochs}")
    print(f"  steps:     {trainer.global_step}")
    print(f"  runtime:   {runtime:.2f} s")
    if accel == "gpu":
        try:
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"  peak GPU:  {peak_mb:.1f} MB")
        except RuntimeError:
            pass
    print(f"  ckpt:      {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
