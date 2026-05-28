#!/usr/bin/env python3
"""
Practical End-to-End script for LuminaST enhancement.

If you do **not** have suitable real local ST data yet (common situation), the script
automatically falls back to high-quality synthetic data that mimics
a pan-cancer spatial-transcriptomics reference baseline.

Usage with real data (recommended when available):
    conda run -n dl python scripts/e2e/enhance_real_st.py \
        --reference /path/to/reference_atlas.h5ad \
        --target    /path/to/real_st_slice.h5ad \
        --cancer    COAD \
        --output    ./enhanced.h5ad

Usage with synthetic data (works today, produces meaningful metrics):
    conda run -n dl python scripts/e2e/enhance_real_st.py --output ./synthetic_enhanced.h5ad

The script always prints training curves + final enhancement metrics so you can
judge whether the model is learning something useful.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from lumina_st.config.lumina_config import LuminaSTConfig  # noqa: E402
from lumina_st.latents.tiny_vae import TinyVAE  # noqa: E402
from lumina_st.latents.scvi_vae import SCVILatentEncoder  # noqa: E402
from lumina_st.models.lumina_transformer import LuminaTransformer  # noqa: E402
from lumina_st.modules.lumina_flow_module import LuminaFlowModule  # noqa: E402
from lumina_st.core.lumina_imputer import LuminaImputer  # noqa: E402
from lumina_st.data.datasets import ReferenceAtlasDataset  # noqa: E402
from lumina_st.data.cancer_registry import CancerRegistry  # noqa: E402
from lumina_st.metrics.enhancement_evaluator import EnhancementEvaluator  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_flow.generate_synthetic_st import generate_synthetic_reference_and_st  # noqa: E402

try:
    from scvi.model import SCVI
except ImportError:
    SCVI = None


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        probe = torch.zeros(1, device="cuda")
        _ = torch.relu(probe)
        return torch.device("cuda")
    except Exception as exc:
        print(f"[WARNING] CUDA is available but failed a kernel probe: {exc}")
        print("          Falling back to CPU for this run.")
        return torch.device("cpu")


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    # === Smart data source selection ===
    # Priority: 1. Explicit --reference / --target
    #           2. Real baseline data from labs/data/baselines/stpainter/ (if present)
    #           3. High-quality synthetic data (always works)

    DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "baselines" / "stpainter"

    if args.reference and args.target:
        ref = sc.read_h5ad(args.reference)
        target = sc.read_h5ad(args.target)
        print("Using user-provided real data files.")
    elif DATA_ROOT.exists() and any((DATA_ROOT / sub).exists() for sub in ("processed", "processed_data")):
        # Accept both folder names — the gdown download yields processed_data/,
        # while the upstream README documents processed/.
        processed = DATA_ROOT / "processed" if (DATA_ROOT / "processed").exists() else DATA_ROOT / "processed_data"
        print(f"\n[INFO] Found real baseline data at: {DATA_ROOT}")
        print(f"       Using {processed.name}/ for E2E run.\n")
        ref_path = processed / "sc_train.h5ad"
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference atlas {ref_path} missing — re-run `python -m lumina_st.cli.download`."
            )
        ref = sc.read_h5ad(ref_path)
        # Pick the first available cancer test file as target
        possible_targets = list(processed.glob("st_*_test.h5ad"))
        if not possible_targets:
            raise FileNotFoundError(f"No st_*_test.h5ad found in {processed}")
        target_path = possible_targets[0]
        target = sc.read_h5ad(target_path)
        if args.cancer is None:
            args.cancer = target_path.stem.replace("st_", "").replace("_test", "")
    else:
        print("\n[INFO] No real data provided and no baseline data found locally.")
        print("       Falling back to high-fidelity synthetic data (mimics the reference baseline usage).\n")

        ref, target, cancer_names = generate_synthetic_reference_and_st(
            n_ref_cells=2500, n_st_cells=500, n_genes=180, n_cancer_types=4, seed=42
        )
        if args.cancer is None:
            args.cancer = cancer_names[0]

    print(f"Reference: {ref.n_obs} cells, {ref.n_vars} genes")
    print(f"Target ST : {target.n_obs} cells, {target.n_vars} genes")

    # Create a minimal registry from the data
    if "cancer_type" in ref.obs:
        cancers = sorted(ref.obs["cancer_type"].astype(str).unique().tolist())
    elif "Tumor Type" in ref.obs:
        cancers = sorted(ref.obs["Tumor Type"].astype(str).unique().tolist())
    else:
        cancers = ["UNKNOWN"]
    registry = CancerRegistry({c: i for i, c in enumerate(cancers)})

    cfg = LuminaSTConfig(
        latent_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        guidance_scale=args.guidance_scale,
        cancer_types=[args.cancer] if args.cancer else cancers[:1],
        vae_batch_key="cancer_type",  # this script keys cancer labels here; #106
    )

    # 1. Train scVI VAE on reference (if not provided)
    if args.scvi_model and Path(args.scvi_model).exists():
        if SCVI is None:
            raise ImportError("scvi-tools is required to load an existing SCVI model")
        print("Loading existing SCVI model...")
        scvi_model = SCVI.load(args.scvi_model)
    else:
        if SCVI is not None:
            print("Training quick SCVI VAE on reference atlas...")
            SCVI.setup_anndata(ref, batch_key="cancer_type" if "cancer_type" in ref.obs else None)
            scvi_model = SCVI(ref, n_latent=cfg.latent_dim)
            scvi_model.train(max_epochs=min(30, args.max_epochs))
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            scvi_save_dir = Path(args.output).parent / "scvi_ref"
            # scvi save() refuses to overwrite by default; allow re-runs to succeed.
            scvi_model.save(str(scvi_save_dir), overwrite=True)
        else:
            print("[WARNING] scvi-tools not found in current env. Using TinyVAE for fast synthetic demo.")
            print("          For real results, run with: conda run -n dl python ...")
            scvi_model = None  # signal to use TinyVAE path

    # 2. Prepare datasets
    dataset = ReferenceAtlasDataset(ref, cfg, registry)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    # 3. Lumina model
    transformer = LuminaTransformer(
        latent_dim=cfg.latent_dim,
        patch_size=1,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=4.0,
        num_classes=len(registry),
        class_dropout_prob=0.1,
    )

    if scvi_model is None:
        vae_wrapper = TinyVAE(input_dim=ref.n_vars, latent_dim=cfg.latent_dim).to(device)
        vae_opt = torch.optim.AdamW(vae_wrapper.parameters(), lr=cfg.lr)
        x_ref = torch.as_tensor(np.asarray(ref.X), dtype=torch.float32, device=device)
        print("Training TinyVAE fallback encoder...")
        for epoch in range(min(5, cfg.max_epochs)):
            vae_loss = vae_wrapper(x_ref)["loss"]
            vae_opt.zero_grad()
            vae_loss.backward()
            vae_opt.step()
            print(f"  TinyVAE epoch {epoch+1} - loss {vae_loss.item():.4f}")
    else:
        vae_wrapper = SCVILatentEncoder(scvi_model)

    module = LuminaFlowModule(cfg, transformer, vae=vae_wrapper).to(device)

    # Quick training of the flow model
    opt = torch.optim.AdamW(module.parameters(), lr=cfg.lr)
    print("Training Lumina flow model on latent space...")
    for epoch in range(cfg.max_epochs):
        last_loss = None
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            z, _ = vae_wrapper.encode_to_latent(x, y)   # may need adaptation for real SCVI
            # For real SCVI we should use the model's get_latent_representation on AnnData batches
            # This is a simplified path — for production use the high-level API on AnnData
            loss_dict = module.transport.training_losses(transformer, z.detach(), {"y": y})
            loss = loss_dict["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss
        if epoch % 5 == 0 and last_loss is not None:
            print(f"  Epoch {epoch+1}/{cfg.max_epochs} - loss {last_loss.item():.4f}")

    # 4. Enhance target
    print("Enhancing target ST slice...")
    imputer = LuminaImputer(cfg, module)
    enhanced = imputer.enhance(target, cancer_type=args.cancer)

    # 5. Metrics
    evaluator = EnhancementEvaluator(enhanced)
    metrics = evaluator.summary()
    print("\n=== Enhancement Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 6. Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced.write(out_path)
    print(f"\nEnhanced AnnData saved to {out_path}")
    print("Key outputs:")
    print("  .layers['imputed'] or .layers['imputed_latent']")
    print("  .obsm['latent_enhanced']")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=None, help="Path to reference scRNA atlas h5ad (optional - will use synthetic)")
    parser.add_argument("--target", default=None, help="Path to target ST h5ad to enhance (optional - will use synthetic)")
    parser.add_argument("--cancer", default=None)
    parser.add_argument("--output", default="./lumina_enhanced.h5ad")
    parser.add_argument("--scvi_model", default=None)
    parser.add_argument("--latent_dim", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    args = parser.parse_args()
    main(args)
