"""
High-level user-facing API for LuminaST.

LuminaImputer is the main class researchers will import and use for both
training and inference. It hides all the Lightning / flow / transformer details.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np

import anndata as ad
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ..config.lumina_config import LuminaSTConfig
from ..data.cancer_registry import CancerRegistry
from ..data.datasets import ReferenceAtlasDataset
from ..models.lumina_transformer import LuminaTransformer
from ..modules.lumina_flow_module import LuminaFlowModule


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python RNGs for deterministic DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _safe_torch_load(path: str | Path, *, map_location: str = "cpu") -> Any:
    """Load local checkpoint tensors without enabling arbitrary pickle execution."""
    return torch.load(path, map_location=map_location, weights_only=True)


class LuminaImputer:
    """
    Main entry point for LuminaST.

    Example:
        imputer = LuminaImputer.from_checkpoint("checkpoints/lumina_50.ckpt", "checkpoints/vae_50.ckpt")
        enhanced = imputer.enhance(st_adata)
    """

    def __init__(self, config: LuminaSTConfig, module: LuminaFlowModule):
        self.config = config
        self.module = module
        # Preload Sparsity Data (matching baseline)
        self.sparsity_ratio = None
        if config.gene_sparsity_ratio_file and Path(config.gene_sparsity_ratio_file).exists():
            import pandas as pd

            try:
                self.sparsity_ratio = pd.read_csv(
                    config.gene_sparsity_ratio_file, index_col=0
                ).squeeze()
            except Exception as e:
                print(f"Warning: Failed to load sparsity file: {e}")

    @classmethod
    def from_config(cls, config: LuminaSTConfig) -> "LuminaImputer":
        registry = None
        if config.cancer_registry_file and Path(config.cancer_registry_file).exists():
            registry = CancerRegistry.from_file(config.cancer_registry_file)
        elif config.cancer_types:
            registry = CancerRegistry({c: i for i, c in enumerate(config.cancer_types)})
        else:
            registry = CancerRegistry.default_pan_cancer()

        transformer = LuminaTransformer(
            latent_dim=config.latent_dim,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_classes=len(registry),
            class_dropout_prob=config.class_dropout_prob,
        )
        module = LuminaFlowModule(config, transformer)
        return cls(config, module)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, vae_checkpoint_path: str, **kwargs
    ) -> "LuminaImputer":
        from ..latents.scvi_vae import SCVILatentEncoder

        # 1. Load the diffusion/transformer checkpoint
        ckpt = _safe_torch_load(checkpoint_path, map_location="cpu")
        hparams = ckpt.get("hyper_parameters", {})

        # Map baseline stPainter hyperparams to LuminaSTConfig fields
        latent_dim = hparams.get("latent_size", 50)
        hidden_size = hparams.get("hidden_size_sit", 256)
        depth = hparams.get("depth", 8)
        num_heads = hparams.get("num_heads", 8)
        mlp_ratio = hparams.get("mlp_ratio", 4.0)
        class_dropout_prob = hparams.get("class_dropout_prob", 0.1)
        patch_size = hparams.get("patch_size", 1)

        # Merge with explicitly passed configuration args
        config_args = {
            "latent_dim": latent_dim,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "class_dropout_prob": class_dropout_prob,
            "patch_size": patch_size,
        }
        for k, v in kwargs.items():
            if k in LuminaSTConfig.model_fields:
                config_args[k] = v

        config = LuminaSTConfig(**config_args)

        # 2. Reconstruct Transformer Model
        from ..data.cancer_registry import CancerRegistry

        registry = None
        if config.cancer_registry_file and Path(config.cancer_registry_file).exists():
            registry = CancerRegistry.from_file(config.cancer_registry_file)
        elif config.cancer_types:
            registry = CancerRegistry({c: i for i, c in enumerate(config.cancer_types)})
        else:
            registry = CancerRegistry.default_pan_cancer()

        num_classes = hparams.get("num_classes", len(registry))

        transformer = LuminaTransformer(
            latent_dim=config.latent_dim,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_classes=num_classes,
            class_dropout_prob=config.class_dropout_prob,
        )

        # 3. Load and reconstruct SCVILatentEncoder VAE
        n_input = hparams.get("input_size", 10000)
        n_batch = hparams.get("num_classes", len(registry))
        n_hidden = hparams.get("hidden_size_vae", 256)
        n_latent = hparams.get("latent_size", 50)
        n_layers = hparams.get("num_layer", 4)

        vae_encoder = SCVILatentEncoder.from_checkpoint(
            checkpoint_path=vae_checkpoint_path,
            n_input=n_input,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
        )

        # 4. Instantiate LuminaFlowModule
        module = LuminaFlowModule(config, transformer, vae=vae_encoder)

        # 5. Load flow weights from checkpoint state_dict
        state_dict = ckpt.get("state_dict", ckpt)
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                clean_state_dict["transformer." + k[6:]] = v
            elif k.startswith("ema_model."):
                clean_state_dict[k] = v
            else:
                clean_state_dict[k] = v

        module.load_state_dict(clean_state_dict, strict=False)
        return cls(config, module)

    def enhance(
        self,
        st_adata: ad.AnnData,
        cancer_type: Optional[str] = None,
        layer: Optional[str] = None,
        ode_style: str = "correct",
        uncond_class: str = "correct",
        sparsity_style: str = "gene",
        held_out_genes: Optional[list] = None,
    ) -> ad.AnnData:
        """
        Full guided enhancement / imputation pipeline.

        Returns a copy of the input AnnData with:
            - .layers['imputed']          : enhanced gene expression
            - .obsm['latent_enhanced']    : improved latent embedding
            - .obsm['latent_observed']    : original encoded latent (for comparison)

        If ``held_out_genes`` is provided, those gene columns are zeroed in the
        input expression matrix **before** the encoder sees them. This supports
        held-out gene recovery benchmarks: pass a list of gene names, then
        compare the imputed values at those columns against the original
        observations to score how well the model recovers masked signal.
        """
        import numpy as np

        cfg = self.config
        adata = st_adata.copy()

        # 1. Determine cancer label
        if cancer_type is None:
            # Try to infer from .obs
            if "cancer_type" in adata.obs:
                cancer_type = str(adata.obs["cancer_type"].iloc[0])
            else:
                cancer_type = cfg.cancer_types[0] if cfg.cancer_types else "UNKNOWN"

        # Resolve cancer index using the registry
        registry = None
        if cfg.cancer_registry_file and Path(cfg.cancer_registry_file).exists():
            registry = CancerRegistry.from_file(cfg.cancer_registry_file)
        elif cfg.cancer_types:
            registry = CancerRegistry({c: i for i, c in enumerate(cfg.cancer_types)})
        else:
            registry = CancerRegistry.default_pan_cancer()
        cancer_idx = registry[cancer_type]

        # 2. Get expression matrix
        if layer is not None and layer in adata.layers:
            expr = adata.layers[layer]
        else:
            expr = adata.X

        if hasattr(expr, "toarray"):
            expr = expr.toarray()

        # Zero out held-out gene columns at the raw-input layer (before
        # normalization or encoding) so the encoder sees them as absent.
        if held_out_genes:
            expr = np.asarray(expr, dtype=np.float32).copy()
            var_names = list(adata.var_names)
            mask_idx = [var_names.index(g) for g in held_out_genes if g in var_names]
            missing = [g for g in held_out_genes if g not in var_names]
            if missing:
                import warnings

                warnings.warn(
                    f"[LuminaST] {len(missing)}/{len(held_out_genes)} held-out genes "
                    f"not found in adata.var_names and will be skipped. "
                    f"First 5 missing: {missing[:5]}"
                )
            if mask_idx:
                expr[:, mask_idx] = 0.0
            print(
                f"[LuminaST] Holding out {len(mask_idx)}/{len(held_out_genes)} "
                f"genes for recovery benchmark"
            )

        x = torch.from_numpy(expr).float()

        # Move inputs to correct device if model is on a GPU
        device = (
            next(self.module.parameters()).device
            if list(self.module.parameters())
            else torch.device("cpu")
        )
        x = x.to(device)

        # 3. Encode to latent space
        y = torch.full((x.shape[0],), cancer_idx, dtype=torch.long, device=device)
        if self.module.vae is not None:
            z_obs, info = self.module.vae.encode_to_latent(x, y)
            library = info.get("library", None)
        else:
            if adata.n_vars != cfg.latent_dim:
                raise ValueError(
                    f"enhance() requires adata.n_vars (={adata.n_vars}) == cfg.latent_dim "
                    f"(={cfg.latent_dim}) when no VAE is configured. "
                    f"Either attach a VAE via vae_checkpoint, or supply latent-space input."
                )
            # Assume data is already in latent space or use identity
            z_obs = x
            library = None

        # 4. Run guided enhancement in latent space
        z_enhanced = self.module.enhance_latent(
            z_obs,
            y,
            ode_style=ode_style,
            uncond_class=uncond_class,
        )

        # 5. Decode back (if VAE present)
        if self.module.vae is not None:
            x_imputed = self.module.vae.decode_from_latent(z_enhanced, y, library=library)
        else:
            x_imputed = z_enhanced

        # Optional: basic post-processing (sparsity)
        if cfg.apply_sparsity and x_imputed.shape[1] == adata.n_vars:
            if sparsity_style == "gene" and self.sparsity_ratio is not None:
                # Apply baseline-matching per-gene sparsity constraint
                import pandas as pd

                vals = None
                if (
                    isinstance(self.sparsity_ratio, pd.DataFrame)
                    and cancer_type in self.sparsity_ratio
                ):
                    vals = self.sparsity_ratio[cancer_type].values
                elif isinstance(self.sparsity_ratio, pd.Series):
                    vals = self.sparsity_ratio.values

                if vals is not None and len(vals) == x_imputed.shape[1]:
                    vals_tensor = torch.tensor(vals, dtype=x_imputed.dtype, device=x_imputed.device)
                    N, G = x_imputed.shape

                    # Sort each column independently
                    sorted_data, _ = torch.sort(x_imputed, dim=0)

                    # Calculate index interpolation bounds
                    indices = vals_tensor * (N - 1)
                    indices_low = torch.floor(indices).long().clamp(0, N - 1)
                    indices_high = torch.ceil(indices).long().clamp(0, N - 1)
                    weight = indices - indices_low.float()

                    # Gather values from sorted_data
                    val_low = torch.gather(sorted_data, 0, indices_low.unsqueeze(0))
                    val_high = torch.gather(sorted_data, 0, indices_high.unsqueeze(0))

                    # Interpolate thresholds
                    thresh = val_low + weight.unsqueeze(0) * (val_high - val_low)

                    # Apply threshold mask
                    mask = x_imputed < thresh
                    active_cols = (vals_tensor > 0.0).unsqueeze(0)
                    mask = mask & active_cols
                    x_imputed[mask] = 0.0

                    # Boundary conditions: ratio >= 1.0 => column must be all zeros
                    zero_cols = (vals_tensor >= 1.0).unsqueeze(0)
                    x_imputed = torch.where(zero_cols, torch.zeros_like(x_imputed), x_imputed)
            else:
                # Simple top-percentile thresholding per cell
                sorted_data, _ = torch.sort(x_imputed, dim=1)
                G = x_imputed.shape[1]
                idx_float = cfg.sparsity_percentile * (G - 1)
                idx_low = int(np.floor(idx_float))
                idx_high = int(np.ceil(idx_float))
                weight = idx_float - idx_low

                val_low = sorted_data[:, idx_low]
                val_high = sorted_data[:, idx_high]
                thresh = val_low + weight * (val_high - val_low)

                mask = x_imputed < thresh.unsqueeze(1)
                x_imputed[mask] = 0.0

        # 6. Store results
        adata.obsm["latent_observed"] = z_obs.detach().cpu().numpy()
        adata.obsm["latent_enhanced"] = z_enhanced.detach().cpu().numpy()

        if x_imputed.shape[1] == adata.n_vars:
            adata.layers["imputed"] = x_imputed.detach().cpu().numpy()
        else:
            # Latent space only
            adata.layers["imputed_latent"] = x_imputed.detach().cpu().numpy()

        return adata

    def fit(
        self,
        trainer: Optional[pl.Trainer] = None,
        *,
        reference_adata: Optional[ad.AnnData] = None,
        train_loader: Optional[DataLoader] = None,
        **trainer_kwargs,
    ) -> pl.Trainer:
        """Train the flow model on a reference atlas.

        The package-level seed is applied before DataLoader/Trainer construction
        so repeated smoke runs with the same ``LuminaSTConfig.seed`` are
        reproducible.
        """
        pl.seed_everything(self.config.seed, workers=True)
        if train_loader is None:
            if reference_adata is None:
                raise ValueError("Provide reference_adata or train_loader to train LuminaST")

            if self.config.cancer_registry_file and Path(self.config.cancer_registry_file).exists():
                registry = CancerRegistry.from_file(self.config.cancer_registry_file)
            elif self.config.cancer_types:
                registry = CancerRegistry({c: i for i, c in enumerate(self.config.cancer_types)})
            else:
                registry = CancerRegistry.default_pan_cancer()

            dataset = ReferenceAtlasDataset(reference_adata, self.config, registry)
            generator = torch.Generator()
            generator.manual_seed(self.config.seed)
            train_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                generator=generator,
                worker_init_fn=_seed_worker,
            )

        if trainer is None:
            trainer = pl.Trainer(
                max_epochs=self.config.max_epochs,
                gradient_clip_val=self.config.gradient_clip_val,
                default_root_dir=str(self.config.output_dir),
                **trainer_kwargs,
            )

        trainer.fit(self.module, train_dataloaders=train_loader)
        return trainer
