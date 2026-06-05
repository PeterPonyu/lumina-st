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


def save_flow_checkpoint(
    module: "LuminaFlowModule",
    config: LuminaSTConfig,
    path: str | Path,
) -> None:
    """Persist a ``LuminaFlowModule`` checkpoint that ``from_checkpoint`` can
    fully restore — including the EMA branch.

    Inference samples exclusively from ``module.ema_model``
    (``lumina_flow_module.py:134/148/185``), so saving only the transformer's
    ``state_dict`` would discard the weights that ``enhance()`` actually uses,
    making reported metrics unreproducible from the saved artifact
    (lumina-st #147). ``module.state_dict()`` naturally carries both the
    ``transformer.*`` and ``ema_model.*`` key prefixes that ``from_checkpoint``
    already understands.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": module.state_dict(),
            "config": config.model_dump_for_checkpoint(),
        },
        path,
    )


def _strict_load_state_dict(module: torch.nn.Module, state_dict: dict) -> None:
    """Load ``state_dict`` into ``module`` and raise on any key mismatch.

    Previously this module used ``strict=False``, which silently swallowed
    missing keys and left those parameters at their random init — producing
    "loaded" checkpoints whose published metrics came from random weights.

    Use ``strict=False`` only to *collect* the mismatch report, then raise a
    ``RuntimeError`` enumerating the offending key names so the failure is
    obvious. This is equivalent to ``strict=True`` but with a stable,
    self-describing error message we control.
    """

    result = module.load_state_dict(state_dict, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint state_dict does not match the model definition. "
            "Loading with strict=False would silently leave parameters at "
            "their random init. Refusing to continue.\n"
            f"  Missing keys ({len(missing)}): {missing}\n"
            f"  Unexpected keys ({len(unexpected)}): {unexpected}"
        )


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

        # Map legacy checkpoint hyperparams to LuminaSTConfig fields
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

        _strict_load_state_dict(module, clean_state_dict)
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
        seed: Optional[int] = None,
        run_dir: Optional[str | Path] = None,
        run_manifest_timestamp: Optional[str] = None,
    ) -> ad.AnnData:
        """
        Full guided enhancement / imputation pipeline.

        Args:
            st_adata: Spatial transcriptomics AnnData input. The method returns
                a copy and does not mutate this object.
            cancer_type: Optional cancer label. If omitted, ``.obs["cancer_type"]``
                is used when present, otherwise the first configured cancer type
                or ``"UNKNOWN"`` is used. Unknown labels map to the registry's
                in-range ``UNKNOWN`` token when available.
            layer: Optional AnnData layer name to enhance. Defaults to ``X``.
            ode_style: ``"correct"`` integrates from ``t_forward`` to 1.0.
                ``"baseline"`` is bug-compatibility only and integrates from
                0.0; do not use it for new results.
            uncond_class: ``"correct"`` uses the CFG null class token.
                ``"baseline"`` is bug-compatibility only and uses class index 0
                as the unconditional branch; do not use it for new results.
            sparsity_style: ``"gene"`` applies per-gene sparsity when a sparsity
                ratio file is configured; any other value uses a per-cell
                percentile threshold. This changes downstream scoring semantics.
            held_out_genes: Optional gene names to zero before encoding for
                held-out gene recovery benchmarks. Compare imputed values at
                these columns against the original observations.
            seed: Optional inference seed. Defaults to ``config.seed`` so
                repeated enhancement calls are reproducible.
            run_dir: Optional directory. When provided, a ``run_manifest.json``
                (see ``lumina_st.experiment.RunManifest``) capturing the seed,
                config snapshot, environment, and git provenance is written
                there. ``None`` (the default) preserves the historical
                behaviour and emits nothing.
            run_manifest_timestamp: Optional injected ISO-8601 timestamp for
                the emitted manifest (deterministic for tests). Ignored unless
                ``run_dir`` is set.

        Returns:
            A copy of the input AnnData with ``layers['imputed']`` (or
            ``layers['imputed_latent']`` for latent-only outputs),
            ``obsm['latent_enhanced']``, and ``obsm['latent_observed']``.
        """
        import numpy as np

        cfg = self.config
        pl.seed_everything(cfg.seed if seed is None else seed, workers=True)
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

        # Validate the expression matrix before it flows into encoding/sampling.
        # NaN/inf cells silently propagate to NaN latents and NaN imputations,
        # and an empty matrix produces opaque downstream errors; reject both with
        # a clear message up front (issue #126).
        expr = np.asarray(expr)
        if expr.ndim != 2:
            raise ValueError(
                f"[LuminaST] enhance() expects a 2-D expression matrix, "
                f"got shape {expr.shape}"
            )
        if expr.shape[0] == 0 or expr.shape[1] == 0:
            raise ValueError(
                f"[LuminaST] enhance() received an empty expression matrix "
                f"(shape {expr.shape}): need at least one cell and one gene"
            )
        if not np.all(np.isfinite(expr)):
            n_nan = int(np.isnan(expr).sum())
            n_inf = int(np.isinf(expr).sum())
            raise ValueError(
                f"[LuminaST] enhance() received a non-finite expression matrix: "
                f"{n_nan} NaN and {n_inf} inf entries. Clean or impute these "
                f"values before enhancement."
            )

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
            seed=cfg.seed if seed is None else seed,
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

        # 7. Optional run-manifest emission (#181). Best-effort: a manifest
        # write must never break or change the enhancement result, so any
        # failure here is swallowed after a warning.
        if run_dir is not None:
            self._emit_run_manifest(
                run_dir=run_dir,
                run_kind="enhance",
                seed=cfg.seed if seed is None else seed,
                n_obs=int(adata.n_obs),
                n_vars=int(adata.n_vars),
                timestamp=run_manifest_timestamp,
            )

        return adata

    def _emit_run_manifest(
        self,
        *,
        run_dir: str | Path,
        run_kind: str,
        seed: Optional[int],
        n_obs: Optional[int] = None,
        n_vars: Optional[int] = None,
        checkpoint_path: Optional[str | Path] = None,
        resume_from: Optional[str | Path] = None,
        timestamp: Optional[str] = None,
    ) -> Optional[Path]:
        """Write a ``run_manifest.json`` into ``run_dir`` (best-effort).

        Shared by ``enhance`` and ``fit``. Captures the seed, config snapshot,
        environment, git provenance, sweep/logging/eval intent, and a
        best-effort dataset id. Never raises: a manifest is provenance, not a
        result, so a failed write only warns and returns ``None``.
        """
        try:
            from ..experiment.run_manifest import RunManifest

            cfg = self.config
            run_dir = Path(run_dir)
            sweep = {"n_obs": n_obs, "n_vars": n_vars, "run_kind": run_kind}
            eval_cadence: dict[str, Any] = {}
            if cfg.early_stopping_patience is not None:
                eval_cadence["early_stopping_patience"] = cfg.early_stopping_patience
            dataset_id = getattr(cfg, "experiment_name", None)

            manifest = RunManifest.create(
                run_id=f"{run_kind}-{getattr(cfg, 'experiment_name', 'lumina')}",
                config=cfg,
                seed=seed,
                timestamp=timestamp,
                sweep_params={k: v for k, v in sweep.items() if v is not None},
                checkpoint_path=checkpoint_path,
                resume_from=resume_from,
                eval_cadence=eval_cadence,
                dataset_id=dataset_id,
            )
            return manifest.to_json(run_dir / "run_manifest.json")
        except Exception as exc:  # pragma: no cover - defensive, never fatal
            print(f"Warning: failed to emit run_manifest.json: {exc}")
            return None

    def fit(
        self,
        trainer: Optional[pl.Trainer] = None,
        *,
        reference_adata: Optional[ad.AnnData] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        run_dir: Optional[str | Path] = None,
        run_manifest_timestamp: Optional[str] = None,
        resume_from: Optional[str | Path] = None,
        **trainer_kwargs,
    ) -> pl.Trainer:
        """Train the flow model on a reference atlas.

        The package-level seed is applied before DataLoader/Trainer construction
        so repeated smoke runs with the same ``LuminaSTConfig.seed`` are
        reproducible.

        When ``val_loader`` is provided and no explicit ``trainer`` is passed,
        validation-driven model selection is wired up (issue #146): a
        ``ModelCheckpoint(monitor="val_loss")`` selects/saves the best epoch and,
        if ``config.early_stopping_patience`` is set, an ``EarlyStopping`` callback
        halts training when ``val_loss`` stops improving. The
        ``LuminaFlowModule.validation_step`` logs ``val_loss`` so these callbacks
        have a metric to monitor.

        When ``run_dir`` is provided (or, by default, the configured
        ``output_dir``), a ``run_manifest.json`` (#181) is written *before*
        training starts so the run is self-describing even if it later crashes.
        Pass ``run_dir=None`` only behaviour is preserved; emission never raises.
        ``resume_from`` is recorded in the manifest's resume field.
        """
        pl.seed_everything(self.config.seed, workers=True)

        # Emit the run manifest up front (#181) so a crashed training run is
        # still self-describing. Defaults to the configured output_dir; callers
        # can redirect or (explicitly) keep the historical no-op by passing a
        # different run_dir. Emission is best-effort and never fatal.
        manifest_dir = run_dir if run_dir is not None else self.config.output_dir
        # Resolve the resumable-checkpoint directory once (#130): explicit
        # config override wins, otherwise the historical output_dir/checkpoints.
        ckpt_dir = (
            Path(self.config.checkpoint_dir)
            if self.config.checkpoint_dir is not None
            else Path(self.config.output_dir) / "checkpoints"
        )
        self._emit_run_manifest(
            run_dir=manifest_dir,
            run_kind="fit",
            seed=self.config.seed,
            checkpoint_path=ckpt_dir,
            resume_from=resume_from,
            timestamp=run_manifest_timestamp,
        )

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
            from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

            callbacks: list[pl.Callback] = []
            if val_loader is not None:
                # Validation-driven model selection (#146): pick the best epoch
                # by val_loss and always keep a resumable ``last.ckpt`` (#130).
                callbacks.append(
                    ModelCheckpoint(
                        monitor="val_loss",
                        mode="min",
                        save_top_k=1,
                        save_last=True,
                        dirpath=str(ckpt_dir),
                        filename="best-{epoch:02d}-{val_loss:.4f}",
                    )
                )
                if self.config.early_stopping_patience is not None:
                    callbacks.append(
                        EarlyStopping(
                            monitor="val_loss",
                            mode="min",
                            patience=self.config.early_stopping_patience,
                        )
                    )
            else:
                # Train-only runs (#130): no val_loss to monitor, but we still
                # write a resumable ``last.ckpt`` so the run can be resumed.
                callbacks.append(
                    ModelCheckpoint(
                        monitor="train_loss",
                        mode="min",
                        save_top_k=1,
                        save_last=True,
                        dirpath=str(ckpt_dir),
                        filename="best-{epoch:02d}-{train_loss:.4f}",
                    )
                )

            trainer = pl.Trainer(
                max_epochs=self.config.max_epochs,
                gradient_clip_val=self.config.gradient_clip_val,
                default_root_dir=str(self.config.output_dir),
                callbacks=callbacks or None,
                **trainer_kwargs,
            )

        # Resume actually resumes (#130): forward the checkpoint path to
        # Lightning so optimizer/scheduler/epoch state are restored, rather
        # than only recording resume_from in the manifest. Only attach
        # ``ckpt_path`` when resuming so the no-resume call signature (and any
        # injected/mock trainer that expects it) is unchanged.
        resume_kwargs = {"ckpt_path": str(resume_from)} if resume_from is not None else {}
        if val_loader is not None:
            trainer.fit(
                self.module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                **resume_kwargs,
            )
        else:
            trainer.fit(
                self.module,
                train_dataloaders=train_loader,
                **resume_kwargs,
            )
        return trainer
