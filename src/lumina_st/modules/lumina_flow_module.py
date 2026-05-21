"""
LuminaFlowModule — PyTorch Lightning training & inference module for LuminaST.

This replaces the original `DiffusionModule` + `VAEModule` with a single,
clean, modern LightningModule that:
- Uses the new `FlowTransport` from lumina_st.flow
- Wraps the `LuminaTransformer`
- Handles EMA
- Performs guided imputation with sparsity post-processing
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..config.lumina_config import LuminaSTConfig
from ..flow import create_flow_transport, FlowSampler
from ..models.lumina_transformer import LuminaTransformer


class LuminaFlowModule(pl.LightningModule):
    """Lightning wrapper for training the conditional latent flow model + guided sampling."""

    def __init__(
        self,
        config: LuminaSTConfig,
        transformer: LuminaTransformer,
        vae: Optional[nn.Module] = None,  # pluggable VAE (scvi or internal)
    ):
        super().__init__()
        self.save_hyperparameters(config.model_dump_for_checkpoint())
        self.config = config
        self.transformer = transformer
        self.vae = vae

        # Flow transport (our clean primitives)
        self.transport = create_flow_transport(
            path=config.path_type,
            prediction=config.prediction,
            loss_weight=config.loss_weight,
            train_eps=config.train_eps,
            sample_eps=config.sample_eps,
        )
        self.sampler = FlowSampler(self.transport)

        # EMA copy
        self.ema_decay = config.ema_decay
        self.ema_model = deepcopy(transformer)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.eval()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.transformer.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch  # from ReferenceAtlasDataset
        # Encode to latent if VAE is present
        if self.vae is not None:
            z, _ = self.vae.encode_to_latent(x, y)  # interface to be defined in latents/
            z = z.detach()
        else:
            z = x  # assume already in latent space for now

        loss_dict = self.transport.training_losses(self.transformer, z, {"y": y})
        loss = loss_dict["loss"]

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self._update_ema()

    @torch.no_grad()
    def _update_ema(self):
        for ema_p, p in zip(self.ema_model.parameters(), self.transformer.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    # ------------------------------------------------------------------
    # Sampling / Imputation (core of the paper)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def enhance_latent(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: Optional[float] = None,
        t_forward: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Perform guided imputation starting from partially noised latents.

        Steps:
            1. Forward diffuse z to time t_forward using the learned path
            2. Run CFG sampling from t_forward back to clean data
        """
        cfg_scale = cfg_scale or self.config.guidance_scale
        t_forward = t_forward or self.config.t_forward

        self.ema_model.eval()
        batch_size = z.shape[0]
        device = z.device

        # Create time tensor for forward diffusion
        t = torch.full((batch_size,), t_forward, device=device)

        # 1. Forward diffuse the observed latent to t_forward
        z_noisy, _ = self.transport.get_noisy_xt(z, t)

        # 2. Prepare for CFG: double the batch (conditional + unconditional)
        z_double = torch.cat([z_noisy, z_noisy], dim=0)

        # For unconditional, we use a special "null" class index
        # We assume the model was trained with class_dropout_prob > 0 so it learned the null token
        null_y = torch.full_like(y, self.transformer.y_embedder.num_classes)  # the dropout token
        y_double = torch.cat([y, null_y], dim=0)

        # Use the transformer's CFG forward
        # We will integrate manually with a simple Euler loop for full control (or use sampler)
        # For now, use the high-level sampler but pass the doubled tensors

        # Simpler approach for Phase 2: use the ema model with explicit CFG in a loop
        # (more robust than relying on sampler for partial-t starts)

        dt = t_forward / max(self.config.num_sampling_steps, 1)
        z_t = z_noisy.clone()

        # Get the null class token (dropout token)
        null_class = getattr(self.ema_model.y_embedder, "num_classes", y.max().item() + 1)
        null_y = torch.full_like(y, null_class)

        for step in range(self.config.num_sampling_steps):
            t_current = torch.full((batch_size,), t_forward - step * dt, device=device)

            # Double for CFG
            z_in = torch.cat([z_t, z_t], dim=0)
            t_in = torch.cat([t_current, t_current], dim=0)
            y_in = torch.cat([y, null_y], dim=0)

            with torch.no_grad():
                out = self.ema_model(z_in, t_in, y_in)
                v_cond, v_uncond = torch.split(out, batch_size, dim=0)
                v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

            # Euler step (velocity field points toward data)
            z_t = z_t + v_guided * dt

        return z_t
