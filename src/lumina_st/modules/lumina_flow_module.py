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
        ode_style: str = "correct",
        uncond_class: str = "correct",
        cfg_decay: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Perform guided imputation starting from partially noised latents.

        Args:
            z: Input latent tensor of shape (B, latent_dim).
            y: Conditional class label tensor of shape (B,).
            cfg_scale: Guidance scale (default is config.guidance_scale).
            t_forward: Forward diffusion noise level (default is config.t_forward).
            ode_style: Either "correct" (integrate forward from t_forward to 1.0) or
                       "baseline" (integrate forward from 0.0 to 1.0 starting at z_noisy,
                       matching the stPainter bug).
            uncond_class: Either "correct" (use y_embedder.num_classes) or
                          "baseline" (use index 0, matching the stPainter bug).
            cfg_decay: Guidance decay schedule. None for constant, "linear", or "cosine".
        """
        cfg_scale = cfg_scale if cfg_scale is not None else self.config.guidance_scale
        t_forward = t_forward if t_forward is not None else self.config.t_forward

        self.ema_model.eval()
        batch_size = z.shape[0]
        device = z.device

        # Create time tensor for forward diffusion
        t = torch.full((batch_size,), t_forward, device=device)

        # 1. Forward diffuse the observed latent to t_forward
        z_noisy, _ = self.transport.get_noisy_xt(z, t)
        z_t = z_noisy.clone()

        # 2. Get the null class token (dropout token)
        if uncond_class == "baseline":
            null_class = 0
        else:
            null_class = getattr(self.ema_model.y_embedder, "num_classes", y.max().item() + 1)
        null_y = torch.full_like(y, null_class)

        num_steps = self.config.num_sampling_steps

        # Pre-allocate inputs to avoid CPU-GPU synchronization and dynamic memory allocation inside the loop
        z_in = torch.empty(batch_size * 2, z.shape[1], device=device, dtype=z.dtype)
        y_in = torch.cat([y, null_y], dim=0)

        # 3. Pre-compute the time steps and tile them on the GPU
        if ode_style == "baseline":
            dt = 1.0 / max(num_steps, 1)
            t_grid = torch.arange(num_steps, device=device, dtype=z.dtype) * dt
        else:
            dt = (1.0 - t_forward) / max(num_steps, 1)
            t_grid = t_forward + torch.arange(num_steps, device=device, dtype=z.dtype) * dt

        t_in_matrix = t_grid.unsqueeze(1).repeat(1, batch_size * 2)

        # 4. ODE Integration Loop
        for step in range(num_steps):
            # Slices/assignments are in-place, avoiding CUDA allocations
            z_in[:batch_size] = z_t
            z_in[batch_size:] = z_t
            t_in = t_in_matrix[step]

            # Dynamic CFG Schedule (Priority 5)
            if cfg_decay is not None and num_steps > 1:
                ratio = step / (num_steps - 1)
                if cfg_decay == "linear":
                    current_cfg = cfg_scale - (cfg_scale - 1.0) * ratio
                elif cfg_decay == "cosine":
                    import math
                    current_cfg = 1.0 + (cfg_scale - 1.0) * (1.0 + math.cos(math.pi * ratio)) / 2.0
                else:
                    current_cfg = cfg_scale
            else:
                current_cfg = cfg_scale

            out = self.ema_model(z_in, t_in, y_in)
            v_cond = out[:batch_size]
            v_uncond = out[batch_size:]
            v_guided = v_uncond + current_cfg * (v_cond - v_uncond)

            z_t = z_t + v_guided * dt

        return z_t
