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

import math
from copy import deepcopy
from typing import Callable, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchdiffeq import odeint

from ..config.lumina_config import LuminaSTConfig
from ..flow import create_flow_transport, FlowSampler
from ..models.lumina_transformer import LuminaTransformer

# Fixed-step solvers integrate the guided drift with an explicit hand-written
# loop; everything else is delegated to ``torchdiffeq.odeint`` (adaptive).
_FIXED_STEP_SOLVERS = frozenset({"euler", "heun"})


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
        seed: Optional[int] = None,
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
            seed: Optional local inference seed. When provided, repeated calls are deterministic.
        """
        if seed is not None:
            torch.manual_seed(seed)

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

        # 2. Get the null class token (dropout token). If the checkpoint was
        # trained without CFG dropout, there is no null row; fall back to the
        # conditional branch instead of indexing out of range.
        y_embedder = self.ema_model.y_embedder
        if uncond_class == "baseline":
            null_class = 0
            has_null_token = True
        else:
            null_class = getattr(y_embedder, "num_classes", int(y.max().item()) + 1)
            has_null_token = null_class < y_embedder.embedding_table.num_embeddings
        use_guidance = cfg_scale != 1.0 and has_null_token
        null_y = torch.full_like(y, null_class) if use_guidance else y

        num_steps = self.config.num_sampling_steps
        sampling_method = self.config.sampling_method

        # 3. Integration interval. "baseline" reproduces the stPainter bug of
        # integrating over the full [0, 1] window even though the state starts at
        # t_forward; "correct" integrates the physically meaningful [t_forward, 1].
        t_start = 0.0 if ode_style == "baseline" else float(t_forward)
        t_end = 1.0
        span = t_end - t_start

        if use_guidance:
            y_in = torch.cat([y, null_y], dim=0)
        else:
            y_in = y

        # 4. Build a guidance-aware drift closure (x, t) -> v_guided(x, t). This
        # is solver-agnostic: the same closure is integrated by the fixed-step
        # loops below and by torchdiffeq.odeint for adaptive solvers. CFG (the
        # doubled-batch trick) and the cfg_decay schedule live entirely inside it
        # so guidance works identically under every solver.
        def drift(x: torch.Tensor, t_scalar: float) -> torch.Tensor:
            if use_guidance:
                z_in = torch.cat([x, x], dim=0)
            else:
                z_in = x
            t_in = torch.full((z_in.shape[0],), t_scalar, device=device, dtype=x.dtype)

            out = self.ema_model(z_in, t_in, y_in)
            out = self.transport.prediction_to_velocity(out, z_in, t_in)

            if not use_guidance:
                return out

            # Dynamic CFG schedule based on progress through [t_start, t_end] so
            # the decay is well-defined for adaptive solvers (no step index).
            current_cfg = cfg_scale
            if cfg_decay is not None and span > 0:
                ratio = (t_scalar - t_start) / span
                ratio = min(max(ratio, 0.0), 1.0)
                if cfg_decay == "linear":
                    current_cfg = cfg_scale - (cfg_scale - 1.0) * ratio
                elif cfg_decay == "cosine":
                    current_cfg = 1.0 + (cfg_scale - 1.0) * (1.0 + math.cos(math.pi * ratio)) / 2.0

            v_cond = out[:batch_size]
            v_uncond = out[batch_size:]
            return v_uncond + current_cfg * (v_cond - v_uncond)

        # 5. Dispatch on the configured solver.
        z_t = self._integrate(
            drift,
            z_t,
            t_start=t_start,
            t_end=t_end,
            num_steps=num_steps,
            sampling_method=sampling_method,
        )

        return z_t

    def _integrate(
        self,
        drift: Callable[[torch.Tensor, float], torch.Tensor],
        x0: torch.Tensor,
        *,
        t_start: float,
        t_end: float,
        num_steps: int,
        sampling_method: str,
    ) -> torch.Tensor:
        """Integrate dx/dt = drift(x, t) from ``t_start`` to ``t_end``.

        Fixed-step ``euler``/``heun`` use explicit loops (fast, checkpoint-parity
        with the previous behaviour). Any other method is treated as an adaptive
        solver and delegated to ``torchdiffeq.odeint`` with the configured
        ``atol``/``rtol``, integrating over ``[t_start, t_end]``.
        """
        method = sampling_method.lower()
        steps = max(int(num_steps), 1)
        dt = (t_end - t_start) / steps

        if method == "euler":
            x = x0
            t = t_start
            for _ in range(steps):
                x = x + drift(x, t) * dt
                t += dt
            return x

        if method == "heun":
            x = x0
            t = t_start
            for _ in range(steps):
                k1 = drift(x, t)
                x_pred = x + k1 * dt
                k2 = drift(x_pred, t + dt)
                x = x + 0.5 * (k1 + k2) * dt
                t += dt
            return x

        if method in _FIXED_STEP_SOLVERS:  # pragma: no cover - defensive
            raise ValueError(f"Unhandled fixed-step solver: {sampling_method}")

        # Adaptive solver via torchdiffeq. odeint passes a scalar time tensor; the
        # drift closure expects a Python float, so unwrap it.
        ts = torch.tensor([t_start, t_end], device=x0.device, dtype=x0.dtype)

        def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return drift(x, float(t))

        try:
            sol = odeint(
                ode_func,
                x0,
                ts,
                method=method,
                atol=self.config.atol,
                rtol=self.config.rtol,
            )
        except ValueError as exc:
            raise ValueError(
                f"Unknown sampling_method {sampling_method!r}; expected one of "
                f"euler, heun, or a torchdiffeq solver (e.g. dopri5)."
            ) from exc
        return sol[-1]
