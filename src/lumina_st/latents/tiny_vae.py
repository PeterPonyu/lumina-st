"""
Tiny Gaussian VAE for LuminaST testing and 5090 verification.

Very small MLP VAE (encoder → latent → decoder) with Gaussian assumption.
Sufficient to verify the full pipeline runs end-to-end on the RTX 5090
without requiring heavy scvi-tools installation for the initial validation.
"""

from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae_interface import LatentEncoder


class TinyVAE(LatentEncoder):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def encode_to_latent(self, x: torch.Tensor, y: torch.Tensor = None):
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, {"mu": mu, "logvar": logvar, "library": torch.ones(x.shape[0], 1, device=x.device)}

    def decode_from_latent(self, z: torch.Tensor, y: torch.Tensor = None, library=None):
        return self.dec(z)

    def forward(self, x, y=None):
        z, info = self.encode_to_latent(x, y)
        recon = self.decode_from_latent(z, y)
        # KL + recon loss (for optional VAE pretraining)
        mu, logvar = info["mu"], info["logvar"]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        return {"z": z, "recon": recon, "loss": recon_loss + 0.1 * kl}
