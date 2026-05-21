"""
Pluggable Latent Encoder interface for LuminaST.

This allows LuminaST to work with:
- scvi-tools VAE (recommended for production)
- A lightweight internal VAE (for minimal dependencies)
- Identity encoder (for unit tests and flow-only development)

The interface is deliberately small and AnnData-agnostic at the tensor level.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn


class LatentEncoder(ABC, nn.Module):
    """
    Abstract interface that any latent encoder must implement.

    Expected behavior:
        encode(x, y) -> (z_mean_or_sample, extra_info)
        decode(z, y, library) -> reconstructed_expression
    """

    @abstractmethod
    def encode_to_latent(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return (z, info_dict) where z is the latent representation."""
        ...

    @abstractmethod
    def decode_from_latent(
        self, z: torch.Tensor, y: torch.Tensor, library: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Reconstruct gene-space expression from latent z."""
        ...

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Default forward for training the VAE itself (if trainable)."""
        raise NotImplementedError("This encoder does not support end-to-end VAE training")


class IdentityLatentEncoder(LatentEncoder):
    """
    Trivial encoder that treats the input as the "latent".
    Useful for:
    - Testing the flow model in isolation
    - When the user already works in a compressed space
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def encode_to_latent(self, x: torch.Tensor, y: torch.Tensor):
        return x, {"library": torch.ones(x.shape[0], 1, device=x.device)}

    def decode_from_latent(self, z: torch.Tensor, y: torch.Tensor, library=None):
        return z  # identity

    def forward(self, x, y):
        return {"z": x, "recon": x}
