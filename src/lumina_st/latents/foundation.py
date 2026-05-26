"""Optional foundation-model latent encoders for LuminaST.

This module provides a lightweight integration seam for scGPT, scGPT-spatial,
Nicheformer, Geneformer, UCE, and similar single-cell/spatial foundation models
without making any of them mandatory dependencies. Foundation models are used as
**encoders**; LuminaST still requires an explicit decoder/readout to return gene
space after latent flow enhancement. For this branch, the safe drop-in path is:

1. wrap any callable/model that returns a cell embedding;
2. optionally project the embedding to ``latent_dim``;
3. use a documented linear readout decoder for smoke tests or downstream
   fine-tuning.

Heavy model-specific loaders should live in adapter packages or user code until
licenses, checkpoints, vocabularies, and GPU budgets are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .vae_interface import LatentEncoder


FOUNDATION_BACKENDS = {
    "scgpt": {
        "import_name": "scgpt",
        "reference": "https://github.com/bowang-lab/scGPT",
        "spatial_reference": "https://github.com/bowang-lab/scGPT-spatial",
    },
    "nicheformer": {
        "import_name": "nicheformer",
        "reference": "https://github.com/theislab/nicheformer",
    },
    "geneformer": {
        "import_name": "geneformer",
        "reference": "https://huggingface.co/ctheodoris/Geneformer",
    },
    "uce": {
        "import_name": "uce",
        "reference": "https://github.com/snap-stanford/UCE",
    },
}


@dataclass(frozen=True)
class FoundationBackendInfo:
    """Resolved metadata for an optional foundation-model backend."""

    name: str
    import_name: str
    available: bool
    reference: str
    error: str | None = None


def inspect_foundation_backend(name: str) -> FoundationBackendInfo:
    """Return availability metadata without importing heavyweight checkpoints."""
    key = name.lower().replace("-", "_")
    if key not in FOUNDATION_BACKENDS:
        supported = ", ".join(sorted(FOUNDATION_BACKENDS))
        raise ValueError(f"unknown foundation backend {name!r}; supported: {supported}")
    meta = FOUNDATION_BACKENDS[key]
    try:
        import_module(meta["import_name"])
    except Exception as exc:  # import errors vary across optional model packages
        return FoundationBackendInfo(
            name=key,
            import_name=meta["import_name"],
            available=False,
            reference=meta["reference"],
            error=str(exc),
        )
    return FoundationBackendInfo(
        name=key,
        import_name=meta["import_name"],
        available=True,
        reference=meta["reference"],
    )


class FoundationLatentEncoder(LatentEncoder):
    """Adapter that makes an embedding model satisfy LuminaST's LatentEncoder API.

    Parameters
    ----------
    encoder:
        Callable or ``nn.Module`` returning a tensor embedding for ``x``. If the
        callable accepts ``(x, y)`` it may consume labels; otherwise ``x`` only is
        retried for compatibility with foundation-model wrappers.
    input_dim:
        Gene/input dimension expected by the decoder readout.
    embedding_dim:
        Dimension returned by the foundation encoder before projection.
    latent_dim:
        Dimension consumed by LuminaST's latent flow.
    decoder:
        Optional readout from latent to gene space. If omitted, a linear decoder
        is created and must be trained/calibrated before paper claims.
    freeze_encoder:
        Freeze the wrapped foundation encoder by default to avoid accidental
        fine-tuning of large optional models during smoke tests.
    """

    def __init__(
        self,
        encoder: Callable[..., torch.Tensor] | nn.Module,
        input_dim: int,
        embedding_dim: int,
        latent_dim: int,
        decoder: Optional[nn.Module] = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder if isinstance(encoder, nn.Module) else _CallableModule(encoder)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.project = nn.Identity() if embedding_dim == latent_dim else nn.Linear(embedding_dim, latent_dim)
        self.decoder = decoder if decoder is not None else nn.Linear(latent_dim, input_dim)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            self.encoder.eval()

    def _call_encoder(self, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        try:
            embedding = self.encoder(x, y)
        except TypeError:
            embedding = self.encoder(x)
        if isinstance(embedding, dict):
            for key in ("cell_emb", "embedding", "embeddings", "latent", "z"):
                if key in embedding:
                    embedding = embedding[key]
                    break
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("foundation encoder must return a tensor or a dict containing a tensor embedding")
        if embedding.ndim != 2:
            raise ValueError(f"foundation encoder must return shape (cells, features), got {tuple(embedding.shape)}")
        if embedding.shape[1] != self.embedding_dim:
            raise ValueError(f"foundation embedding dim mismatch: got {embedding.shape[1]}, expected {self.embedding_dim}")
        return embedding

    def encode_to_latent(self, x: torch.Tensor, y: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        embedding = self._call_encoder(x, y)
        z = self.project(embedding)
        return z, {"foundation_embedding": embedding, "library": torch.ones(x.shape[0], 1, device=x.device)}

    def decode_from_latent(self, z: torch.Tensor, y: torch.Tensor | None = None, library: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        z, info = self.encode_to_latent(x, y)
        recon = self.decode_from_latent(z, y, info.get("library"))
        return {"z": z, "recon": recon, **info}


class _CallableModule(nn.Module):
    def __init__(self, fn: Callable[..., torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def rank_foundation_options(prefer_spatial: bool = True) -> list[str]:
    """Return recommended backend order for LuminaST planning docs/tests."""
    if prefer_spatial:
        return ["nicheformer", "scgpt", "geneformer", "uce"]
    return ["scgpt", "geneformer", "uce", "nicheformer"]
