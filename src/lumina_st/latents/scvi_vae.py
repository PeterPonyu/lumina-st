"""
Real scvi-tools VAE wrapper for LuminaST.

This allows using a properly trained scVI / SCVI model from the user's 'dl' environment
as the latent encoder, which is the recommended path for real data work.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

try:
    from scvi.model import SCVI
    from scvi.module import VAE
    _HAS_SCVI = True
except ImportError:
    _HAS_SCVI = False
    SCVI = None
    VAE = None


class SCVILatentEncoder(nn.Module):
    """
    Wrapper around a trained scvi.model.SCVI or scvi.module.VAE that implements 
    the LatentEncoder interface expected by LuminaFlowModule and LuminaImputer.
    """

    def __init__(self, vae_module_or_scvi: nn.Module | "SCVI"):
        super().__init__()
        if not _HAS_SCVI:
            raise ImportError("scvi-tools is required for SCVILatentEncoder")
        
        if SCVI is not None and isinstance(vae_module_or_scvi, SCVI):
            self.scvi_model = vae_module_or_scvi
            self.vae_module = vae_module_or_scvi.module
        else:
            self.scvi_model = None
            self.vae_module = vae_module_or_scvi
            
        self.latent_dim = self.vae_module.n_latent

    @classmethod
    def from_path(cls, model_path: str):
        if not _HAS_SCVI:
            raise ImportError("scvi-tools not installed in the current environment")
        m = SCVI.load(model_path)
        return cls(m)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        n_input: int,
        n_batch: int,
        n_hidden: int,
        n_latent: int,
        n_layers: int,
        **kwargs
    ) -> SCVILatentEncoder:
        if not _HAS_SCVI:
            raise ImportError("scvi-tools is required for SCVILatentEncoder")
        
        # Instantiate raw VAE module
        vae_module = VAE(
            n_input=n_input,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            **kwargs
        )
        
        # Load weights from checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)
        
        # Strip 'model.' prefix
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                clean_state_dict[k[6:]] = v
            else:
                clean_state_dict[k] = v
                
        vae_module.load_state_dict(clean_state_dict)
        return cls(vae_module)

    def encode_to_latent(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Ensure correct spatial input shape
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        if y.dim() == 1:
            batch_index = y.long().unsqueeze(1)
        else:
            batch_index = y.long()
            
        inputs = {
            "x": x,
            "batch_index": batch_index,
        }
        
        # Run inference
        out = self.vae_module.inference(**inputs)
        z = out["qz"].mean
        return z, {"library": out["library"]}

    def decode_from_latent(self, z: torch.Tensor, y: torch.Tensor, library: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y.dim() == 1:
            batch_index = y.long().unsqueeze(1)
        else:
            batch_index = y.long()
            
        if library is None:
            # Default to log(10000) as in baseline
            library = torch.full((len(z), 1), 10000.0, dtype=torch.float32, device=z.device)
            library = torch.log(library)
        elif library.max() > 100.0:
            # If the user passed raw library values, log them
            library = torch.log(library)
            
        inputs = {
            "z": z,
            "library": library,
            "batch_index": batch_index,
        }
        
        out = self.vae_module.generative(**inputs)
        return out["px"].mu

