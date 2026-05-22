"""
LuminaTransformer — the core velocity prediction network for LuminaST.

This is a fresh, renamed re-implementation of the DiT-style architecture
originally called "GiT" in the stPainter baseline. All identifiers, docstrings,
and structure have been rewritten for the LuminaST brand.

Key features:
- Patch embedding of latent vectors
- adaLN-Zero blocks (from DiT)
- Joint timestep + class conditioning
- Native support for classifier-free guidance via forward_with_cfg
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention_mlp import Attention, Mlp
from .embeddings import TimestepEmbedder, LabelEmbedder, PatchEmbedder


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LuminaBlock(nn.Module):
    """DiT-style transformer block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(hidden_size, hidden_features=mlp_hidden, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class LuminaFinalLayer(nn.Module):
    """The final layer of the model, projecting features back to the latent space."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LuminaTransformer(nn.Module):
    """
    The main velocity / noise / score prediction network for LuminaST.

    Replaces the original "GiT" for the latent diffusion case.
    """

    def __init__(
        self,
        latent_dim: int,
        patch_size: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        num_classes: int,
        class_dropout_prob: float = 0.1,
        learn_sigma: bool = False,  # kept for compatibility, velocity models usually predict velocity directly
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.learn_sigma = learn_sigma

        self.x_embedder = PatchEmbedder(latent_dim, patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.num_patches = (latent_dim + patch_size - 1) // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            LuminaBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        out_channels = 1  # we predict velocity per patch element
        self.final_layer = LuminaFinalLayer(hidden_size, patch_size, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Frozen sinusoidal positional embedding
        pos = self._get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        # Zero-out final layer and adaLN for stable training (DiT trick)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @staticmethod
    def _get_1d_sincos_pos_embed(embed_dim: int, length: int):
        import numpy as np
        omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
        omega = 1.0 / (10000.0 ** omega)
        pos = np.arange(length, dtype=np.float64)[:, None] * omega[None]
        emb = np.concatenate([np.sin(pos), np.cos(pos)], axis=1)
        return emb

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_patches, patch_size] -> [B, latent_dim]
        return x.reshape(x.shape[0], -1)[:, : self.latent_dim]

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Standard forward (used during training and for the conditional branch of CFG)."""
        x = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        c = t_emb + y_emb

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        """
        Classifier-free guidance forward pass.
        Expects doubled batch (cond + uncond) as the first half of the input.
        """
        model_out = self.forward(x, t, y)
        cond, uncond = torch.split(model_out, len(model_out) // 2, dim=0)
        guided = uncond + cfg_scale * (cond - uncond)
        return torch.cat([guided, guided], dim=0)
