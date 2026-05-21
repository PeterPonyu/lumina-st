"""
Proper pytest tests for LuminaST (and shared flow).

Run with:
    pytest tests/ -q --tb=line
"""

import pytest
import torch
import numpy as np

from lumina_st.flow import create_flow_transport, LinearPath
from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.latents.tiny_vae import TinyVAE
from lumina_st.models.lumina_transformer import LuminaTransformer


def test_linear_path_velocity():
    path = LinearPath()
    t = torch.tensor([0.3, 0.7])
    x0 = torch.randn(2, 8)
    x1 = torch.randn(2, 8)
    _, _, ut = path.plan(t, x0, x1)
    assert torch.allclose(ut[0], x1[0] - x0[0], atol=1e-5)


def test_flow_transport_loss_finite():
    transport = create_flow_transport("linear", "velocity")

    class DummyModel(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            return x  # velocity prediction = identity for test

    model = DummyModel()
    x1 = torch.randn(4, 16)
    losses = transport.training_losses(model, x1, {"y": torch.zeros(4, dtype=torch.long)})
    assert torch.isfinite(losses["loss"])


def test_lumina_transformer_shapes():
    model = LuminaTransformer(
        latent_dim=32, patch_size=1, hidden_size=32, depth=2,
        num_heads=2, mlp_ratio=4.0, num_classes=5, class_dropout_prob=0.1
    )
    x = torch.randn(3, 32)
    t = torch.rand(3)
    y = torch.randint(0, 5, (3,))
    out = model(x, t, y)
    assert out.shape == (3, 32)


def test_tiny_vae_roundtrip():
    vae = TinyVAE(48, 16)
    x = torch.randn(5, 48)
    z, _ = vae.encode_to_latent(x)
    recon = vae.decode_from_latent(z)
    assert recon.shape == x.shape
