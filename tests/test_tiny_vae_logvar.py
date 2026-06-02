"""Regression test for TinyVAE log-variance clamping.

Without clamping, a large encoder ``logvar`` overflows ``exp()`` to ``inf`` in
both the reparameterization (``encode_to_latent``) and the KL term
(``forward``), corrupting downstream flow training. These tests assert finite
outputs under an extreme logvar while leaving the normal regime unchanged.
"""

import torch

from lumina_st.latents.tiny_vae import TinyVAE, LOGVAR_MIN, LOGVAR_MAX


def _vae_with_logvar_bias(bias: float) -> TinyVAE:
    vae = TinyVAE(input_dim=20, latent_dim=4, hidden=8)
    with torch.no_grad():
        vae.logvar.bias.fill_(bias)
    return vae


def test_encode_finite_under_extreme_logvar():
    vae = _vae_with_logvar_bias(90.0)
    z, info = vae.encode_to_latent(torch.randn(5, 20))
    assert torch.isfinite(z).all()
    assert torch.isfinite(info["logvar"]).all()
    # logvar must be clamped to the documented range
    assert float(info["logvar"].max()) <= LOGVAR_MAX
    assert float(info["logvar"].min()) >= LOGVAR_MIN


def test_loss_finite_under_extreme_logvar():
    vae = _vae_with_logvar_bias(90.0)
    out = vae(torch.randn(5, 20))
    assert torch.isfinite(out["loss"]).all()
    assert torch.isfinite(out["z"]).all()


def test_normal_regime_unchanged():
    """In the normal logvar range the clamp is a no-op and outputs are finite."""
    vae = _vae_with_logvar_bias(0.0)
    out = vae(torch.randn(8, 20))
    assert torch.isfinite(out["loss"]).all()
    _, info = vae.encode_to_latent(torch.randn(8, 20))
    assert float(info["logvar"].abs().max()) < 5.0  # well inside the clamp bounds
