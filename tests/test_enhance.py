"""Regression tests for issue #148.

The default ``t_forward`` controls how far the observed latents are noised before
the reverse ODE integrates them back. With the LinearPath convention
(``t=1`` is data, ``t=0`` is noise) the previous default ``t_forward=0.9`` only
swapped in ~10% noise and then integrated over ``[0.9, 1.0]``, producing a
near-identity "enhancement". The docstring further described it as a "noise
level", which inverted the actual semantics.

This test pins:

1. The default ``t_forward`` is small enough that ``enhance_latent`` actually
   moves the input latents (i.e. is not a near-identity).
2. The :class:`LuminaSTConfig.t_forward` docstring (the comment beside the
   field) states the linear-path convention explicitly.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer
from lumina_st.flow import create_flow_transport


class _ConstantVelocity(nn.Module):
    """Velocity-prediction model that returns a fixed non-zero velocity."""

    def __init__(self, latent_dim: int, num_classes: int, value: float = 0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("v_const", torch.full((latent_dim,), value))
        self.y_embedder = nn.Module()
        self.y_embedder.num_classes = num_classes
        self.y_embedder.embedding_table = nn.Embedding(num_classes, 4)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.v_const.to(x.device, x.dtype).expand_as(x)


def _make_module(latent_dim: int = 4):
    cfg = LuminaSTConfig(
        latent_dim=latent_dim,
        hidden_size=8,
        depth=1,
        num_heads=2,
        guidance_scale=1.0,
        sampling_method="euler",
        num_sampling_steps=64,
        apply_sparsity=False,
    )
    module = LuminaImputer.from_config(cfg).module
    module.transport = create_flow_transport("linear", "velocity")
    module.ema_model = _ConstantVelocity(latent_dim, num_classes=2, value=0.5)
    module.ema_model.eval()
    return cfg, module


def test_t_forward_default_meaningful_enhancement() -> None:
    """Default ``t_forward`` must actually noise the input meaningfully.

    Concretely, calling ``enhance_latent`` with the default config must change
    the input by more than an L2 distance equivalent to the historical
    ``t_forward=0.9`` near-identity (which only displaced the latent by
    ``sqrt(1-0.9) * sigma_x0 ≈ 0.316`` of the noise standard deviation).
    """
    cfg, module = _make_module(latent_dim=8)
    torch.manual_seed(0)
    z = torch.randn(16, cfg.latent_dim)
    y = torch.zeros(16, dtype=torch.long)

    enhanced = module.enhance_latent(z, y, seed=0)

    # The displacement on the linear path due to noising alone is
    #   ||z_noisy - z|| ≈ ||(1 - t) * x0||  (since alpha(t)*z + sigma(t)*x0).
    # The historical t_forward=0.9 default produces a displacement scale of
    # ~0.1 * ||x0||. The new default must be substantially larger so the
    # enhancement is not a near-identity.
    historical_scale = 0.1
    actual_scale = (enhanced - z).norm() / z.norm()
    assert actual_scale > historical_scale, (
        f"enhance_latent with default t_forward={cfg.t_forward} produces a "
        f"relative displacement of {float(actual_scale):.4f}, which is "
        f"<= the historical near-identity bound {historical_scale}. "
        f"The default must noise the observed latents meaningfully."
    )


def test_t_forward_docstring_documents_convention() -> None:
    """The configuration must document the linear-path convention.

    With the LinearPath, ``t=1`` is data and ``t=0`` is noise, so ``t_forward``
    is a *data fraction* (or equivalently, the inverse of a noise level). The
    field's inline doc must say this explicitly so users don't read it as a
    "noise level" (the previous wording) and pick the wrong number.
    """
    src = inspect.getsource(LuminaSTConfig).lower()
    # Find the line that declares the t_forward field, then look at the
    # comment block immediately preceding it.
    lines = src.splitlines()
    field_idx = next(
        (i for i, line in enumerate(lines) if "t_forward" in line and ":" in line and "float" in line),
        None,
    )
    assert field_idx is not None, "t_forward field declaration not found in LuminaSTConfig source"
    # Walk backwards over the comment block that documents this field.
    doc_lines: list[str] = []
    for j in range(field_idx - 1, -1, -1):
        stripped = lines[j].strip()
        if stripped.startswith("#"):
            doc_lines.append(stripped)
        elif stripped == "":
            continue
        else:
            break
    doc_block = "\n".join(doc_lines)
    # The doc block must document the convention (t=1=data / t=0=noise) or
    # equivalently note that t_forward is a data-fraction (not a noise level).
    assert ("t=1" in doc_block and "t=0" in doc_block) or (
        "data fraction" in doc_block
    ), (
        "LuminaSTConfig.t_forward must document the linear-path convention "
        f"(t=1=data, t=0=noise). Doc block: {doc_block!r}"
    )
