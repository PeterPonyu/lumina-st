"""Regression tests for issue #251.

``FlowSampler.sample_ode(t_forward=...)`` must perform a mathematically sound
forward diffusion of the observed clean signal along the configured path
(``x_t = alpha(t_forward) * x_start + sigma(t_forward) * x0``) and integrate the
reverse ODE over ``[t_forward, 1]`` — instead of the old placeholder
``x = randn * t_forward ** 0.5`` which discarded ``x_start`` and was not a valid
noising operator.
"""

from __future__ import annotations

import pytest
import torch

from lumina_st.flow.transport import FlowSampler, create_flow_transport


class _ZeroVelocityModel(torch.nn.Module):
    """A model whose velocity prediction is identically zero.

    With a zero drift the probability-flow ODE keeps its state constant, so the
    sampler output equals exactly the forward-diffused starting state — which is
    what we want to assert against.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Gives .parameters() a tensor so the sampler can read a device.
        self._p = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros_like(x)


def test_t_forward_without_x_start_raises():
    sampler = FlowSampler(create_flow_transport(path="linear", prediction="velocity"))
    model = _ZeroVelocityModel(4)
    with pytest.raises(ValueError, match="x_start"):
        sampler.sample_ode(model, shape=(3, 4), t_forward=0.5)


def test_forward_diffusion_matches_path_not_placeholder():
    transport = create_flow_transport(path="linear", prediction="velocity")
    sampler = FlowSampler(transport)
    model = _ZeroVelocityModel(4)

    x_start = torch.randn(3, 4)
    t_forward = 0.7

    torch.manual_seed(123)
    out = sampler.sample_ode(model, t_forward=t_forward, x_start=x_start)

    # Reconstruct the expected forward-diffused start: get_noisy_xt draws the
    # first randn_like(x_start), so re-seeding reproduces the same x0.
    torch.manual_seed(123)
    x0 = torch.randn_like(x_start)
    expected = t_forward * x_start + (1.0 - t_forward) * x0  # LinearPath alpha=t, sigma=1-t

    assert torch.allclose(out, expected, atol=1e-5)

    # The old placeholder (randn * sqrt(t_forward)) ignored x_start entirely; the
    # corrected output must depend on x_start.
    out2 = sampler.sample_ode(model, t_forward=t_forward, x_start=x_start + 5.0)
    assert not torch.allclose(out, out2)


def test_unconditional_sampling_still_works():
    """t_forward=None keeps the noise->data path over [0, 1]."""
    sampler = FlowSampler(create_flow_transport(path="linear", prediction="velocity"))
    model = _ZeroVelocityModel(4)
    out = sampler.sample_ode(model, shape=(2, 4))
    assert out.shape == (2, 4)
    assert torch.all(torch.isfinite(out))
