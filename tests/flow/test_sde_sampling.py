"""Unit tests for the stochastic (SDE) sampling path (issue #285).

``FlowSampler.sample_sde`` wires the model velocity into the reverse-time SDE
integrator (``integrators.sde``). These tests verify, on CPU and fast:

1. Output shape correctness for unconditional and guided (imputation) modes.
2. Output is finite.
3. Determinism under a fixed ``torch.manual_seed``.
4. On a trivial affine drift the SDE sample mean tracks the ODE sample mean
   (sanity that the velocity / score / reverse-SDE wiring is consistent).
5. Guided mode without ``x_start`` raises ``ValueError``.
6. The classifier-free-guidance branch (``cfg_scale != 1.0``) executes.

Run with:
    pytest tests/flow/test_sde_sampling.py -q
"""

import sys
from pathlib import Path

import torch

# Allow running the test directly without pip install -e
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from lumina_st.flow import FlowSampler, create_flow_transport


class ConstantTargetVelocity(torch.nn.Module):
    """A trivial affine velocity field pointing toward a fixed target."""

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        # Register as a buffer so ``next(model.parameters())`` still works via a
        # dummy parameter while the target rides along on the right device.
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("target", target)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kw) -> torch.Tensor:
        return self.target - x


def _make_sampler() -> FlowSampler:
    transport = create_flow_transport(path="linear", prediction="velocity")
    return FlowSampler(transport)


def test_sde_unconditional_shape_and_finite() -> None:
    torch.manual_seed(0)
    sampler = _make_sampler()
    model = ConstantTargetVelocity(torch.randn(8, 16))
    out = sampler.sample_sde(model, shape=(8, 16), num_steps=50)
    assert out.shape == (8, 16)
    assert torch.isfinite(out).all()


def test_sde_guided_shape_and_finite() -> None:
    torch.manual_seed(0)
    sampler = _make_sampler()
    model = ConstantTargetVelocity(torch.randn(4, 16))
    x_start = torch.randn(4, 16)
    out = sampler.sample_sde(model, num_steps=50, t_forward=0.3, x_start=x_start)
    assert out.shape == x_start.shape
    assert torch.isfinite(out).all()


def test_sde_determinism_under_fixed_seed() -> None:
    sampler = _make_sampler()
    target = torch.randn(6, 12)
    model = ConstantTargetVelocity(target)

    torch.manual_seed(1234)
    out_a = sampler.sample_sde(model, shape=(6, 12), num_steps=40)
    torch.manual_seed(1234)
    out_b = sampler.sample_sde(model, shape=(6, 12), num_steps=40)

    assert torch.equal(out_a, out_b)


def test_sde_mean_tracks_ode_mean() -> None:
    """On an affine drift the stochastic sample mean tracks the ODE sample mean.

    Generous tolerance + many samples: this is a consistency sanity check on the
    velocity/score/reverse-SDE wiring, not a tight numerical claim.
    """
    sampler = _make_sampler()
    target = torch.zeros(2048, 4)  # broadcastable fixed target -> affine drift
    model = ConstantTargetVelocity(target)

    torch.manual_seed(7)
    sde_out = sampler.sample_sde(model, shape=(2048, 4), num_steps=200)

    ode_out = sampler.sample_ode(
        model, shape=(2048, 4), num_steps=200, solver="euler"
    )

    assert torch.isfinite(sde_out).all()
    sde_mean = sde_out.mean(dim=0)
    ode_mean = ode_out.mean(dim=0)
    assert torch.allclose(sde_mean, ode_mean, atol=0.15), (
        f"SDE mean {sde_mean} does not track ODE mean {ode_mean}"
    )


def test_sde_guided_requires_x_start() -> None:
    sampler = _make_sampler()
    model = ConstantTargetVelocity(torch.randn(4, 16))
    with pytest.raises(ValueError):
        sampler.sample_sde(model, num_steps=10, t_forward=0.3)


def test_sde_cfg_branch_executes() -> None:
    torch.manual_seed(0)
    sampler = _make_sampler()
    model = ConstantTargetVelocity(torch.randn(4, 8))
    out = sampler.sample_sde(model, shape=(4, 8), num_steps=30, cfg_scale=2.0)
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()
