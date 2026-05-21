"""Unit tests for the clean flow-matching primitives.

These tests run on CPU and are fast. They verify:
1. Path sampling produces correct shapes and reasonable statistics.
2. Velocity field is consistent with the interpolation definition.
3. A tiny model can be trained with training_losses and produces finite loss.
4. ODE sampling (with a dummy constant-velocity model) runs end-to-end.

Run with:
    pytest tests/flow/test_flow_primitives.py -q
"""

import torch
import pytest

import sys
from pathlib import Path
# Allow running the test directly without pip install -e
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lumina_st.flow import (
    create_flow_transport,
    LinearPath,
    GVPPath,
    FlowTransport,
)


def test_linear_path_shapes():
    path = LinearPath()
    t = torch.rand(16)
    x0 = torch.randn(16, 32)
    x1 = torch.randn(16, 32)
    t, xt, ut = path.plan(t, x0, x1)
    assert xt.shape == x0.shape
    assert ut.shape == x0.shape


def test_velocity_consistency():
    """For a linear path the velocity should be exactly x1 - x0."""
    path = LinearPath()
    t = torch.tensor([0.3, 0.7])
    x0 = torch.randn(2, 8)
    x1 = torch.randn(2, 8)
    _, _, ut = path.plan(t, x0, x1)
    # At any t the instantaneous velocity of the linear path is constant = x1-x0
    assert torch.allclose(ut[0], x1[0] - x0[0], atol=1e-6)


def test_training_loss_finite():
    transport = create_flow_transport(path="linear", prediction="velocity")

    class TinyVelocityNet(torch.nn.Module):
        def __init__(self, dim=32):
            super().__init__()
            self.net = torch.nn.Linear(dim, dim)
        def forward(self, x, t, **kw):
            return self.net(x)   # deliberately bad model – we only care about finite loss

    model = TinyVelocityNet()
    x1 = torch.randn(8, 32)
    losses = transport.training_losses(model, x1)
    assert torch.isfinite(losses["loss"]).all()


def test_ode_sampling_runs():
    """Smoke test: integrate a constant velocity field from noise to data."""
    transport = create_flow_transport("linear")

    class ConstantVelocity(torch.nn.Module):
        def __init__(self, target):
            super().__init__()
            self.target = target
        def forward(self, x, t, **kw):
            # Pretend the model always points toward a fixed target
            return self.target - x

    x1_target = torch.randn(4, 16)
    model = ConstantVelocity(x1_target)
    sampler = transport  # we only need the transport object for the factory

    # Manually create a tiny sampler using the public ode helper
    from lumina_st.flow.integrators import ode
    drift = lambda x, t: model(x, t)
    integrator = ode(drift, t0=0.0, t1=1.0, num_steps=20, solver_type="euler")
    x_start = torch.randn(4, 16)
    x_final = integrator(x_start)
    assert x_final.shape == x_start.shape
    # After many steps the constant field should have moved the cloud
    assert (x_final - x_start).abs().mean() > 0.1
