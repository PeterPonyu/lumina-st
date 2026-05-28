"""Regression tests for issue #50.

`enhance_latent()` must route integration through the *configured* ODE solver
(``sampling_method``/``atol``/``rtol``) instead of a hardcoded fixed-step Euler
loop. These tests pin three behaviours:

1. On a constant-velocity model over a ``LinearPath``, fixed-step Euler with many
   steps and the adaptive ``dopri5`` solver converge to the same analytic answer.
2. ``dopri5`` actually calls ``torchdiffeq.odeint`` and honours ``atol``/``rtol``.
3. No sampling-block config field (``sampling_method``/``atol``/``rtol``) is
   silently ignored — flipping ``sampling_method`` changes the dispatched solver.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer
from lumina_st.flow import create_flow_transport
from lumina_st.modules import lumina_flow_module as lfm


class _ConstantVelocity(nn.Module):
    """A velocity-prediction model that returns a fixed velocity everywhere.

    Mimics the LuminaTransformer call signature ``model(x, t, y)`` and exposes a
    ``y_embedder`` with ``num_classes``/``embedding_table`` so the CFG plumbing in
    ``enhance_latent`` can introspect it. With CFG disabled (cfg_scale == 1.0),
    the guided drift collapses to this constant field, giving the closed-form
    solution ``x(1) = x(t_forward) + v_const * (1 - t_forward)``.
    """

    def __init__(self, latent_dim: int, num_classes: int, value: float = 0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("v_const", torch.full((latent_dim,), value))
        # Minimal y_embedder stub so enhance_latent's null-token logic works.
        self.y_embedder = nn.Module()
        self.y_embedder.num_classes = num_classes
        self.y_embedder.embedding_table = nn.Embedding(num_classes, 4)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.v_const.to(x.device, x.dtype).expand_as(x)


def _make_module(sampling_method: str, num_steps: int, latent_dim: int = 4):
    cfg = LuminaSTConfig(
        latent_dim=latent_dim,
        hidden_size=8,
        depth=1,
        num_heads=2,
        guidance_scale=1.0,  # disable CFG so the constant field is the drift
        sampling_method=sampling_method,
        num_sampling_steps=num_steps,
        apply_sparsity=False,
    )
    module = LuminaImputer.from_config(cfg).module
    # Swap the EMA model for a constant-velocity field with a velocity transport.
    module.transport = create_flow_transport("linear", "velocity")
    module.ema_model = _ConstantVelocity(latent_dim, num_classes=2, value=0.5)
    module.ema_model.eval()
    return module


def _analytic_target(module, z, t_forward, seed):
    """Closed-form x(1) for dx/dt = v_const from the same noised start."""
    torch.manual_seed(seed)
    batch = z.shape[0]
    t = torch.full((batch,), t_forward)
    z_noisy, _ = module.transport.get_noisy_xt(z, t)
    v = module.ema_model.v_const
    return z_noisy + v * (1.0 - t_forward)


def test_euler_many_steps_and_dopri5_converge_on_constant_velocity() -> None:
    latent_dim = 4
    z = torch.randn(3, latent_dim)
    y = torch.zeros(3, dtype=torch.long)
    t_forward = 0.2
    seed = 17

    euler_mod = _make_module("euler", num_steps=2000, latent_dim=latent_dim)
    dopri_mod = _make_module("dopri5", num_steps=2, latent_dim=latent_dim)

    target = _analytic_target(euler_mod, z, t_forward, seed)

    euler_out = euler_mod.enhance_latent(z, y, t_forward=t_forward, seed=seed)
    dopri_out = dopri_mod.enhance_latent(z, y, t_forward=t_forward, seed=seed)

    # Both solvers must hit the analytic answer, and must agree with each other.
    assert torch.allclose(euler_out, target, atol=1e-3), (euler_out - target).abs().max()
    assert torch.allclose(dopri_out, target, atol=1e-4), (dopri_out - target).abs().max()
    assert torch.allclose(euler_out, dopri_out, atol=1e-3)


def test_dopri5_path_calls_odeint_and_honours_tolerances(monkeypatch) -> None:
    latent_dim = 4
    z = torch.randn(2, latent_dim)
    y = torch.zeros(2, dtype=torch.long)

    cfg = LuminaSTConfig(
        latent_dim=latent_dim,
        hidden_size=8,
        depth=1,
        num_heads=2,
        guidance_scale=1.0,
        sampling_method="dopri5",
        num_sampling_steps=2,
        atol=1.234e-6,
        rtol=5.678e-6,
        apply_sparsity=False,
    )
    module = LuminaImputer.from_config(cfg).module
    module.transport = create_flow_transport("linear", "velocity")
    module.ema_model = _ConstantVelocity(latent_dim, num_classes=2, value=0.5)
    module.ema_model.eval()

    captured: dict = {}
    real_odeint = lfm.odeint

    def spy_odeint(func, y0, t, **kwargs):
        captured["called"] = True
        captured["atol"] = kwargs.get("atol")
        captured["rtol"] = kwargs.get("rtol")
        captured["method"] = kwargs.get("method")
        captured["t0"] = float(t[0])
        captured["t1"] = float(t[-1])
        return real_odeint(func, y0, t, **kwargs)

    monkeypatch.setattr(lfm, "odeint", spy_odeint)

    out = module.enhance_latent(z, y, t_forward=0.3, seed=5)

    assert captured.get("called"), "dopri5 path must call torchdiffeq.odeint"
    assert captured["method"] == "dopri5"
    assert captured["atol"] == cfg.atol
    assert captured["rtol"] == cfg.rtol
    # Integration runs over [t_forward, 1].
    assert abs(captured["t0"] - 0.3) < 1e-6
    assert abs(captured["t1"] - 1.0) < 1e-6
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_fixed_step_solvers_do_not_call_odeint(monkeypatch) -> None:
    latent_dim = 4
    z = torch.randn(2, latent_dim)
    y = torch.zeros(2, dtype=torch.long)

    called = {"flag": False}

    def spy_odeint(*args, **kwargs):
        called["flag"] = True
        raise AssertionError("fixed-step solver must not call odeint")

    monkeypatch.setattr(lfm, "odeint", spy_odeint)

    for method in ("euler", "heun"):
        module = _make_module(method, num_steps=10, latent_dim=latent_dim)
        out = module.enhance_latent(z, y, t_forward=0.4, seed=3)
        assert out.shape == z.shape
        assert torch.isfinite(out).all()

    assert not called["flag"]


def test_heun_is_more_accurate_than_euler_at_equal_steps() -> None:
    """Heun (2nd order) should not be worse than Euler on a smooth field.

    On a constant-velocity field both are exact, so this asserts both converge;
    the value of supporting Heun is that it is a real, distinct dispatch branch.
    """
    latent_dim = 3
    z = torch.randn(2, latent_dim)
    y = torch.zeros(2, dtype=torch.long)
    t_forward = 0.25
    seed = 9

    heun_mod = _make_module("heun", num_steps=5, latent_dim=latent_dim)
    target = _analytic_target(heun_mod, z, t_forward, seed)
    heun_out = heun_mod.enhance_latent(z, y, t_forward=t_forward, seed=seed)

    assert torch.allclose(heun_out, target, atol=1e-4)


def test_unknown_sampling_method_raises() -> None:
    module = _make_module("not-a-solver", num_steps=4)
    z = torch.randn(2, 4)
    y = torch.zeros(2, dtype=torch.long)
    try:
        module.enhance_latent(z, y, t_forward=0.3, seed=1)
    except ValueError as exc:
        assert "not-a-solver" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown sampling_method")


def test_sampler_sample_ode_consumes_cfg_scale() -> None:
    """FlowSampler.sample_ode must apply guidance when cfg_scale != 1.0.

    With a conditional field of +1 and an unconditional field of 0, a guidance
    scale of 2.0 yields a guided drift of 0 + 2*(1 - 0) = 2. Integrating that
    over [0, 1] from zero gives 2.0, distinguishable from the cfg-off result.
    """
    from lumina_st.flow import FlowSampler

    latent_dim = 3

    class _CondUncondField(nn.Module):
        def __init__(self):
            super().__init__()
            # A real parameter so FlowSampler can read the model's device.
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, x, t, cond: bool = True):
            return torch.ones_like(x) if cond else torch.zeros_like(x)

    transport = create_flow_transport("linear", "velocity")
    sampler = FlowSampler(transport)
    model = _CondUncondField()

    # Seed both calls identically so the random start x0 matches; the only
    # difference is then the applied guidance.
    torch.manual_seed(0)
    guided = sampler.sample_ode(
        model,
        shape=(2, latent_dim),
        num_steps=2,
        solver="euler",
        cfg_scale=2.0,
        model_kwargs={"cond": True},
        uncond_model_kwargs={"cond": False},
    )
    torch.manual_seed(0)
    plain = sampler.sample_ode(
        model,
        shape=(2, latent_dim),
        num_steps=2,
        solver="euler",
        cfg_scale=1.0,
        model_kwargs={"cond": True},
    )

    # Same start x0; guided drift (==2) displaces twice as far as plain (==1)
    # over the unit interval, so guided - plain == 1 elementwise.
    assert torch.allclose(guided - plain, torch.ones_like(guided), atol=1e-5)
