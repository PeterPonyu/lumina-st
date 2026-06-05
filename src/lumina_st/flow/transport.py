"""Core flow-matching transport and sampling logic (Lumina / Aether).

This module is the mathematical engine shared by LuminaST (latent gene diffusion)
and Aether3D (multi-modal spatial + expression + cell-type velocity fields).

Design goals for the refactor:
- Zero original strings / class names from the baselines.
- Clear separation between "what path" and "what the model predicts".
- First-class support for classifier-free guidance (CFG).
- Easy to unit-test numerical fidelity on synthetic data.
- Modern Python (dataclasses, type hints, no global state).

The public API that higher-level modules (LuminaFlowModule, AetherFlowModule)
will use is intentionally small:

    transport = create_flow_transport(path="linear", prediction="velocity")
    sampler   = FlowSampler(transport)
    losses    = transport.training_losses(model, x1, model_kwargs)
    x_gen     = sampler.sample_ode(model, shape=(batch, latent_dim), ...)

Note:
    ``sample_ode`` integrates the drift returned by the model/transport. When
    ``cfg_scale != 1.0`` it applies classifier-free guidance by blending the
    conditional drift (``model_kwargs``) with the unconditional drift
    (``uncond_model_kwargs``); see its docstring for the convention.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .path import EPS_ALPHA, InterpolationPath, get_path
from .integrators import ode, sde
from .utils import expand_time_like_data, mean_flat, validate_guidance_scale


class PredictionTarget(str, enum.Enum):
    """What the neural network is trained to regress."""
    VELOCITY = "velocity"
    NOISE = "noise"
    SCORE = "score"


class LossWeighting(str, enum.Enum):
    NONE = "none"
    VELOCITY = "velocity"
    LIKELIHOOD = "likelihood"


@dataclass
class FlowTransport:
    """Encapsulates a probability path + what the model should predict."""

    path: InterpolationPath
    prediction: PredictionTarget = PredictionTarget.VELOCITY
    loss_weight: LossWeighting = LossWeighting.NONE
    train_eps: float = 0.0
    sample_eps: float = 0.0

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------
    def training_losses(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the flow-matching regression loss for a batch.

        The model is expected to receive (xt, t, **model_kwargs) and return
        a velocity / noise / score prediction of the same shape as xt.
        """
        model_kwargs = model_kwargs or {}
        batch = x1.shape[0]
        device = x1.device

        # Sample noise and time
        x0 = torch.randn_like(x1)
        t = torch.rand(batch, device=device) * (1 - self.train_eps) + self.train_eps

        # Interpolate
        _, xt, ut = self.path.plan(t, x0, x1)

        # Model prediction – the model must accept (xt, t, **kwargs)
        pred = model(xt, t, **model_kwargs)

        # Convert prediction target if necessary
        if self.prediction == PredictionTarget.VELOCITY:
            target = ut
        elif self.prediction == PredictionTarget.NOISE:
            target = x0
        else:  # SCORE
            target = self.path.velocity_to_score(ut, xt, t)

        loss = mean_flat((pred - target) ** 2)

        if self.loss_weight == LossWeighting.VELOCITY:
            # Weight by the magnitude of the velocity (common trick)
            loss = loss * mean_flat(ut**2).detach()
        elif self.loss_weight == LossWeighting.LIKELIHOOD:
            # Per-sample weight = sigma(t)^2, the standard score-matching
            # ELBO weighting that ties the regression objective to a data
            # log-likelihood bound. See Song et al. 2021, "Score-Based
            # Generative Modeling through SDEs". For LinearPath this is
            # (1 - t)^2; for GVP / VP paths it follows the path's sigma.
            # Issues #113 / #135 (was a silent no-op falling through to NONE);
            # mirrors PeterPonyu/aether-3d#137 (PR #156).
            sigma_t, _ = self.path.sigma(t)
            loss = loss * (sigma_t ** 2).detach()

        return {"loss": loss.mean(), "t": t, "per_sample": loss}

    # ------------------------------------------------------------------
    # Drift / score functions for sampling
    # ------------------------------------------------------------------
    def prediction_to_velocity(
        self, pred: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Convert the configured model prediction target into an ODE velocity."""
        if self.prediction == PredictionTarget.VELOCITY:
            return pred
        if self.prediction == PredictionTarget.NOISE:
            return self.path.noise_to_velocity(pred, x, t)
        return self.path.score_to_velocity(pred, x, t)

    def get_drift(self, model: nn.Module, **model_kwargs) -> Callable:
        """Return a callable (x, t) -> velocity that the integrators can use."""
        def drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            pred = model(x, t, **model_kwargs)
            return self.prediction_to_velocity(pred, x, t)
        return drift

    # ------------------------------------------------------------------
    # Interval helpers (avoid t=0 / t=1 singularities)
    # ------------------------------------------------------------------
    def check_interval(
        self,
        train_eps: float,
        sample_eps: float,
        *,
        sde: bool = False,
        reverse: bool = False,
    ) -> Tuple[float, float]:
        t0, t1 = 0.0, 1.0
        if self.prediction != PredictionTarget.VELOCITY:
            t0 = max(t0, sample_eps if sde else train_eps)
        if reverse:
            t0, t1 = t1, t0
        return t0, t1

    # ------------------------------------------------------------------
    # Forward diffusion helpers (critical for guided imputation)
    # ------------------------------------------------------------------
    def get_noisy_xt(self, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffuse clean data x1 to time t.
        Returns (x_t, x0_noise) so the caller can also recover the added noise if needed.
        """
        x0 = torch.randn_like(x1)
        _, xt, _ = self.path.plan(t, x0, x1)
        return xt, x0

    def get_velocity(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, **model_kwargs) -> torch.Tensor:
        """Convenience: get model output converted to a velocity prediction."""
        pred = model(x, t, **model_kwargs)
        return self.prediction_to_velocity(pred, x, t)


@dataclass
class FlowSampler:
    """High-level sampler (ODE and SDE) that higher-level modules call."""

    transport: FlowTransport
    model: Optional[nn.Module] = None  # can be set later

    def sample_ode(
        self,
        model: Optional[nn.Module] = None,
        shape: Optional[Tuple[int, ...]] = None,
        *,
        num_steps: Optional[int] = None,
        solver: str = "dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        cfg_scale: float = 1.0,
        model_kwargs: Optional[Dict[str, Any]] = None,
        uncond_model_kwargs: Optional[Dict[str, Any]] = None,
        t_forward: Optional[float] = None,
        x_start: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Integrate the probability-flow ODE from noise (or a noised signal) to data.

        Two modes:

        * **Unconditional sampling** (``t_forward is None``): start from pure
          noise ``x0 ~ N(0, I)`` of the given ``shape`` and integrate the ODE
          over ``[0, 1]``. ``shape`` is required for meaningful sampling; if
          omitted, a tiny placeholder shape is used only for backwards-compatible
          smoke tests.

        * **Guided imputation** (``t_forward`` set): forward-diffuse the observed
          clean signal ``x_start`` to time ``t_forward`` along the configured
          path — ``x_t = alpha(t_forward) * x_start + sigma(t_forward) * x0`` via
          :meth:`FlowTransport.get_noisy_xt` — and integrate the reverse ODE over
          ``[t_forward, 1]``. This is a mathematically sound forward-diffusion
          step; the previous placeholder ``x = randn * t_forward**0.5`` was not a
          valid noising operator and discarded ``x_start`` entirely (issue #251).
          ``x_start`` is required in this mode.

        Classifier-free guidance is applied here when ``cfg_scale != 1.0``: the
        conditional drift (from ``model_kwargs``) and the unconditional drift
        (from ``uncond_model_kwargs``) are combined as
        ``v_uncond + cfg_scale * (v_cond - v_uncond)``. ``uncond_model_kwargs``
        defaults to ``model_kwargs`` with any ``"y"`` label removed so the same
        ``model_kwargs`` shape used elsewhere works out of the box.
        """
        model = model or self.model
        assert model is not None, "No model provided to sampler"

        cfg_scale = validate_guidance_scale(cfg_scale)

        device = next(model.parameters()).device

        if t_forward is not None:
            # Guided imputation: properly forward-diffuse the observed signal.
            if x_start is None:
                raise ValueError(
                    "sample_ode(t_forward=...) performs forward diffusion of a "
                    "clean starting signal: pass x_start (the observed data to be "
                    "partially noised). Leave t_forward=None to sample from pure "
                    "noise."
                )
            x_start = x_start.to(device)
            t0 = float(t_forward)
            t_vec = torch.full(
                (x_start.shape[0],), t0, device=device, dtype=x_start.dtype
            )
            x, _ = self.transport.get_noisy_xt(x_start, t_vec)
        else:
            t0 = 0.0
            if shape is None:
                # Infer from a dummy forward (common pattern)
                dummy = torch.zeros(1, device=device)
                shape = (1, *dummy.shape[1:])  # placeholder – user should pass real shape
            x = torch.randn(shape, device=device)

        cond_kwargs = model_kwargs or {}

        if cfg_scale == 1.0:
            drift = self.transport.get_drift(model, **cond_kwargs)
        else:
            # Real CFG: blend the conditional and unconditional drifts.
            if uncond_model_kwargs is None:
                uncond_model_kwargs = {k: v for k, v in cond_kwargs.items() if k != "y"}
            cond_drift = self.transport.get_drift(model, **cond_kwargs)
            uncond_drift = self.transport.get_drift(model, **uncond_model_kwargs)

            def drift(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                v_cond = cond_drift(x_t, t)
                v_uncond = uncond_drift(x_t, t)
                return v_uncond + cfg_scale * (v_cond - v_uncond)

        integrator = ode(
            drift,
            t0=t0,
            t1=1.0,
            num_steps=num_steps,
            solver_type=solver,
            atol=atol,
            rtol=rtol,
        )
        return integrator(x)

    # SDE sampling (for diversity or likelihood evaluation)
    def sample_sde(
        self,
        model: Optional[nn.Module] = None,
        shape: Optional[Tuple[int, ...]] = None,
        *,
        num_steps: int = 250,
        sampler_type: str = "Euler",
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        cfg_scale: float = 1.0,
        model_kwargs: Optional[Dict[str, Any]] = None,
        uncond_model_kwargs: Optional[Dict[str, Any]] = None,
        t_forward: Optional[float] = None,
        x_start: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Integrate a reverse-time SDE whose marginals match the probability-flow ODE.

        This mirrors :meth:`sample_ode` (same two modes, same classifier-free
        guidance convention) but injects stochasticity. For any non-negative
        diffusion schedule ``w(t)`` the SDE

            ``dx = [ v(x, t) + 0.5 * w(t) * score(x, t) ] dt + sqrt(w(t)) dW``

        shares the marginal densities of the deterministic flow ``dx = v dt``
        (the standard velocity <-> score <-> reverse-SDE relationship; Song et
        al. 2021). The velocity ``v`` is the model's drift; the score is
        recovered from it via :meth:`InterpolationPath.velocity_to_score`, and
        ``w(t)`` comes from :meth:`InterpolationPath.diffusion` (selected by
        ``diffusion_form`` / ``diffusion_norm``).

        Two modes:

        * **Unconditional sampling** (``t_forward is None``): start from pure
          noise ``x0 ~ N(0, I)`` of the given ``shape`` and integrate over
          ``[0, 1]``. ``shape`` is required for meaningful sampling.

        * **Guided imputation** (``t_forward`` set): forward-diffuse the observed
          clean signal ``x_start`` to ``t_forward`` via
          :meth:`FlowTransport.get_noisy_xt` and integrate over
          ``[t_forward, 1]``. ``x_start`` is required in this mode (same
          ``ValueError`` contract as :meth:`sample_ode`).

        Classifier-free guidance, when ``cfg_scale != 1.0``, blends the
        conditional and unconditional velocities as
        ``v_uncond + cfg_scale * (v_cond - v_uncond)`` before the score
        correction is added.

        Raises:
            ValueError: If guided mode is requested without ``x_start``, if the
                guidance scale is invalid, or if the configured diffusion form
                yields a negative coefficient ``w(t)`` (no real ``sqrt``).
        """
        model = model or self.model
        assert model is not None, "No model provided to sampler"

        cfg_scale = validate_guidance_scale(cfg_scale)

        device = next(model.parameters()).device

        if t_forward is not None:
            # Guided imputation: properly forward-diffuse the observed signal.
            if x_start is None:
                raise ValueError(
                    "sample_sde(t_forward=...) performs forward diffusion of a "
                    "clean starting signal: pass x_start (the observed data to be "
                    "partially noised). Leave t_forward=None to sample from pure "
                    "noise."
                )
            x_start = x_start.to(device)
            t0 = float(t_forward)
            t_vec = torch.full(
                (x_start.shape[0],), t0, device=device, dtype=x_start.dtype
            )
            x, _ = self.transport.get_noisy_xt(x_start, t_vec)
        else:
            t0 = 0.0
            if shape is None:
                dummy = torch.zeros(1, device=device)
                shape = (1, *dummy.shape[1:])  # placeholder – user should pass real shape
            x = torch.randn(shape, device=device)

        cond_kwargs = model_kwargs or {}

        if cfg_scale == 1.0:
            velocity_fn = self.transport.get_drift(model, **cond_kwargs)
        else:
            # Real CFG: blend the conditional and unconditional velocities.
            if uncond_model_kwargs is None:
                uncond_model_kwargs = {k: v for k, v in cond_kwargs.items() if k != "y"}
            cond_drift = self.transport.get_drift(model, **cond_kwargs)
            uncond_drift = self.transport.get_drift(model, **uncond_model_kwargs)

            def velocity_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                v_cond = cond_drift(x_t, t)
                v_uncond = uncond_drift(x_t, t)
                return v_uncond + cfg_scale * (v_cond - v_uncond)

        path = self.transport.path

        def diffusion_coeff(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Path-consistent squared diffusion coefficient ``w(t)``, broadcast to ``x_t``.

            Mirrors :meth:`InterpolationPath.diffusion` but evaluates the path's
            ``alpha``/``sigma`` directly so the time tensor is expanded exactly
            once (the path's own ``diffusion`` double-expands ``t`` and is not
            usable on the ``(x, t)`` 1-D interface the integrator supplies).
            """
            te = expand_time_like_data(t, x_t)
            a, da = path.alpha(te)
            s, ds = path.sigma(te)
            if diffusion_form == "constant":
                w = torch.full_like(a, diffusion_norm)
            elif diffusion_form == "SBDM":
                # Natural diffusion of the path's Fokker-Planck form.
                safe_a = torch.where(a.abs() > EPS_ALPHA, a, torch.ones_like(a))
                w = diffusion_norm * (da / safe_a * s**2 - s * ds)
            elif diffusion_form == "sigma":
                w = diffusion_norm * s
            elif diffusion_form == "linear":
                w = diffusion_norm * (1 - te)
            else:
                raise ValueError(
                    f"unsupported diffusion form {diffusion_form!r}; reverse-SDE "
                    "sampling supports 'constant', 'SBDM', 'sigma', 'linear'."
                )
            return torch.broadcast_to(w, x_t.shape)

        def diffusion(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            w = diffusion_coeff(x_t, t)
            if (w < 0).any():
                raise ValueError(
                    f"diffusion form {diffusion_form!r} produced a negative "
                    "coefficient w(t); a reverse SDE needs w(t) >= 0 for a real "
                    "sqrt. Choose a non-negative diffusion form (e.g. 'constant', "
                    "'sigma', 'SBDM' on this path) or adjust diffusion_norm."
                )
            return torch.sqrt(w)

        def drift(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            v = velocity_fn(x_t, t)
            score = path.velocity_to_score(v, x_t, t)
            w = diffusion_coeff(x_t, t)
            return v + 0.5 * w * score

        integrator = sde(
            drift,
            diffusion,
            t0=t0,
            t1=1.0,
            num_steps=num_steps,
            sampler_type=sampler_type,
        )
        return integrator(x)


# ----------------------------------------------------------------------
# Public factory (mirrors the original create_transport but with new names)
# ----------------------------------------------------------------------
def create_flow_transport(
    path: str = "linear",
    prediction: str = "velocity",
    loss_weight: Optional[str] = None,
    train_eps: Optional[float] = None,
    sample_eps: Optional[float] = None,
) -> FlowTransport:
    """Create a FlowTransport with sensible defaults for biology-scale data."""

    path_obj = get_path(path)  # type: ignore[arg-type]

    pred = PredictionTarget(prediction)
    lw = LossWeighting.NONE if loss_weight is None else LossWeighting(loss_weight)

    # Reasonable eps per path family (same logic as original, cleaner)
    if pred != PredictionTarget.VELOCITY:
        train_eps = train_eps or 1e-3
        sample_eps = sample_eps or 1e-3
    else:
        train_eps = train_eps or 0.0
        sample_eps = sample_eps or 0.0

    return FlowTransport(
        path=path_obj,
        prediction=pred,
        loss_weight=lw,
        train_eps=train_eps or 0.0,
        sample_eps=sample_eps or 0.0,
    )
