"""Tests for configurable flow-matching time sampling (issue #136).

`FlowTransport.training_losses` historically drew the regression time
uniformly: ``t = torch.rand(batch) * (1 - train_eps) + train_eps``. This adds
two SD3-style alternatives selected by ``time_sampling``:

* ``"logit_normal"``: ``t = sigmoid(mean + std * z)``, ``z ~ N(0, 1)`` —
  concentrates training time near ``sigmoid(mean)``.
* ``"cosmap"``: the SD3 cosine map ``t = 1 - 1/(tan(pi/2 * u) + 1)``,
  ``u ~ U(0, 1)``.

The ``"uniform"`` default must remain bit-identical to the legacy draw so
existing seeded loss/flow tests are unaffected.
"""

from __future__ import annotations

import math

import torch

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.flow import create_flow_transport


def test_config_default_time_sampling_is_uniform() -> None:
    """The config default preserves historical uniform-time behaviour."""
    cfg = LuminaSTConfig()
    assert cfg.time_sampling == "uniform"
    assert cfg.logit_normal_mean == 0.0
    assert cfg.logit_normal_std == 1.0


def test_uniform_is_bit_identical_to_legacy_formula() -> None:
    """Uniform sampling reproduces the legacy ``torch.rand`` draw exactly."""
    transport = create_flow_transport(
        path="linear", prediction="velocity", time_sampling="uniform"
    )
    batch, device = 64, torch.device("cpu")
    train_eps = transport.train_eps

    torch.manual_seed(1234)
    t_actual = transport._sample_time(batch, device)

    torch.manual_seed(1234)
    t_legacy = torch.rand(batch, device=device) * (1 - train_eps) + train_eps

    assert torch.equal(t_actual, t_legacy)


def test_logit_normal_in_unit_interval_and_concentrates_near_sigmoid_mean() -> None:
    """logit_normal draws lie in (0, 1) and concentrate near sigmoid(mean)."""
    mean, std = 0.7, 1.0
    transport = create_flow_transport(
        path="linear",
        prediction="velocity",
        time_sampling="logit_normal",
        logit_normal_mean=mean,
        logit_normal_std=std,
    )
    torch.manual_seed(0)
    t = transport._sample_time(20000, torch.device("cpu"))

    assert torch.isfinite(t).all()
    assert (t > 0).all() and (t < 1).all()

    expected_median = 1.0 / (1.0 + math.exp(-mean))  # sigmoid(mean)
    # Median of sigmoid(N(mean, std)) is exactly sigmoid(mean); generous tol.
    assert abs(t.median().item() - expected_median) < 0.05


def test_cosmap_in_unit_interval_and_finite() -> None:
    """cosmap draws are finite and lie strictly in (0, 1)."""
    transport = create_flow_transport(
        path="linear", prediction="velocity", time_sampling="cosmap"
    )
    torch.manual_seed(7)
    t = transport._sample_time(20000, torch.device("cpu"))

    assert torch.isfinite(t).all()
    assert (t > 0).all() and (t < 1).all()
