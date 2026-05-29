"""Regression tests for flow-matching loss weighting.

Issues #113 / #135 — `LossWeighting.LIKELIHOOD` is a public enum value
and a valid `create_flow_transport(loss_weight="likelihood")` argument,
but `training_losses` only branched on VELOCITY. Selecting `"likelihood"`
silently fell through to unweighted NONE behaviour, so a likelihood
ablation actually produced no weighting at all.

Same issue addressed in PeterPonyu/aether-3d#137 (PR #156); this mirrors
its sigma(t)^2 ELBO-weighting semantics.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lumina_st.flow import create_flow_transport
from lumina_st.flow.transport import LossWeighting


class _ConstVelocityModel(nn.Module):
    """Predicts zeros — keeps the regression target the source of variation."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return torch.zeros_like(x) + self.dummy


def test_likelihood_weighting_applied() -> None:
    """LIKELIHOOD multiplies the per-sample loss by sigma(t)^2 (issues #113/#135)."""
    torch.manual_seed(42)
    x1 = torch.randn(8, 16)

    transport_none = create_flow_transport(
        path="linear", prediction="velocity", loss_weight="none"
    )
    transport_lik = create_flow_transport(
        path="linear", prediction="velocity", loss_weight="likelihood"
    )

    model = _ConstVelocityModel()

    # Same seed for both so the only difference is the weighting branch.
    torch.manual_seed(0)
    loss_none = transport_none.training_losses(model, x1)
    torch.manual_seed(0)
    loss_lik = transport_lik.training_losses(model, x1)

    assert torch.allclose(loss_none["t"], loss_lik["t"]), (
        "time draws differ; weighting comparison would be confounded"
    )

    # Linear path: sigma(t) = 1 - t, so the weight is (1 - t)^2.
    expected = loss_none["per_sample"] * (1.0 - loss_lik["t"]) ** 2
    assert torch.allclose(loss_lik["per_sample"], expected, atol=1e-6), (
        "LIKELIHOOD weighting did not multiply per-sample loss by sigma(t)^2"
    )

    # Must clearly differ from NONE — i.e. no silent no-op.
    assert not torch.allclose(loss_lik["per_sample"], loss_none["per_sample"]), (
        "LIKELIHOOD weighting is a no-op vs NONE (issues #113/#135 regression)"
    )


def test_likelihood_enum_recognised() -> None:
    """The factory must accept loss_weight='likelihood' without raising."""
    transport = create_flow_transport(
        path="linear", prediction="velocity", loss_weight="likelihood"
    )
    assert transport.loss_weight == LossWeighting.LIKELIHOOD
