"""Regression tests for VPPath endpoint numerics (issue #125).

VPPath.sigma divided ``ds`` by ``-2 * s`` and used ``sqrt(1 - exp(2*log_a))``
without any boundary clamp. At ``t = 1`` the variance-preserving path has
``log_a -> 0`` so:

* ``s = sqrt(1 - exp(0)) = sqrt(0) = 0``
* ``ds = exp(0) * 2 * d_log_mean(1) / (-2 * 0) -> +/-inf``

and for parameter regimes that drive ``log_a`` slightly positive numerically,
``1 - exp(2*log_a)`` can dip below zero so ``sqrt`` returns NaN. Either
failure mode propagates straight into the training loss and the
velocity↔score conversions.

Same epsilon shape as the already-open Phase-2 fix in PR #157 (cross-repo
velocity↔score boundary cluster).
"""

from __future__ import annotations

import pytest
import torch

from lumina_st.flow.path import get_path


# Mirrors PR #157's BOUNDARY_TS grid; ``1 - 1e-6`` and ``1`` are the cases
# that fail on origin/main.
_EPS = 1e-6
BOUNDARY_TS = [0.0, _EPS, 0.5, 1.0 - _EPS, 1.0]


@pytest.mark.parametrize("t_val", BOUNDARY_TS)
def test_vp_sigma_derivative_finite_at_t1(t_val: float) -> None:
    """``VPPath.sigma`` must return finite (s, ds) for every t in [0, 1]."""
    path = get_path("vp")
    t = torch.tensor([t_val], dtype=torch.float64)
    s, ds = path.sigma(t)
    assert torch.isfinite(s).all(), f"sigma(t={t_val}) had non-finite values: {s}"
    assert torch.isfinite(ds).all(), f"d_sigma(t={t_val}) had non-finite values: {ds}"


@pytest.mark.parametrize("t_val", BOUNDARY_TS)
def test_vp_velocity_field_finite_at_boundaries(t_val: float) -> None:
    """End-to-end check: the path velocity must be finite at the boundaries.

    The drift/SDE construction multiplies through ``sigma(t)``'s outputs, so
    a non-finite ``ds`` from the previous test would propagate into training.
    """
    path = get_path("vp")
    t = torch.tensor([t_val], dtype=torch.float64)
    x0 = torch.zeros(1, 4, dtype=torch.float64)
    x1 = torch.ones(1, 4, dtype=torch.float64)
    xt = path.sample_xt(t, x0, x1)
    v = path.velocity(t, x0, x1, xt)
    assert torch.isfinite(v).all(), f"velocity(t={t_val}) had non-finite values: {v}"
