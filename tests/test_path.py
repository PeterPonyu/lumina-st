"""Regression test for lumina-st #123.

``InterpolationPath.drift`` used a batch-wide alpha gate:

    drift = da / a * x if a.abs().min() > 1e-8 else torch.zeros_like(x)

``a.abs().min()`` reduces across the entire batch, so a single sample landing
near a path boundary (small ``alpha``) zeroes the drift for *every* sample in
the batch — silently producing wrong outputs for flow-matching steps that
happen to mix near-boundary and interior times.

The fix replaces the batch-wide gate with a per-sample mask (shared template
with aether-3d #136):

    EPS_ALPHA = 1e-6
    mask = (a.abs() > EPS_ALPHA)
    drift = torch.where(mask, da / a * x, torch.zeros_like(x))
"""

from __future__ import annotations

import torch

from lumina_st.flow.path import LinearPath


def test_drift_per_sample_mask() -> None:
    """A near-boundary sample must NOT zero the drift of an interior sample.

    Construct a batch of two samples whose only difference is the time ``t``:
    sample 0 has ``t`` very close to a boundary (small alpha); sample 1 has
    ``t`` deep in the interior (alpha well above eps). With the batch-wide
    gate, both rows of ``drift`` collapse to zero; with the per-sample mask,
    row 0 is zero (gated) while row 1 retains its real value.
    """

    path = LinearPath()

    # LinearPath: alpha(t) = t, so t=0 forces alpha=0 (gated), t=0.5 is interior.
    t = torch.tensor([1e-9, 0.5])
    x = torch.ones(2, 4)

    drift, _ = path.drift(x, t)

    assert drift.shape == x.shape
    # Row 0 is the near-boundary sample — its drift must be zero.
    assert torch.allclose(drift[0], torch.zeros_like(drift[0])), drift
    # Row 1 is the interior sample — its drift must NOT be zero.
    assert not torch.allclose(drift[1], torch.zeros_like(drift[1])), drift
    # Row 1 must be finite (no inf/nan from the division).
    assert torch.isfinite(drift[1]).all(), drift
