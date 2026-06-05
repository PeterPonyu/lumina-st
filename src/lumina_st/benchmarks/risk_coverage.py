"""Selective-prediction risk-coverage analysis for ST imputation (#303).

Reframes the uncertainty story from "is empirical coverage close to nominal?"
(#291) to "if a downstream user keeps only the most-confident predictions, how
fast does error drop?". This is the standard selective-prediction setup
(El-Yaniv & Wiener; Geifman & El-Yaniv): rank predictions by an uncertainty
score (e.g. conformal interval width), then sweep a *coverage* fraction
c ∈ (0, 1] and report the *risk* (error) over the c most-confident predictions.

A useful uncertainty score yields a monotonically rising curve: discarding the
least-confident predictions should reduce risk. The scalar summary is the area
under the risk-coverage curve (AURC) — lower is better. A random / uninformative
uncertainty score gives a roughly flat curve at the full-set risk.

Pure NumPy, no AnnData dependency, deterministic for a fixed input.
"""

from __future__ import annotations

import numpy as np

_RISK_KINDS = ("rmse", "mae", "one_minus_pearson")


def _risk(truth: np.ndarray, pred: np.ndarray, kind: str) -> float:
    """Scalar risk over a 1-D pair of truth / pred values."""
    if truth.size == 0:
        return float("nan")
    if kind == "rmse":
        return float(np.sqrt(np.mean((truth - pred) ** 2)))
    if kind == "mae":
        return float(np.mean(np.abs(truth - pred)))
    if kind == "one_minus_pearson":
        if truth.size < 2 or np.std(truth) == 0 or np.std(pred) == 0:
            return float("nan")
        r = float(np.corrcoef(truth, pred)[0, 1])
        return 1.0 - r
    raise ValueError(f"risk must be one of {_RISK_KINDS}; got {kind!r}")


def risk_coverage_curve(
    truth: np.ndarray,
    pred: np.ndarray,
    uncertainty: np.ndarray,
    *,
    risk: str = "rmse",
    n_points: int = 20,
) -> dict[str, object]:
    """Selective-prediction risk-coverage curve and its area (AURC).

    Predictions are sorted ascending by ``uncertainty`` (most confident first);
    for each coverage fraction c the risk is computed over the most-confident
    ``ceil(c * N)`` predictions. AURC is the trapezoidal area under the
    resulting curve, integrated over the coverage axis.

    Args:
        truth: 1-D array of ground-truth values (flattened if higher-dim).
        pred: 1-D array of point predictions, same length as ``truth``.
        uncertainty: 1-D array of per-prediction uncertainty scores
            (higher ⇒ less confident, e.g. conformal interval width).
        risk: one of ``"rmse"``, ``"mae"``, ``"one_minus_pearson"``.
        n_points: number of coverage grid points in (0, 1].

    Returns:
        dict with:
          - ``coverage``: 1-D float array of coverage fractions in (0, 1].
          - ``risk``: 1-D float array of risk at each coverage (same length).
          - ``aurc``: float area under the risk-coverage curve.
          - ``risk_kind``: the risk name used.
          - ``n``: number of predictions scored.
    """
    truth = np.asarray(truth, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    uncertainty = np.asarray(uncertainty, dtype=np.float64).ravel()
    if not (truth.shape == pred.shape == uncertainty.shape):
        raise ValueError(
            "truth, pred, uncertainty must share shape; got "
            f"{truth.shape}, {pred.shape}, {uncertainty.shape}"
        )
    if risk not in _RISK_KINDS:
        raise ValueError(f"risk must be one of {_RISK_KINDS}; got {risk!r}")
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1; got {n_points}")

    n = truth.size
    if n == 0:
        return {
            "coverage": np.zeros(0, dtype=np.float64),
            "risk": np.zeros(0, dtype=np.float64),
            "aurc": float("nan"),
            "risk_kind": risk,
            "n": 0,
        }

    # Sort most-confident (lowest uncertainty) first. mergesort = stable, so the
    # curve is deterministic for ties / fixed input.
    order = np.argsort(uncertainty, kind="mergesort")
    truth_s = truth[order]
    pred_s = pred[order]

    coverages = np.linspace(1.0 / n_points, 1.0, n_points)
    risks = np.empty(n_points, dtype=np.float64)
    for i, c in enumerate(coverages):
        k = max(1, int(np.ceil(c * n)))
        risks[i] = _risk(truth_s[:k], pred_s[:k], risk)

    # Trapezoidal area under risk vs coverage. NaN risks (e.g. degenerate
    # Pearson on a tiny prefix) are dropped so AURC stays finite.
    valid = ~np.isnan(risks)
    if valid.sum() >= 2:
        aurc = float(np.trapz(risks[valid], coverages[valid]))
    elif valid.sum() == 1:
        aurc = float(risks[valid][0])
    else:
        aurc = float("nan")

    return {
        "coverage": coverages,
        "risk": risks,
        "aurc": aurc,
        "risk_kind": risk,
        "n": int(n),
    }
