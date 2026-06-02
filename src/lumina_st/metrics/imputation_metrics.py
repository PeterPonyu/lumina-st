"""Pure-function imputation-quality metrics.

Round 11 W002 + W003 — closes the SSIM, Jensen-Shannon, and cluster
concordance metric gaps against the corresponding sections of the baseline
paper (Section 2.2 / 2.3 metric blocks).

Every function is a pure (input -> scalar-or-dict) call with deterministic
output on identical input. None of the functions assume AnnData or any
benchmark-runner context; they take numpy arrays / label sequences.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


__all__ = [
    "ssim",
    "jensen_shannon_divergence",
    "cluster_concordance",
]


def _ssim_scalar(a: np.ndarray, b: np.ndarray,
                 c1: float, c2: float, eps: float) -> float:
    """Single SSIM value for one paired 1-D sample (luminance·contrast·structure).

    Operates on the supplied paired vectors with the given C1/C2 stabilizers.
    Returns NaN when the denominator collapses (degenerate constant window).
    """
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(((a - mu_a) ** 2).mean())
    var_b = float(((b - mu_b) ** 2).mean())
    cov_ab = float(((a - mu_a) * (b - mu_b)).mean())
    num = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    if den <= eps:
        return float("nan")
    return num / den


def _grid_window_ids(xy: np.ndarray, n_windows: int) -> np.ndarray:
    """Assign each point to a cell of an ``n_windows`` × ``n_windows`` 2-D grid.

    Bins the two leading spatial coordinates independently into ``n_windows``
    equal-width bins over their observed range; the window id is the flattened
    (bin_x, bin_y) pair. Degenerate (zero-extent) axes collapse to a single bin.
    """
    ids = np.zeros(xy.shape[0], dtype=np.int64)
    for d in range(2):
        col = xy[:, d]
        lo = float(col.min())
        hi = float(col.max())
        if hi <= lo:
            binned = np.zeros(col.shape[0], dtype=np.int64)
        else:
            binned = np.clip(
                ((col - lo) / (hi - lo) * n_windows).astype(np.int64),
                0, n_windows - 1)
        ids = ids * n_windows + binned
    return ids


def ssim(truth: np.ndarray, recon: np.ndarray,
         data_range: float | None = None, eps: float = 1e-12,
         spatial: np.ndarray | None = None, n_windows: int = 8,
         min_window: int = 4) -> float:
    """Structural Similarity Index between two flat-or-2D arrays.

    Two modes:

    * **Global (default, ``spatial=None``)** — the historical 1-D surrogate:
      a single luminance/contrast/structure SSIM over the flattened arrays.
      Captures global distributional similarity only (no spatial sensitivity).
    * **Spatially-windowed (``spatial`` given)** — a 2-D structural-similarity
      index matching the reference (image-style) SSIM definition: points are
      placed at their ``spatial`` coordinates, binned into an ``n_windows`` ×
      ``n_windows`` grid, a local SSIM is computed per window (windows with at
      least ``min_window`` points), and the per-window scores are averaged. The
      shared C1/C2 stabilizers derive from the global ``data_range`` so
      low-variance windows are handled like standard windowed SSIM. This is
      sensitive to local spatial texture, which the global surrogate ignores.

    C1, C2 derive from ``data_range``; when None it defaults to
    ``truth.max() - truth.min()`` (or 1.0 if degenerate).

    Args:
        truth: ground-truth array of any shape; flattened internally.
        recon: reconstructed array; must match ``truth.shape``.
        data_range: dynamic range of the data (e.g. 1.0 for z-normalized).
        eps: small constant for numerical stability.
        spatial: optional ``(N, >=2)`` spatial coordinates (e.g.
            ``obsm['spatial']``). When provided, the windowed 2-D SSIM is
            computed instead of the global surrogate.
        n_windows: number of bins per spatial axis (windowed mode only).
        min_window: minimum points per window for it to contribute (windowed mode).

    Returns:
        Scalar SSIM in [-1, 1] (1.0 = identical). NaN if inputs are
        constant in a way that makes SSIM undefined, or (windowed mode) if
        no window meets ``min_window``.
    """
    a = np.asarray(truth, dtype=np.float64).reshape(-1)
    b = np.asarray(recon, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return float("nan")
    dr = float(data_range) if data_range is not None else float(a.max() - a.min())
    if dr <= 0.0:
        dr = 1.0
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2
    if spatial is None:
        return _ssim_scalar(a, b, c1, c2, eps)
    coords = np.asarray(spatial, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] != a.size:
        raise ValueError(
            f"spatial must be (N, >=2) matching the signal length {a.size}; "
            f"got {coords.shape}")
    window_ids = _grid_window_ids(coords[:, :2], int(n_windows))
    scores: list[float] = []
    for w in np.unique(window_ids):
        sel = window_ids == w
        if int(sel.sum()) < int(min_window):
            continue
        s = _ssim_scalar(a[sel], b[sel], c1, c2, eps)
        if math.isfinite(s):
            scores.append(s)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray,
                              base: float = 2.0, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two non-negative vectors.

    Inputs are normalized to sum-1 (probability) before the
    divergence is computed; zero-mass entries are stabilized with
    `eps`. Result is in [0, log_base(2)] (0 = identical, log_base(2)
    = maximally separated).

    Args:
        p: non-negative vector of length K.
        q: non-negative vector of length K.
        base: log base (2.0 yields JS in [0, 1]).

    Returns:
        Scalar JS divergence; NaN if both inputs sum to zero.
    """
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    b = np.asarray(q, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    sa, sb = float(a.sum()), float(b.sum())
    if sa <= 0 or sb <= 0:
        return float("nan")
    pa = np.clip(a / sa, eps, None)
    qb = np.clip(b / sb, eps, None)
    m = 0.5 * (pa + qb)
    # KL(p || m) + KL(q || m) — both in nats first
    kl_pm = float((pa * (np.log(pa) - np.log(m))).sum())
    kl_qm = float((qb * (np.log(qb) - np.log(m))).sum())
    js_nats = 0.5 * (kl_pm + kl_qm)
    # Convert to chosen base; clip tiny negatives from float noise.
    js = js_nats / math.log(base)
    if js < 0 and js > -1e-12:
        js = 0.0
    return js


def cluster_concordance(truth_labels: Sequence[Any] | np.ndarray,
                        pred_labels: Sequence[Any] | np.ndarray) -> dict[str, float]:
    """Cluster-pair concordance returning {ARI, AMI, Homo, NMI}.

    Matches the baseline paper Section 2.3 clustering metric block. Wraps
    sklearn so the helper is a single import for downstream code.

    Args:
        truth_labels: reference / ground-truth cluster IDs (length N).
        pred_labels: predicted / imputed cluster IDs (length N).

    Returns:
        Dict with keys "ari", "ami", "homo", "nmi", each in [0, 1]
        (ARI can dip slightly negative on misalignment). NaN-tolerant
        on degenerate empty input.
    """
    truth = np.asarray(truth_labels)
    pred = np.asarray(pred_labels)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {pred.shape}")
    if truth.size == 0:
        return {"ari": float("nan"), "ami": float("nan"),
                "homo": float("nan"), "nmi": float("nan")}
    from sklearn.metrics import (
        adjusted_rand_score,
        adjusted_mutual_info_score,
        homogeneity_score,
        normalized_mutual_info_score,
    )
    return {
        "ari": float(adjusted_rand_score(truth, pred)),
        "ami": float(adjusted_mutual_info_score(truth, pred)),
        "homo": float(homogeneity_score(truth, pred)),
        "nmi": float(normalized_mutual_info_score(truth, pred)),
    }
