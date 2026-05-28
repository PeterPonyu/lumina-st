"""Pure-function imputation-quality metrics.

Round 11 W002 + W003 — closes the SSIM, Jensen-Shannon, and cluster
concordance gaps identified by docs/SCAFFOLD_READINESS_AUDIT.md against
the corresponding sections of the baseline paper (Section 2.2 / 2.3 metric blocks).

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


def ssim(truth: np.ndarray, recon: np.ndarray,
         data_range: float | None = None, eps: float = 1e-12) -> float:
    """Structural Similarity Index between two flat-or-2D arrays.

    Uses the standard SSIM formulation with C1, C2 derived from
    `data_range`. When `data_range` is None, it defaults to
    `truth.max() - truth.min()` (or 1.0 if degenerate).

    Args:
        truth: ground-truth array of any shape; flattened internally.
        recon: reconstructed array; must match `truth.shape`.
        data_range: dynamic range of the data (e.g. 1.0 for z-normalized).
        eps: small constant for numerical stability.

    Returns:
        Scalar SSIM in [-1, 1] (1.0 = identical). NaN if inputs are
        constant in a way that makes SSIM undefined.
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
