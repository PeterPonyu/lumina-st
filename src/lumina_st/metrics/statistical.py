"""Statistical apparatus: permutation null + multiple-testing correction.

Round 12 W006 — covers the statistical machinery referenced (but never
explicitly named) by stPainter §2.5 DEG analysis. Provides:

- `permutation_null`: generic permutation-null builder for any
  stat(x, y) function on two groups.
- `empirical_pvalue`: two-sided empirical p-value from a null distribution.
- `benjamini_hochberg`: BH FDR correction returning q-values + reject mask.

These are pure functions with deterministic outputs given the same RNG
seed. Used by `lumina_st.benchmarks.subpopulation.per_subtype_degs`
downstream, and re-exported via the aether-3d side when needed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


__all__ = [
    "permutation_null",
    "empirical_pvalue",
    "benjamini_hochberg",
]


def permutation_null(
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 1000,
    random_state: int = 0,
) -> np.ndarray:
    """Build a permutation-null distribution for `stat_fn(x, y)`.

    Concatenates `x` and `y`, shuffles labels `n_perm` times, and at each
    iteration recomputes `stat_fn` on the relabeled split. Returns the
    array of `n_perm` null statistics.

    Args:
        stat_fn: callable taking two same-length arrays and returning a scalar.
        x: (n1,) observed group A.
        y: (n2,) observed group B.
        n_perm: number of permutations.
        random_state: seed for reproducibility.

    Returns:
        (n_perm,) array of null statistics.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    n1 = a.size
    if n1 == 0 or b.size == 0:
        return np.full(n_perm, np.nan)
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(random_state)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        rng.shuffle(pool)
        null[i] = float(stat_fn(pool[:n1], pool[n1:]))
    return null


def empirical_pvalue(observed: float, null: np.ndarray,
                     alternative: str = "two-sided") -> float:
    """Two-sided / one-sided empirical p-value from a null distribution.

    Args:
        observed: observed statistic.
        null: (M,) null distribution values.
        alternative: "two-sided" (default), "greater", or "less".

    Returns:
        Scalar p-value in (0, 1]; uses the (k+1) / (M+1) convention so
        the minimum reportable p-value is 1 / (M + 1).
    """
    null_arr = np.asarray(null, dtype=np.float64)
    null_arr = null_arr[~np.isnan(null_arr)]
    if null_arr.size == 0:
        return float("nan")
    m = null_arr.size
    if alternative == "greater":
        k = int((null_arr >= observed).sum())
    elif alternative == "less":
        k = int((null_arr <= observed).sum())
    elif alternative == "two-sided":
        center = float(np.median(null_arr))
        k = int((np.abs(null_arr - center) >= abs(observed - center)).sum())
    else:
        raise ValueError(f"unsupported alternative {alternative!r}")
    return (k + 1) / (m + 1)


def benjamini_hochberg(p_values: np.ndarray,
                       alpha: float = 0.05) -> dict[str, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Returns BH q-values aligned with the input order plus a reject mask
    at level `alpha`.

    Args:
        p_values: (M,) array of raw p-values; NaNs propagate as NaN q.
        alpha: FDR target.

    Returns:
        {"q_values": (M,) array,
         "reject": (M,) bool array,
         "n_rejected": int}
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = p.size
    if m == 0:
        return {"q_values": np.array([]), "reject": np.array([], dtype=bool),
                "n_rejected": 0}
    valid_mask = ~np.isnan(p)
    n_valid = int(valid_mask.sum())
    q = np.full(m, np.nan)
    if n_valid == 0:
        return {"q_values": q, "reject": np.zeros(m, dtype=bool), "n_rejected": 0}
    p_valid = p[valid_mask]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    # BH adjusted: q[i] = min over k >= rank(i) of (n_valid / k) * p_sorted[k]
    factors = n_valid / np.arange(1, n_valid + 1)
    adj = ranked * factors
    # Enforce monotonicity from the largest p downward.
    cummin = np.minimum.accumulate(adj[::-1])[::-1]
    cummin = np.minimum(cummin, 1.0)
    q_sorted = np.empty(n_valid, dtype=np.float64)
    q_sorted[order] = cummin
    q[valid_mask] = q_sorted
    reject = np.zeros(m, dtype=bool)
    reject[valid_mask] = q_sorted <= alpha
    return {"q_values": q, "reject": reject,
            "n_rejected": int(reject.sum())}
