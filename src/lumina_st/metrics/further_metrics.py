"""Reference-grounded "further" metrics for the LuminaST enhancement run.

Wires the previously-coded-but-unused functions in
``lumina_st.metrics.imputation_metrics`` (``ssim``, ``jensen_shannon_divergence``,
``cluster_concordance``) onto the REAL ``st_COAD_test`` ground truth that is
baked into the enhanced AnnData (``obs['annotation']`` = 16 cell types,
``obs['spatial_cluster']`` = 5 domains, ``var['impute_mask']`` = 3,053 observed
genes), and adds the field-standard spatial-autocorrelation metrics
(Moran's I + SVG-rank preservation) that the spatial baseline reports.

Design mirrors ``imputation_metrics``: every function is a deterministic
numpy/label call with no hidden global state. Clustering helpers take an
explicit ``seed`` and a subsample cap so the run stays bounded.

HONESTY: all held-out gene recovery is scored ONLY over the
``impute_mask``-observed genes — i.e. genes that carry real measured ground
truth. Never-observed genes are excluded, so PCC/SSIM/JSD are recovered-vs-real
comparisons, not recovered-vs-fabricated.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .imputation_metrics import jensen_shannon_divergence, ssim, ssim_2d_windowed


__all__ = [
    "spearman_rank",
    "morans_i",
    "morans_i_per_gene",
    "pergene_recovery_stats",
    "svg_rank_preservation",
    "leiden_labels",
    "clustering_agreement",
    "silhouette_of_latent",
]


# --------------------------------------------------------------------------- #
# correlation helpers
# --------------------------------------------------------------------------- #
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_rank(a: Sequence[float] | np.ndarray,
                  b: Sequence[float] | np.ndarray) -> float:
    """Spearman rank correlation between two 1-D vectors (NaN on degenerate)."""
    from scipy.stats import spearmanr

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return float("nan")
    rho = spearmanr(a, b).correlation
    return float(rho) if rho is not None and math.isfinite(rho) else float("nan")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --------------------------------------------------------------------------- #
# spatial autocorrelation (Moran's I over a kNN graph)
# --------------------------------------------------------------------------- #
def _knn_weights(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric binary kNN adjacency (row-index, col-index) over ``coords``.

    Returns the (row, col) index arrays of the undirected kNN graph (each
    edge appears once per direction it was created). Self-edges excluded.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    k = int(min(k, max(1, n - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)  # drop self (column 0)
    return rows, cols


def morans_i(values: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    """Moran's I of a 1-D signal over a precomputed binary kNN graph.

    ``rows``/``cols`` are the edge index arrays from :func:`_knn_weights`; each
    edge contributes weight 1. Returns NaN if the signal has zero variance.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    xm = x - x.mean()
    denom = float((xm ** 2).sum())
    if denom <= 0.0:
        return float("nan")
    w_sum = float(rows.shape[0])
    if w_sum <= 0.0:
        return float("nan")
    num = float((xm[rows] * xm[cols]).sum())
    return (n / w_sum) * (num / denom)


def morans_i_per_gene(matrix: np.ndarray, coords: np.ndarray,
                      k: int = 15) -> np.ndarray:
    """Per-gene Moran's I for an (n_cells, n_genes) matrix over a kNN graph."""
    matrix = np.asarray(matrix, dtype=np.float64)
    rows, cols = _knn_weights(coords, k)
    out = np.empty(matrix.shape[1], dtype=np.float64)
    for g in range(matrix.shape[1]):
        out[g] = morans_i(matrix[:, g], rows, cols)
    return out


# --------------------------------------------------------------------------- #
# Per-gene recovery statistics (spatial-baseline Section 2.2 metric block pattern)
# --------------------------------------------------------------------------- #
def _scale_max(col: np.ndarray) -> np.ndarray:
    """SSIM/JS preprocessing: divide by max (0-max guarded)."""
    col = np.asarray(col, dtype=np.float64)
    m = float(col.max()) if col.size else 0.0
    return col / m if m > 0 else col


def _per_gene_scores(observed: np.ndarray, recovered: np.ndarray,
                     coords: np.ndarray | None = None) -> dict[str, float]:
    """PCC / Spearman / RMSE / SSIM / JSD for ONE gene column.

    SSIM provenance:

    * ``ssim_2d_windowed`` (**primary**) — the true 2-D windowed SSIM over the
      spatial layout (scikit-image backend). Emitted only when ``coords`` (the
      per-spot ``obsm['spatial']`` array) is supplied; NaN otherwise.
    * ``ssim_reference_reported`` — the historical 1-D global surrogate, kept
      available but **demoted** (not the claimed SSIM). When ``coords`` are
      supplied it carries the old 2-D-windowed-on-points value; without coords
      it is the flat 1-D surrogate.
    """
    o = np.asarray(observed, dtype=np.float64).reshape(-1)
    r = np.asarray(recovered, dtype=np.float64).reshape(-1)
    o_pos = np.clip(o, 0.0, None)
    r_pos = np.clip(r, 0.0, None)
    # SSIM/JS on max-scaled non-negative profiles (spatial-baseline convention).
    os_, rs_ = _scale_max(o), _scale_max(r)
    return {
        "pcc": _pearson(o, r),
        "spearman": spearman_rank(o, r),
        "rmse": _rmse(o, r),
        # PRIMARY: true 2-D windowed SSIM over the spatial grid (skimage).
        "ssim_2d_windowed": (
            ssim_2d_windowed(os_, rs_, spatial=coords)
            if coords is not None else float("nan")
        ),
        # DEMOTED surrogate (reference_reported): historical windowed-on-points /
        # 1-D global value — kept for provenance, NOT the claimed SSIM.
        "ssim_reference_reported": ssim(os_, rs_, spatial=coords),
        "jsd": jensen_shannon_divergence(o_pos, r_pos, base=2.0),
    }


def pergene_recovery_stats(
    observed: np.ndarray,
    recovered: np.ndarray,
    gene_var: np.ndarray,
    n_partitions: int = 5,
    hvg_k: Sequence[int] = (10, 20, 50, 100),
    seed: int = 0,
    coords: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-gene recovery statistics over the ``impute_mask``-observed genes.

    Scores the already-computed ``recovered`` layer against the real ``observed``
    expression for every impute_mask-observed gene (a single pass; no re-imputation
    is performed). Per-gene PCC/Spearman/RMSE/SSIM(2-D windowed)/JSD are averaged
    over the top-K HVG (ranked by ``gene_var`` variance) for each K in ``hvg_k``.
    The claimed SSIM is ``ssim_2d_windowed`` (true 2-D windowed, skimage); the
    1-D surrogate is retained as ``ssim_reference_reported`` (demoted).

    Spread (CI) is estimated from gene-partition variance: genes are randomly
    partitioned into ``n_partitions`` groups and the spread of per-partition means
    gives a 95% normal CI. This is a **gene-partition spread**, not a
    cross-validation CI — the method label uses ``gene_partition_ci`` to make
    that explicit.

    Degenerate flag: when fewer than 50% of the top-K genes yield a finite value
    for a metric (e.g. because the recovered signal is constant for most HVGs),
    the mean for that (metric, K) combination is set to ``None`` and a sibling
    ``degenerate`` flag is set to ``True``.

    Args:
        observed: (n_cells, n_observed_genes) real measured expression.
        recovered: (n_cells, n_observed_genes) model-recovered expression
            (the ``imputed`` layer subset to the impute_mask-observed genes).
        gene_var: (n_observed_genes,) per-gene variance for HVG ranking.
        n_partitions: number of gene partitions for spread estimation.
        hvg_k: top-K HVG cut-offs to average over.
        seed: RNG seed for the partition assignment.
        coords: optional (n_cells, >=2) ``obsm['spatial']`` coordinates. When
            supplied, the primary ``ssim_2d_windowed`` metric is the true 2-D
            windowed SSIM over the spatial grid (skimage, ``ssim_mode='spatial'``);
            without coords it is NaN and only the demoted 1-D surrogate
            (``ssim_reference_reported``) is meaningful (``ssim_mode='surrogate'``).

    Returns:
        Nested dict with keys ``schema_version``, ``ssim_mode``, ``n_partitions``,
        ``n_genes``, ``method_note`` (honesty label), and ``per_metric`` ->
        per-K -> {mean, gene_partition_ci_low, gene_partition_ci_high,
        n_finite_genes, degenerate, per_partition_means}.
    """
    observed = np.asarray(observed, dtype=np.float64)
    recovered = np.asarray(recovered, dtype=np.float64)
    gene_var = np.asarray(gene_var, dtype=np.float64)
    n_genes = observed.shape[1]

    coords_arr = np.asarray(coords, dtype=np.float64) if coords is not None else None

    metric_names = ("pcc", "spearman", "rmse",
                    "ssim_2d_windowed", "ssim_reference_reported", "jsd")
    per_gene: dict[str, np.ndarray] = {m: np.full(n_genes, np.nan) for m in metric_names}
    for g in range(n_genes):
        sc = _per_gene_scores(observed[:, g], recovered[:, g], coords=coords_arr)
        for m in metric_names:
            per_gene[m][g] = sc[m]

    # HVG ranking (descending variance) over the observed genes.
    hvg_order = np.argsort(gene_var)[::-1]

    # Gene-partition assignment (seeded permutation).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_genes)
    partition_of = np.empty(n_genes, dtype=np.int64)
    partition_of[perm] = np.arange(n_genes) % int(n_partitions)

    out: dict[str, Any] = {
        "schema_version": "1",
        # "spatial" preserved for backward-compat with the metrics-flattening
        # consumer; it now denotes the true 2-D windowed SSIM (skimage) primary.
        "ssim_mode": "spatial" if coords_arr is not None else "surrogate",
        "n_partitions": int(n_partitions),
        "n_genes": int(n_genes),
        "method_note": (
            "Single imputation pass scored per-gene against real observed expression "
            "(impute_mask-observed genes only). No re-imputation per fold. "
            "CI is gene-partition spread (n_partitions groups), not cross-validation CI."
        ),
        "per_metric": {},
    }
    z95 = 1.959963984540054
    for m in metric_names:
        out["per_metric"][m] = {}
        for k in hvg_k:
            topk = hvg_order[: min(int(k), n_genes)]
            allvals = per_gene[m][topk]
            finite_vals = allvals[np.isfinite(allvals)]
            n_finite = int(finite_vals.size)
            degenerate = n_finite < 0.5 * int(k)

            # Per-partition means (used for spread only when not degenerate).
            partition_means: list[float] = []
            for p in range(int(n_partitions)):
                sel = topk[partition_of[topk] == p]
                vals = per_gene[m][sel]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    partition_means.append(float(vals.mean()))

            if degenerate or n_finite == 0:
                out["per_metric"][m][f"k{k}"] = {
                    "mean": None,
                    "gene_partition_ci_low": None,
                    "gene_partition_ci_high": None,
                    "n_finite_genes": n_finite,
                    "degenerate": True,
                    "per_partition_means": partition_means,
                }
                continue

            mean = float(finite_vals.mean())
            if len(partition_means) >= 2:
                sd = float(np.std(partition_means, ddof=1))
                se = sd / math.sqrt(len(partition_means))
                ci_low = mean - z95 * se
                ci_high = mean + z95 * se
            else:
                ci_low = ci_high = mean
            out["per_metric"][m][f"k{k}"] = {
                "mean": mean,
                "gene_partition_ci_low": float(ci_low),
                "gene_partition_ci_high": float(ci_high),
                "n_finite_genes": n_finite,
                "degenerate": False,
                "per_partition_means": partition_means,
            }
    return out


# --------------------------------------------------------------------------- #
# SVG-rank preservation (Spearman of Moran's-I ranks observed vs recovered)
# --------------------------------------------------------------------------- #
def svg_rank_preservation(
    observed: np.ndarray,
    recovered: np.ndarray,
    coords: np.ndarray,
    k: int = 15,
) -> dict[str, Any]:
    """Spearman of per-gene Moran's-I ranks: observed vs recovered held-out genes.

    Quantifies whether the recovered expression preserves the spatial-variable-
    gene ordering of the real data (spatial-baseline SVG-rank claim).
    """
    mi_obs = morans_i_per_gene(observed, coords, k=k)
    mi_rec = morans_i_per_gene(recovered, coords, k=k)
    mask = np.isfinite(mi_obs) & np.isfinite(mi_rec)
    rho = spearman_rank(mi_obs[mask], mi_rec[mask]) if mask.sum() >= 2 else float("nan")
    return {
        "svg_rank_spearman": rho,
        "n_genes_scored": int(mask.sum()),
        "morans_i_observed_mean": float(np.nanmean(mi_obs)) if mask.any() else float("nan"),
        "morans_i_recovered_mean": float(np.nanmean(mi_rec)) if mask.any() else float("nan"),
    }


# --------------------------------------------------------------------------- #
# clustering agreement vs REAL labels + enhancement uplift
# --------------------------------------------------------------------------- #
def leiden_labels(latent: np.ndarray, seed: int = 0,
                  n_neighbors: int = 15, resolution: float = 1.0) -> np.ndarray:
    """Seeded Leiden clustering of a latent matrix (returns integer labels)."""
    import anndata as ad
    import scanpy as sc

    latent = np.asarray(latent, dtype=np.float32)
    tmp = ad.AnnData(X=np.zeros((latent.shape[0], 1), dtype=np.float32))
    tmp.obsm["_latent"] = latent
    sc.pp.neighbors(tmp, use_rep="_latent", n_neighbors=n_neighbors, random_state=seed)
    sc.tl.leiden(tmp, key_added="_leiden", resolution=resolution,
                 random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    return tmp.obs["_leiden"].astype(int).to_numpy()


def _labelled_mask(truth: np.ndarray) -> np.ndarray:
    """Boolean mask of cells carrying a usable ground-truth label.

    Real ST annotations (e.g. ``spatial_cluster``) leave some cells unlabelled,
    surfacing as ``NaN`` (float arrays) or the strings ``"nan"`` / ``""`` /
    ``"unknown"`` (categorical/object arrays). Those cells carry no ground truth
    and must be dropped before any GT-backed clustering score — keeping them
    would either crash sklearn (NaN) or fabricate a spurious "missing" class.
    """
    truth = np.asarray(truth)
    if truth.dtype.kind in "fc":
        return np.isfinite(truth)
    str_truth = truth.astype(str)
    bad = {"nan", "none", "", "unknown", "na"}
    return np.array([s.strip().lower() not in bad for s in str_truth], dtype=bool)


def clustering_agreement(
    latent: np.ndarray,
    truth_labels: Sequence[Any] | np.ndarray,
    seed: int = 0,
    n_neighbors: int = 15,
    resolution: float = 1.0,
) -> dict[str, float]:
    """Leiden(latent) vs real ``truth_labels`` -> {ari, ami, nmi, homo}.

    Wraps :func:`imputation_metrics.cluster_concordance` so the GT-backed
    clustering block matches the spatial-baseline Section 2.3 metric set.

    Cells whose ground-truth label is missing (NaN / "nan" / "unknown") are
    dropped from the SCORING pair only — the Leiden clustering still runs on the
    full latent (unsupervised) so the predicted partition is unbiased; we then
    compare on the labelled subset. ``n_scored`` records how many cells survived.
    """
    from .imputation_metrics import cluster_concordance

    pred = leiden_labels(latent, seed=seed, n_neighbors=n_neighbors, resolution=resolution)
    truth = np.asarray(truth_labels)
    keep = _labelled_mask(truth)
    truth_s = truth[keep]
    pred_s = pred[keep]
    scores = cluster_concordance(truth_s, pred_s)
    scores["n_pred_clusters"] = float(len(set(pred.tolist())))
    scores["n_scored"] = float(int(keep.sum()))
    return scores


def silhouette_of_latent(latent: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette of ``latent`` against ``labels`` (NaN on single cluster).

    Cells with a missing ground-truth label are dropped before scoring (real
    annotations leave some cells unlabelled).
    """
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    latent = np.asarray(latent, dtype=np.float64)
    keep = _labelled_mask(labels)
    labels = labels[keep]
    latent = latent[keep]
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(latent, labels))
