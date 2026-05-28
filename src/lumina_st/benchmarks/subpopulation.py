"""Sub-clustering + per-subtype differential-expression scaffold.

Round 12 W002 — closes the §2.5 content gap (fine-grained dissection of cell
subpopulations) content-variety gap identified in
`docs/CONTENT_VARIETY_AUDIT.md`. Pure-function helpers; no AnnData
dependency in the signatures; downstream code can wrap with AnnData.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


__all__ = [
    "subcluster_within_label",
    "per_subtype_degs",
    "validate_subtype_markers",
]


def subcluster_within_label(
    embedding: np.ndarray,
    labels: Sequence[Any],
    target_label: Any,
    n_subclusters: int = 5,
    method: str = "kmeans",
    random_state: int = 0,
) -> np.ndarray:
    """Sub-cluster cells whose label matches `target_label`.

    Returns a sub-cluster ID array aligned with `labels`. Cells whose
    label is NOT `target_label` receive `-1` so callers can filter them.

    Args:
        embedding: (N, D) latent or expression embedding.
        labels: length-N major-cell-type labels.
        target_label: which major cell type to sub-cluster.
        n_subclusters: number of sub-clusters to find.
        method: "kmeans" (default) or "agglomerative".
        random_state: seed for KMeans / fallback.

    Returns:
        (N,) int array of sub-cluster IDs; non-target cells = -1.
    """
    if embedding.ndim != 2:
        raise ValueError(f"embedding must be 2-D, got shape {embedding.shape}")
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != embedding.shape[0]:
        raise ValueError(
            f"labels length {labels_arr.shape[0]} != embedding rows {embedding.shape[0]}"
        )

    mask = labels_arr == target_label
    out = np.full(labels_arr.shape[0], -1, dtype=np.int64)
    if mask.sum() < n_subclusters:
        # Degenerate: not enough cells to form n_subclusters; assign all to 0.
        out[mask] = 0
        return out

    sub_emb = embedding[mask]
    if method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_subclusters, random_state=random_state, n_init=10)
        sub_ids = km.fit_predict(sub_emb)
    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        ac = AgglomerativeClustering(n_clusters=n_subclusters)
        sub_ids = ac.fit_predict(sub_emb)
    else:
        raise ValueError(f"unsupported method {method!r}")
    out[mask] = sub_ids.astype(np.int64)
    return out


def per_subtype_degs(
    X: np.ndarray,
    var_names: Sequence[str],
    sub_labels: Sequence[Any],
    n_top: int = 20,
    eps: float = 1e-12,
) -> dict[Any, list[dict[str, float]]]:
    """For each sub-label, find genes differentially expressed vs all others.

    Uses Mann-Whitney U (rank-sum) so it's distribution-free and works on
    raw or normalized expression. Returns the top-N positively-enriched
    genes per sub-label with effect-size + raw p-value. The caller
    applies Benjamini-Hochberg via `lumina_st.metrics.statistical`.

    Args:
        X: (N, G) expression matrix.
        var_names: length-G gene names.
        sub_labels: length-N sub-cluster labels (-1 = unassigned, ignored).
        n_top: number of top genes per sub-label.
        eps: small constant to avoid div-by-zero in effect-size.

    Returns:
        dict {sub_label: [{"gene", "auc", "u", "p_value", "log2fc"}, ...]}
        sorted by AUC descending (most enriched first).
    """
    from scipy.stats import mannwhitneyu

    arr = np.asarray(X, dtype=np.float64)
    if arr.shape[1] != len(var_names):
        raise ValueError(
            f"X has {arr.shape[1]} genes but var_names has {len(var_names)}"
        )
    labels_arr = np.asarray(sub_labels)
    if labels_arr.shape[0] != arr.shape[0]:
        raise ValueError("sub_labels length != X rows")

    valid = labels_arr != -1
    arr = arr[valid]
    labels_arr = labels_arr[valid]
    unique_labels = sorted(set(labels_arr.tolist()))

    result: dict[Any, list[dict[str, float]]] = {}
    for ul in unique_labels:
        group_mask = labels_arr == ul
        if group_mask.sum() < 3 or (~group_mask).sum() < 3:
            result[ul] = []
            continue
        in_X = arr[group_mask]
        out_X = arr[~group_mask]
        rows: list[dict[str, Any]] = []
        in_mean = in_X.mean(axis=0)
        out_mean = out_X.mean(axis=0)
        for j in range(arr.shape[1]):
            in_col = in_X[:, j]
            out_col = out_X[:, j]
            if in_col.std() < eps and out_col.std() < eps:
                continue
            try:
                u, p = mannwhitneyu(in_col, out_col, alternative="greater")
            except ValueError:
                continue
            n1, n2 = in_col.size, out_col.size
            auc = float(u) / (n1 * n2)
            log2fc = float(np.log2((in_mean[j] + eps) / (out_mean[j] + eps)))
            rows.append({
                "gene": var_names[j],
                "auc": float(auc),
                "u": float(u),
                "p_value": float(p),
                "log2fc": log2fc,
            })
        rows.sort(key=lambda r: r["auc"], reverse=True)
        result[ul] = rows[:n_top]
    return result


def validate_subtype_markers(
    deg_table: dict[Any, list[dict[str, float]]],
    expected_markers: dict[Any, Sequence[str]],
    top_k: int = 20,
) -> dict[Any, dict[str, Any]]:
    """Check whether canonical markers per subtype appear in the top-N DEGs.

    The reference papers validate sub-clusters by surfacing canonical
    lineage markers (e.g. FOXP3 in Treg, CCR7 in CD4+Tn). This helper
    confirms that the DEG ranking from `per_subtype_degs` carries those
    markers among the top hits.

    Args:
        deg_table: output of `per_subtype_degs`.
        expected_markers: {sub_label: [marker_gene, ...]}.
        top_k: how far down the AUC-sorted list to look.

    Returns:
        dict {sub_label: {"expected": [...], "found_in_top_k": [...],
                          "missing": [...], "hit_rate": 0..1, "top_rank": {gene: rank_or_none}}}
    """
    out: dict[Any, dict[str, Any]] = {}
    for sub_label, expected in expected_markers.items():
        rows = deg_table.get(sub_label, [])
        top_genes = [r["gene"] for r in rows[:top_k]]
        gene_to_rank = {g: i + 1 for i, g in enumerate(top_genes)}
        expected_list = list(expected)
        found = [g for g in expected_list if g in gene_to_rank]
        missing = [g for g in expected_list if g not in gene_to_rank]
        hit_rate = (len(found) / len(expected_list)) if expected_list else 0.0
        out[sub_label] = {
            "expected": expected_list,
            "found_in_top_k": found,
            "missing": missing,
            "hit_rate": float(hit_rate),
            "top_rank": {g: gene_to_rank.get(g) for g in expected_list},
        }
    return out
