"""Per-patch tissue concordance metric.

Round 12 W003 — closes the §2.4 / §2.3 "patch-based
cell-type concordance" content-variety gap. Segments tissue into an
NxN grid, computes per-cell-type proportions in each patch, and
correlates those proportions against ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


__all__ = [
    "segment_into_patches",
    "per_patch_celltype_proportions",
    "patch_concordance",
]


def segment_into_patches(
    coords: np.ndarray,
    n_x: int,
    n_y: int,
    bbox: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Assign each (x, y) coordinate to a 2D patch index.

    Args:
        coords: (N, 2) array of (x, y) cell positions.
        n_x: number of patches along x.
        n_y: number of patches along y.
        bbox: optional (xmin, xmax, ymin, ymax); defaults to data extent.

    Returns:
        (N,) int array of flat patch indices in [0, n_x*n_y - 1];
        out-of-bbox points get -1.
    """
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"coords must be (N, 2), got {c.shape}")
    if n_x < 1 or n_y < 1:
        raise ValueError(f"n_x and n_y must be >= 1, got {n_x}, {n_y}")
    if bbox is None:
        xmin, ymin = c.min(axis=0)
        xmax, ymax = c.max(axis=0)
    else:
        xmin, xmax, ymin, ymax = bbox
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"degenerate bbox: ({xmin}, {xmax}, {ymin}, {ymax})")

    # Bin so cells exactly at xmax/ymax land in the last patch.
    bx = np.minimum(((c[:, 0] - xmin) / (xmax - xmin) * n_x).astype(np.int64), n_x - 1)
    by = np.minimum(((c[:, 1] - ymin) / (ymax - ymin) * n_y).astype(np.int64), n_y - 1)
    out = bx * n_y + by
    out[(c[:, 0] < xmin) | (c[:, 0] > xmax) |
        (c[:, 1] < ymin) | (c[:, 1] > ymax)] = -1
    return out


def per_patch_celltype_proportions(
    patch_ids: np.ndarray,
    labels: Sequence[Any],
    n_patches: int,
) -> tuple[np.ndarray, list[Any]]:
    """For each patch, compute per-cell-type proportions.

    Args:
        patch_ids: (N,) flat patch indices; -1 ignored.
        labels: length-N cell-type labels.
        n_patches: total number of patches (typically n_x * n_y).

    Returns:
        proportions: (n_patches, n_celltypes) array; row sums to 1 or 0
            (zero when no cells fall in that patch).
        celltype_list: sorted list of unique labels matching column order.
    """
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != patch_ids.shape[0]:
        raise ValueError("labels length != patch_ids length")
    celltypes = sorted({lbl for lbl in labels_arr.tolist()})
    ct_to_col = {ct: j for j, ct in enumerate(celltypes)}
    counts = np.zeros((n_patches, len(celltypes)), dtype=np.float64)
    for pid, lbl in zip(patch_ids.tolist(), labels_arr.tolist()):
        if pid < 0:
            continue
        counts[pid, ct_to_col[lbl]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    proportions = counts / safe
    proportions[row_sums.ravel() == 0] = 0.0
    return proportions, celltypes


def patch_concordance(
    truth_coords: np.ndarray,
    truth_labels: Sequence[Any],
    recon_labels: Sequence[Any],
    n_x: int = 10,
    n_y: int = 10,
) -> dict[str, Any]:
    """Per-cell-type Pearson concordance of per-patch proportions.

    Both inputs share the same coordinate grid (typically same cells with
    truth-derived vs reconstruction-derived labels — e.g.
    baseline-imputed vs CODEX-derived annotations).

    Args:
        truth_coords: (N, 2) coordinates.
        truth_labels: length-N labels from ground truth.
        recon_labels: length-N labels from reconstruction.
        n_x, n_y: patch grid dimensions.

    Returns:
        {"n_patches", "per_celltype_pearson": {ct: r}, "mean_pearson"}.
    """
    if len(truth_labels) != len(recon_labels):
        raise ValueError("truth and recon label lengths differ")
    pid = segment_into_patches(truth_coords, n_x, n_y)
    n_p = n_x * n_y
    t_prop, t_cts = per_patch_celltype_proportions(pid, truth_labels, n_p)
    r_prop, r_cts = per_patch_celltype_proportions(pid, recon_labels, n_p)
    # Align columns across the union of cell types.
    all_cts = sorted(set(t_cts) | set(r_cts))
    aligned_t = np.zeros((n_p, len(all_cts)), dtype=np.float64)
    aligned_r = np.zeros((n_p, len(all_cts)), dtype=np.float64)
    t_idx = {ct: j for j, ct in enumerate(t_cts)}
    r_idx = {ct: j for j, ct in enumerate(r_cts)}
    for j, ct in enumerate(all_cts):
        if ct in t_idx:
            aligned_t[:, j] = t_prop[:, t_idx[ct]]
        if ct in r_idx:
            aligned_r[:, j] = r_prop[:, r_idx[ct]]
    per_ct: dict[Any, float] = {}
    for j, ct in enumerate(all_cts):
        a = aligned_t[:, j]
        b = aligned_r[:, j]
        if a.std() < 1e-12 or b.std() < 1e-12:
            per_ct[ct] = float("nan")
        else:
            per_ct[ct] = float(np.corrcoef(a, b)[0, 1])
    finite = [v for v in per_ct.values() if not np.isnan(v)]
    return {
        "n_patches": n_p,
        "per_celltype_pearson": per_ct,
        "mean_pearson": float(np.mean(finite)) if finite else float("nan"),
    }
