"""Sparsity / detection-rate sweep for LuminaST imputation benchmarks.

Models Xenium-style detection-rate variation by Bernoulli-masking each
transcript count in the observed gene columns with probability `fraction`.
Held-out genes are excluded from the sweep — they remain zero across all
sparsity levels, which is the correct audit behavior (the model must impute
from the genes it's allowed to see, regardless of how sparse those genes are).

Citation frame for the manuscript:
- Bilous, M. et al. "From transcripts to cells: dissecting sensitivity, signal
  contamination, and specificity in Xenium spatial transcriptomics." bioRxiv
  (2025) — establishes that Xenium detection sensitivity varies substantially
  across slides and genes.
- Marco Salas, S. et al. "Optimizing Xenium in situ data utility by quality
  assessment and best-practice analysis workflows." Nat. Methods (2025) — sets
  the QC framing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import anndata as ad
import numpy as np

from .contract import AdapterInput, AdapterResult, BaseAdapter, compute_imputation_metrics
from .panels import MarkerPanel


@dataclass(frozen=True)
class SparsityPoint:
    """One step on the sparsity-sweep grid."""

    fraction: float  # detection rate; 1.0 keeps everything, 0.05 keeps ~5 %
    seed: int = 0


def downsample_detection_rate(
    adata: ad.AnnData,
    fraction: float,
    seed: int = 0,
    exclude_genes: Sequence[str] = (),
) -> ad.AnnData:
    """Bernoulli-mask each transcript with probability (1 - fraction).

    Returns a copy of `adata` with the observed gene columns downsampled.
    `exclude_genes` are passed through unchanged — typically the held-out
    panel, which has already been zeroed by AdapterInput.masked_input().

    Notes:
        - `fraction=1.0` returns an exact copy with no zeroing.
        - `fraction=0.0` zeros every excluded-set-complement entry.
        - Integer-valued matrices get a Bernoulli-per-count semantics: for
          a cell with count k, the survival is Binomial(k, fraction). This
          matches actual detection-failure physics rather than a per-cell
          fixed-drop.
    """
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0, 1]; got {fraction}")

    rng = np.random.default_rng(seed)
    out = adata.copy()

    X = out.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32).copy()

    if fraction == 1.0:
        out.X = X
        return out

    var_names = list(out.var_names)
    if exclude_genes:
        excl_idx = {var_names.index(g) for g in exclude_genes if g in var_names}
    else:
        excl_idx = set()
    keep_cols = np.array([j for j in range(X.shape[1]) if j not in excl_idx], dtype=int)
    if keep_cols.size == 0:
        out.X = X
        return out

    sub = X[:, keep_cols]
    # Bernoulli-thin each integer count: Binomial(k, fraction) per entry.
    # For float matrices we apply a Bernoulli mask to the entry itself.
    rounded = np.round(sub).astype(np.int64)
    is_integer = np.allclose(sub, rounded, atol=1e-6)
    if is_integer:
        # Sample Binomial(k, fraction) per entry. Cheap vectorized version.
        sub = rng.binomial(rounded, fraction).astype(np.float32)
    else:
        keep_mask = rng.random(sub.shape) < fraction
        sub = sub * keep_mask.astype(np.float32)

    X[:, keep_cols] = sub
    out.X = X
    return out


def run_sparsity_sweep(
    adapters: Sequence[BaseAdapter],
    adata: ad.AnnData,
    panel: MarkerPanel,
    fractions: Sequence[float],
    seed: int = 0,
    dataset_name: str = "unknown",
    cancer_type: str | None = None,
) -> list[dict]:
    """For each fraction × adapter, downsample observed genes, run, and score.

    Returns a list of dicts with keys (fraction, adapter, result), each
    `result` being a serializable AdapterResult.to_dict().
    """
    out: list[dict] = []
    for fraction in fractions:
        # Construct the audit-safe input once per fraction; held-out columns
        # are zeroed first, then the *remaining* observed columns are
        # Bernoulli-thinned.
        inp_template = AdapterInput(
            input_h5ad=adata,
            held_out_genes=list(panel.genes),
            seed=seed,
            cancer_type=cancer_type,
            extra={"dataset": dataset_name, "panel": panel.name, "fraction": fraction},
        )
        masked = inp_template.masked_input()
        thinned = downsample_detection_rate(
            masked, fraction=fraction, seed=seed,
            exclude_genes=panel.genes,
        )

        # Run each adapter on the thinned input. We still want the
        # held-out-gene metrics, so we score against the *original* truth.
        truth = inp_template.truth_matrix()
        for adapter in adapters:
            point_inp = AdapterInput(
                input_h5ad=thinned,
                held_out_genes=list(panel.genes),
                seed=seed,
                cancer_type=cancer_type,
                extra={"dataset": dataset_name, "panel": panel.name, "fraction": fraction},
            )
            adapter_result = adapter.run(point_inp)

            # Override the metrics with truth from the *un-thinned* matrix so
            # we measure imputation quality against the real held-out values.
            if adapter_result.status == "ok" and adapter_result.imputed_h5ad is not None:
                adapter_result.metrics_json = compute_imputation_metrics(
                    truth=truth,
                    imputed=adapter_result.imputed_h5ad,
                    held_out_genes=list(panel.genes),
                )

            row = adapter_result.to_dict()
            row["fraction"] = fraction
            row["sparsity_observed_thinned"] = float(np.mean(np.asarray(thinned.X) == 0))
            out.append(row)
    return out


def aggregate_sparsity_results(
    results: Sequence[dict],
    dataset_name: str,
    panel_name: str,
) -> dict:
    """JSON-serializable aggregation of sparsity-sweep rows."""
    return {
        "schema_version": "1",
        "dataset": dataset_name,
        "panel": panel_name,
        "n_rows": len(results),
        "rows": list(results),
    }
