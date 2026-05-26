#!/usr/bin/env python
"""Leave-one-context cross-validation harness for LuminaST.

This script ships the harness — running it on synthetic data is a stable CI
smoke test. To validate the pan-cancer / zero-shot generalization claim, the
user supplies real Xenium / SPATCH AnnData per context (COAD, OV, LIHC, BRCA,
NSCLC, PRAD, CESC, ...) and a trainable factory that finetunes LuminaImputer
on the train_contexts dict at every fold.

Stateless baselines (mean, knn) work without modification. The wrapped
external adapters (SpaIM, TISSUE, ...) report unavailable until installed.

Usage:
    python scripts/ci/run_leave_one_context.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np

from lumina_st.benchmarks import (
    aggregate_cv_results,
    get_panel,
    run_cv,
    stateless_factory,
    summarize_cv,
)
from lumina_st.benchmarks.adapters import KNNAdapter, MeanAdapter


CANCER_CONTEXTS = ("COAD", "OV", "LIHC", "BRCA", "NSCLC", "CESC", "PRAD")


def _make_context(name: str, n_cells: int = 120, seed: int = 0) -> ad.AnnData:
    """Synthetic per-cancer-context AnnData."""
    rng = np.random.default_rng(seed)
    base = [f"GENE_{i:03d}" for i in range(40)]
    markers = list(get_panel("tme-immune-stromal").genes)
    all_genes = base + markers
    X = rng.poisson(3.0, size=(n_cells, len(all_genes))).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = all_genes
    adata.obs["cancer_type"] = [name] * n_cells
    return adata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts", nargs="+", default=list(CANCER_CONTEXTS[:4]),
        help="Context names to sweep (synthetic generator used by default)",
    )
    parser.add_argument("--n-cells", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--out", default="results/benchmark/leave_one_context.json")
    args = parser.parse_args()

    contexts: dict[str, ad.AnnData] = {
        name: _make_context(name, n_cells=args.n_cells, seed=args.seed + i)
        for i, name in enumerate(args.contexts)
    }
    panel = get_panel("tme-immune-stromal")

    print(f"[loo-cv] contexts={list(contexts)} panel={panel.name}")

    # Two stateless baselines drive the synthetic smoke. Real runs swap in a
    # trainable factory that finetunes LuminaImputer per fold.
    adapter_factories = {
        "mean": stateless_factory(MeanAdapter),
        "knn": stateless_factory(KNNAdapter, k=10),
    }

    all_aggregated: dict[str, object] = {"schema_version": "1", "per_adapter": {}}
    for adapter_name, factory in adapter_factories.items():
        folds = run_cv(contexts, factory, panel, seed=args.seed)
        summary = summarize_cv(folds, n_boot=args.n_boot, seed=args.seed)
        aggregated = aggregate_cv_results(folds, panel.name, summary=summary)
        all_aggregated["per_adapter"][adapter_name] = aggregated

        print(f"\n[loo-cv] {adapter_name}:")
        for metric_key in ("mean_pearson", "mean_spearman", "mean_rmse"):
            m = summary["per_metric"][metric_key]
            print(f"  {metric_key:14s}: point={m['point']:.4f} "
                  f"95% CI=[{m['lower']:.4f}, {m['upper']:.4f}] "
                  f"per_fold={['{:.3f}'.format(v) for v in m['per_fold']]}")

    out_path = Path(__file__).resolve().parents[2] / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_aggregated, indent=2, default=str))
    print(f"\n[loo-cv] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
