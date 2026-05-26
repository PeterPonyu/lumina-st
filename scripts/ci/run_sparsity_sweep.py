#!/usr/bin/env python
"""End-to-end sparsity / detection-rate sweep for LuminaST adapters.

Runs every available adapter on a synthetic COAD-shaped AnnData at five
detection-rate levels {1.0, 0.5, 0.25, 0.1, 0.05}. The result JSON powers the
'Robustness to sparse panels' figure in the LuminaST manuscript and unblocks
the L4 row in the claim ledger.

Usage:
    python scripts/ci/run_sparsity_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np

from lumina_st.benchmarks import (
    aggregate_sparsity_results,
    get_panel,
    run_sparsity_sweep,
)
from lumina_st.benchmarks.adapters import (
    CellTAdapter,
    KNNAdapter,
    MeanAdapter,
    SpaIMAdapter,
    STMCDIAdapter,
    TISSUEAdapter,
)


def make_synthetic_coad(n_cells: int = 300, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    base = [f"GENE_{i:03d}" for i in range(60)]
    markers = list(get_panel("tme-immune-stromal").genes)
    all_genes = base + markers
    # Integer-valued counts so the Binomial-thinning path exercises correctly.
    X = rng.poisson(4.0, size=(n_cells, len(all_genes))).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = all_genes
    adata.obs["cancer_type"] = ["COAD"] * n_cells
    return adata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/benchmark/sparsity_sweep.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cells", type=int, default=300)
    args = parser.parse_args()

    adata = make_synthetic_coad(n_cells=args.n_cells, seed=args.seed)
    panel = get_panel("tme-immune-stromal")
    fractions = [1.0, 0.5, 0.25, 0.1, 0.05]

    def _mean_base(masked, _inp):
        return MeanAdapter()._impute(masked, _inp)

    adapters = [
        MeanAdapter(),
        KNNAdapter(k=10),
        SpaIMAdapter(reference_adata=None),
        TISSUEAdapter(base_imputer=_mean_base, alpha=0.1),
        STMCDIAdapter(),
        CellTAdapter(),
    ]

    print(f"[sparsity] {len(adapters)} adapters × {len(fractions)} fractions "
          f"on n_cells={args.n_cells}, panel={panel.name}")
    rows = run_sparsity_sweep(
        adapters, adata, panel, fractions=fractions, seed=args.seed,
        dataset_name="synthetic-coad-sparsity",
    )

    print(f"{'adapter':12s} {'fraction':>9s} {'observed-sparsity':>18s} "
          f"{'mean_pearson':>14s} {'mean_rmse':>11s} status")
    for r in rows:
        m = r.get("metrics_json", {})
        mp = m.get("mean_pearson", float("nan"))
        rm = m.get("mean_rmse", float("nan"))
        s = r["status"][:40]
        print(f"{r['method']:12s} {r['fraction']:>9.3f} "
              f"{r['sparsity_observed_thinned']:>18.4f} "
              f"{mp:>14.4f} {rm:>11.4f} {s}")

    aggregated = aggregate_sparsity_results(
        rows, "synthetic-coad-sparsity", panel.name
    )
    out_path = Path(__file__).resolve().parents[2] / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"[sparsity] Wrote {out_path}")

    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        print("[sparsity] FAIL: no adapter succeeded at any fraction", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
