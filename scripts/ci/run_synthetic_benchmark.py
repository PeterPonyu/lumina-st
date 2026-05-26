#!/usr/bin/env python
"""End-to-end smoke benchmark: runs every available adapter on a synthetic
COAD-shaped AnnData against the TME marker panel and writes the result JSON.

Exists so the CI lane can prove the full benchmark contract works without
external dependencies (SpaIM/TISSUE/etc. are gracefully recorded as
unavailable). Promoted to `lumina-st/results/benchmark/synthetic_smoke.json`
for downstream figure scripts.

Usage:
    python scripts/ci/run_synthetic_benchmark.py
    python scripts/ci/run_synthetic_benchmark.py --out results/benchmark/foo.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np

from lumina_st.benchmarks import (
    aggregate_results,
    get_panel,
    run_panel,
    write_results_json,
)
from lumina_st.benchmarks.adapters import (
    CellTAdapter,
    KNNAdapter,
    MeanAdapter,
    SpaIMAdapter,
    STMCDIAdapter,
    TISSUEAdapter,
)


def make_synthetic_coad(n_cells: int = 200, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    base = [f"GENE_{i:03d}" for i in range(60)]
    markers = list(get_panel("tme-immune-stromal").genes)
    all_genes = base + markers
    X = rng.poisson(2.5, size=(n_cells, len(all_genes))).astype(np.float32)
    # Sparsify ~30% like Xenium-style data
    sparsity_mask = rng.random(X.shape) < 0.3
    X[sparsity_mask] = 0.0
    adata = ad.AnnData(X=X)
    adata.var_names = all_genes
    adata.obs["cancer_type"] = ["COAD"] * n_cells
    return adata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="results/benchmark/synthetic_smoke.json",
        help="Output JSON path (relative to lumina-st/ root)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cells", type=int, default=200)
    args = parser.parse_args()

    adata = make_synthetic_coad(n_cells=args.n_cells, seed=args.seed)
    panel = get_panel("tme-immune-stromal")

    def _mean_base(masked, _inp):
        return MeanAdapter()._impute(masked, _inp)

    adapters = [
        MeanAdapter(),
        KNNAdapter(k=10),
        # External competitors — all record unavailable until installed.
        SpaIMAdapter(reference_adata=None),
        TISSUEAdapter(base_imputer=_mean_base, alpha=0.1),
        STMCDIAdapter(),
        CellTAdapter(),
    ]

    print(f"[smoke] Running {len(adapters)} adapters on n_cells={args.n_cells}, "
          f"panel={panel.name} ({len(panel.genes)} genes)")
    results = run_panel(
        adapters, adata, panel, seed=args.seed,
        dataset_name="synthetic-coad-smoke",
        cancer_type="COAD",
    )

    for r in results:
        m = r.metrics_json.get("mean_pearson", float("nan"))
        print(f"  {r.method:12s} status={r.status:40s} mean_pearson={m:.4f} runtime={r.runtime_s:.3f}s")

    aggregated = aggregate_results({("synthetic-coad-smoke", panel.name): results})

    out_path = Path(__file__).resolve().parents[2] / args.out
    write_results_json(aggregated, out_path)
    print(f"[smoke] Wrote {out_path}")

    # Sanity check: at least one adapter must have succeeded.
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        print("[smoke] FAIL: no adapter succeeded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
