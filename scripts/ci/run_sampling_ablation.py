#!/usr/bin/env python
"""Sampling-time ablation for LuminaST (issue #144).

Sweeps the ODE solver, the step budget, and the forward integration start time
on a tiny synthetic COAD-shaped problem and emits a benchmark-schema JSON with
one row per grid point — each row carrying held-out Pearson/RMSE AND a per-call
``runtime_s`` (``time.perf_counter``) so the quality/latency trade-off across
solvers and step counts is recorded. A RunManifest records the sweep grid.

As in the guidance ablation, this builds LuminaST without a VAE, so the latent
space equals the gene space (``n_vars == latent_dim``) and the held-out genes
are a subset of the latent dimensions. Runs on CPU with an untrained tiny model
in seconds — the deliverable is the harness + schema-valid JSON.

Usage:
    python scripts/ci/run_sampling_ablation.py --out results/benchmark/sampling_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import torch

from lumina_st.benchmarks.contract import compute_imputation_metrics
from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer
from lumina_st.experiment.run_manifest import RunManifest

# Default sweep grid (issue #144).
DEFAULT_SAMPLING_METHODS: tuple[str, ...] = ("euler", "heun", "dopri5")
DEFAULT_NUM_STEPS: tuple[int, ...] = (10, 25, 50, 100)
DEFAULT_T_FORWARDS: tuple[float, ...] = (0.7, 0.8, 0.9, 0.95)

HELD_OUT_GENES: tuple[str, ...] = ("CD8A", "EPCAM", "POSTN")


def make_synthetic_coad(
    latent_dim: int = 16, n_cells: int = 64, seed: int = 0
) -> ad.AnnData:
    """Synthetic COAD-shaped AnnData whose gene space equals the latent space."""
    rng = np.random.default_rng(seed)
    held = list(HELD_OUT_GENES)
    n_extra = latent_dim - len(held)
    if n_extra < 1:
        raise ValueError(f"latent_dim={latent_dim} too small for {len(held)} held-out genes")
    genes = held + [f"GENE_{i:03d}" for i in range(n_extra)]
    X = rng.standard_normal((n_cells, latent_dim)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.obs["cancer_type"] = ["COAD"] * n_cells
    return adata


def _tiny_config(latent_dim: int, seed: int) -> LuminaSTConfig:
    """A cheap untrained config: small width/depth (steps are swept)."""
    return LuminaSTConfig(
        latent_dim=latent_dim,
        hidden_size=16,
        depth=1,
        num_heads=2,
        cancer_types=["COAD"],
        apply_sparsity=False,
        seed=seed,
    )


def run_ablation(
    *,
    sampling_methods: Sequence[str] = DEFAULT_SAMPLING_METHODS,
    num_sampling_steps: Sequence[int] = DEFAULT_NUM_STEPS,
    t_forwards: Sequence[float] = DEFAULT_T_FORWARDS,
    latent_dim: int = 16,
    n_cells: int = 64,
    seed: int = 0,
) -> list[dict]:
    """Run the method × steps × t_forward grid; one benchmark row per point.

    Each row records the swept params, a ``status``, the measured per-call
    ``runtime_s``, and a ``metrics_json`` (held-out Pearson/Spearman/RMSE).
    """
    adata = make_synthetic_coad(latent_dim=latent_dim, n_cells=n_cells, seed=seed)
    held_out = [g for g in HELD_OUT_GENES if g in adata.var_names]
    config = _tiny_config(latent_dim=latent_dim, seed=seed)
    imputer = LuminaImputer.from_config(config)
    var_names = list(adata.var_names)
    held_idx = [var_names.index(g) for g in held_out]

    x_full = np.asarray(adata.X, dtype=np.float32)
    truth = x_full.copy()
    x_masked = x_full.copy()
    x_masked[:, held_idx] = 0.0
    z_in = torch.from_numpy(x_masked).float()
    y = torch.zeros(z_in.shape[0], dtype=torch.long)

    rows: list[dict] = []
    for method in sampling_methods:
        for steps in num_sampling_steps:
            for tf in t_forwards:
                # Drive solver/step choices through the config (validate_assignment).
                imputer.config.sampling_method = method
                imputer.config.num_sampling_steps = int(steps)
                t0 = time.perf_counter()
                try:
                    z_out = imputer.module.enhance_latent(
                        z_in,
                        y,
                        t_forward=float(tf),
                        seed=seed,
                    )
                    runtime_s = time.perf_counter() - t0
                    imputed = ad.AnnData(X=z_out.detach().cpu().numpy().astype(np.float32))
                    imputed.var_names = var_names
                    metrics = compute_imputation_metrics(
                        truth=truth, imputed=imputed, held_out_genes=held_out
                    )
                    status = "ok"
                except Exception as exc:  # pragma: no cover - surfaced as a row status
                    runtime_s = time.perf_counter() - t0
                    metrics = {}
                    status = f"error:{type(exc).__name__}: {exc}"
                rows.append(
                    {
                        "sampling_method": method,
                        "num_sampling_steps": int(steps),
                        "t_forward": float(tf),
                        "status": status,
                        "runtime_s": runtime_s,
                        "metrics_json": metrics,
                    }
                )
    return rows


def _aggregate(rows: list[dict], seed: int) -> dict:
    """JSON-serializable aggregation mirroring the sparsity-sweep schema."""
    return {
        "schema_version": "1",
        "ablation": "sampling_time",
        "dataset": "synthetic-coad-sampling",
        "held_out_genes": list(HELD_OUT_GENES),
        "seed": seed,
        "n_rows": len(rows),
        "rows": rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/benchmark/sampling_ablation.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--n-cells", type=int, default=64)
    parser.add_argument("--sampling-methods", nargs="+", default=None)
    parser.add_argument("--num-steps", type=int, nargs="+", default=None)
    parser.add_argument("--t-forwards", type=float, nargs="+", default=None)
    args = parser.parse_args(argv)

    sampling_methods = (
        tuple(args.sampling_methods) if args.sampling_methods else DEFAULT_SAMPLING_METHODS
    )
    num_steps = tuple(args.num_steps) if args.num_steps else DEFAULT_NUM_STEPS
    t_forwards = tuple(args.t_forwards) if args.t_forwards else DEFAULT_T_FORWARDS

    rows = run_ablation(
        sampling_methods=sampling_methods,
        num_sampling_steps=num_steps,
        t_forwards=t_forwards,
        latent_dim=args.latent_dim,
        n_cells=args.n_cells,
        seed=args.seed,
    )
    aggregated = _aggregate(rows, seed=args.seed)

    out_arg = Path(args.out)
    out_path = out_arg if out_arg.is_absolute() else Path(__file__).resolve().parents[2] / out_arg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"[sampling] {len(rows)} grid points -> {out_path}")

    manifest = RunManifest.create(
        run_id="sampling-ablation",
        config=_tiny_config(latent_dim=args.latent_dim, seed=args.seed),
        seed=args.seed,
        sweep_params={
            "sampling_method": list(sampling_methods),
            "num_sampling_steps": list(num_steps),
            "t_forward": list(t_forwards),
        },
    )
    manifest.to_json(out_path.with_name(out_path.stem + ".manifest.json"))

    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        print("[sampling] FAIL: no grid point succeeded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
