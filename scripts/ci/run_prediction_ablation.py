#!/usr/bin/env python
"""Prediction-parameterization ablation for LuminaST (issue #139).

Sweeps ``LuminaSTConfig.prediction ∈ {"velocity", "noise", "score"}`` on a
tiny synthetic COAD-shaped problem with a fixed linear path, fixed tiny
transformer, and a fixed seed.  Emits a benchmark-schema JSON with one row per
parameterization (plus a RunManifest recording the sweep coordinate space).

Because LuminaST is built here without a VAE, the latent space and the gene
space coincide (``n_vars == latent_dim``); the held-out "genes" are therefore a
subset of the latent dimensions, zeroed at input and recovered at output.  This
exercises the full prediction-target routing path on CPU with an untrained tiny
model — the deliverable is the harness + schema-valid JSON, not the metric
values.

An important secondary measurement is **NaN incidence**: the ``score`` and
``noise`` prediction targets can be boundary-fragile near t=0/t=1; the
``n_nonfinite`` column records how many entries in the held-out output are
non-finite so fragility is visible without the run failing (issue #139).

Usage:
    python scripts/ci/run_prediction_ablation.py --out results/benchmark/prediction_ablation.json
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

# Default sweep grid (issue #139).  All three prediction targets are always run.
DEFAULT_PREDICTIONS: tuple[str, ...] = ("velocity", "noise", "score")

# A small held-out marker set living inside the latent/gene space.
HELD_OUT_GENES: tuple[str, ...] = ("CD8A", "EPCAM", "POSTN")


def make_synthetic_coad(
    latent_dim: int = 16, n_cells: int = 64, seed: int = 0
) -> ad.AnnData:
    """Synthetic COAD-shaped AnnData whose gene space equals the latent space.

    With no VAE attached, ``enhance`` requires ``n_vars == latent_dim``; the
    named held-out genes occupy the first columns so they can be masked and
    scored as a held-out-recovery panel.
    """
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


def _tiny_config(latent_dim: int, seed: int, prediction: str = "velocity") -> LuminaSTConfig:
    """A cheap untrained config: small width/depth, few sampling steps, fixed linear path."""
    return LuminaSTConfig(
        latent_dim=latent_dim,
        hidden_size=16,
        depth=1,
        num_heads=2,
        cancer_types=["COAD"],
        path_type="linear",
        prediction=prediction,  # type: ignore[arg-type]
        sampling_method="euler",
        num_sampling_steps=8,
        apply_sparsity=False,
        seed=seed,
    )


def run_ablation(
    *,
    predictions: Sequence[str] = DEFAULT_PREDICTIONS,
    latent_dim: int = 16,
    n_cells: int = 64,
    seed: int = 0,
) -> list[dict]:
    """Run the prediction sweep and return one benchmark row per parameterization.

    Each row is shaped like a ``benchmarks`` result row: the swept param, a
    ``status`` field, a ``metrics_json`` dict from ``compute_imputation_metrics``
    (mean_pearson / mean_spearman / mean_rmse), and a ``n_nonfinite`` count of
    non-finite entries in the held-out imputed output — the NaN-incidence column
    requested by issue #139.
    """
    adata = make_synthetic_coad(latent_dim=latent_dim, n_cells=n_cells, seed=seed)
    held_out = [g for g in HELD_OUT_GENES if g in adata.var_names]
    var_names = list(adata.var_names)
    held_idx = [var_names.index(g) for g in held_out]

    # Build the latent input once: held-out columns zeroed (audit boundary).
    x_full = np.asarray(adata.X, dtype=np.float32)
    truth = x_full.copy()
    x_masked = x_full.copy()
    x_masked[:, held_idx] = 0.0
    z_in = torch.from_numpy(x_masked).float()
    y = torch.zeros(z_in.shape[0], dtype=torch.long)

    rows: list[dict] = []
    for pred in predictions:
        config = _tiny_config(latent_dim=latent_dim, seed=seed, prediction=pred)
        imputer = LuminaImputer.from_config(config)
        t0 = time.perf_counter()
        try:
            z_out = imputer.module.enhance_latent(
                z_in,
                y,
                seed=seed,
            )
            runtime_s = time.perf_counter() - t0
            z_np = z_out.detach().cpu().numpy().astype(np.float32)

            # NaN-incidence: count non-finite entries in the held-out columns only.
            held_out_vals = z_np[:, held_idx]
            n_nonfinite = int(np.sum(~np.isfinite(held_out_vals)))
            nan_incidence = float(n_nonfinite) / max(held_out_vals.size, 1)

            # Replace non-finite values with 0 for metric computation so a
            # boundary-fragile parameterization still produces a row (the
            # n_nonfinite column is the audit signal, not a crash).
            z_np_safe = z_np.copy()
            z_np_safe[~np.isfinite(z_np_safe)] = 0.0

            imputed = ad.AnnData(X=z_np_safe)
            imputed.var_names = var_names
            metrics = compute_imputation_metrics(
                truth=truth, imputed=imputed, held_out_genes=held_out
            )
            status = "ok"
        except Exception as exc:  # pragma: no cover - surfaced as a row status
            runtime_s = time.perf_counter() - t0
            metrics = {}
            n_nonfinite = -1
            nan_incidence = float("nan")
            status = f"error:{type(exc).__name__}: {exc}"
        rows.append(
            {
                "prediction": pred,
                "path_type": "linear",
                "status": status,
                "runtime_s": runtime_s,
                "n_nonfinite": n_nonfinite,
                "nan_incidence": nan_incidence,
                "metrics_json": metrics,
            }
        )
    return rows


def _aggregate(rows: list[dict], seed: int) -> dict:
    """JSON-serializable aggregation mirroring the guidance-sweep schema."""
    return {
        "schema_version": "1",
        "ablation": "prediction_param",
        "dataset": "synthetic-coad-prediction",
        "held_out_genes": list(HELD_OUT_GENES),
        "seed": seed,
        "n_rows": len(rows),
        "rows": rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/benchmark/prediction_ablation.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--n-cells", type=int, default=64)
    parser.add_argument(
        "--predictions",
        nargs="+",
        default=None,
        help="Override the prediction grid (default: velocity noise score).",
    )
    args = parser.parse_args(argv)

    predictions = (
        tuple(args.predictions) if args.predictions else DEFAULT_PREDICTIONS
    )

    rows = run_ablation(
        predictions=predictions,
        latent_dim=args.latent_dim,
        n_cells=args.n_cells,
        seed=args.seed,
    )
    aggregated = _aggregate(rows, seed=args.seed)

    out_arg = Path(args.out)
    out_path = out_arg if out_arg.is_absolute() else Path(__file__).resolve().parents[2] / out_arg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"[prediction] {len(rows)} grid points -> {out_path}")

    manifest = RunManifest.create(
        run_id="prediction-ablation",
        config=_tiny_config(latent_dim=args.latent_dim, seed=args.seed),
        seed=args.seed,
        sweep_params={
            "prediction": list(predictions),
        },
    )
    manifest.to_json(out_path.with_name(out_path.stem + ".manifest.json"))

    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        print("[prediction] FAIL: no grid point succeeded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
