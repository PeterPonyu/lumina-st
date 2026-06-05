#!/usr/bin/env python
"""CFG / guidance-strength ablation for LuminaST (issue #143).

Sweeps the classifier-free-guidance scale and the guidance-decay schedule on a
tiny synthetic COAD-shaped problem and emits a benchmark-schema JSON with one
row per grid point (plus a RunManifest recording the sweep coordinate space).

Because LuminaST is built here without a VAE, the latent space and the gene
space coincide (``n_vars == latent_dim``); the held-out "genes" are therefore a
subset of the latent dimensions, zeroed at input and recovered at output. This
exercises the full guided-sampling path on CPU with an untrained tiny model —
the deliverable is the harness + schema-valid JSON, not the metric values.

Usage:
    python scripts/ci/run_guidance_ablation.py --out results/benchmark/guidance_ablation.json
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

# Default sweep grid (issue #143). guidance_scale=1.0 is the CFG-off baseline.
DEFAULT_GUIDANCE_SCALES: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 7.5)
DEFAULT_CFG_DECAYS: tuple[Optional[str], ...] = (None, "linear", "cosine")

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


def _tiny_config(latent_dim: int, seed: int) -> LuminaSTConfig:
    """A cheap untrained config: small width/depth, few sampling steps."""
    return LuminaSTConfig(
        latent_dim=latent_dim,
        hidden_size=16,
        depth=1,
        num_heads=2,
        cancer_types=["COAD"],
        sampling_method="euler",
        num_sampling_steps=8,
        apply_sparsity=False,
        seed=seed,
    )


def run_ablation(
    *,
    guidance_scales: Sequence[float] = DEFAULT_GUIDANCE_SCALES,
    cfg_decays: Sequence[Optional[str]] = DEFAULT_CFG_DECAYS,
    latent_dim: int = 16,
    n_cells: int = 64,
    seed: int = 0,
) -> list[dict]:
    """Run the guidance × cfg_decay grid and return one benchmark row per point.

    Each row is shaped like a ``benchmarks`` result row: the swept params, a
    ``status`` field, and a ``metrics_json`` dict from
    ``compute_imputation_metrics`` (mean_pearson / mean_spearman / mean_rmse).
    """
    adata = make_synthetic_coad(latent_dim=latent_dim, n_cells=n_cells, seed=seed)
    held_out = [g for g in HELD_OUT_GENES if g in adata.var_names]
    config = _tiny_config(latent_dim=latent_dim, seed=seed)
    imputer = LuminaImputer.from_config(config)
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
    for gs in guidance_scales:
        for decay in cfg_decays:
            t0 = time.perf_counter()
            try:
                z_out = imputer.module.enhance_latent(
                    z_in,
                    y,
                    cfg_scale=float(gs),
                    cfg_decay=decay,
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
                    "guidance_scale": float(gs),
                    "cfg_decay": decay,
                    "cfg_off_baseline": float(gs) == 1.0,
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
        "ablation": "guidance_cfg",
        "dataset": "synthetic-coad-guidance",
        "held_out_genes": list(HELD_OUT_GENES),
        "seed": seed,
        "n_rows": len(rows),
        "rows": rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/benchmark/guidance_ablation.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--n-cells", type=int, default=64)
    parser.add_argument(
        "--guidance-scales",
        type=float,
        nargs="+",
        default=None,
        help="Override the guidance-scale grid (default: 1.0 1.5 2.0 3.0 5.0 7.5).",
    )
    parser.add_argument(
        "--cfg-decays",
        nargs="+",
        default=None,
        help="Override the cfg_decay grid; use 'none' for constant guidance.",
    )
    args = parser.parse_args(argv)

    guidance_scales = (
        tuple(args.guidance_scales) if args.guidance_scales else DEFAULT_GUIDANCE_SCALES
    )
    if args.cfg_decays:
        cfg_decays = tuple(None if d.lower() == "none" else d for d in args.cfg_decays)
    else:
        cfg_decays = DEFAULT_CFG_DECAYS

    rows = run_ablation(
        guidance_scales=guidance_scales,
        cfg_decays=cfg_decays,
        latent_dim=args.latent_dim,
        n_cells=args.n_cells,
        seed=args.seed,
    )
    aggregated = _aggregate(rows, seed=args.seed)

    out_arg = Path(args.out)
    out_path = out_arg if out_arg.is_absolute() else Path(__file__).resolve().parents[2] / out_arg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"[guidance] {len(rows)} grid points -> {out_path}")

    manifest = RunManifest.create(
        run_id="guidance-ablation",
        config=_tiny_config(latent_dim=args.latent_dim, seed=args.seed),
        seed=args.seed,
        sweep_params={
            "guidance_scale": list(guidance_scales),
            "cfg_decay": list(cfg_decays),
        },
    )
    manifest.to_json(out_path.with_name(out_path.stem + ".manifest.json"))

    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        print("[guidance] FAIL: no grid point succeeded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
