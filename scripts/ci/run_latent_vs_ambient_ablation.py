#!/usr/bin/env python
"""Latent vs ambient (gene-space) diffusion ablation for LuminaST (issue #138).

LuminaST normally runs its conditional flow inside a compressed VAE latent
space. The ambient alternative lets the flow operate directly on gene/HVG space
via the :class:`IdentityLatentEncoder` seam. This ablation compares the two
modes on the SAME synthetic COAD-shaped AnnData, with an identical seed,
sampling steps, and transformer width+depth — the ONLY thing that varies is
where the flow lives:

* ``latent``  — a tiny VAE (``latent_encoder_backend="tiny_vae"``) compresses the
  gene matrix to ``latent_dim`` (small) and the flow runs in that latent space.
* ``ambient`` — an :class:`IdentityLatentEncoder` makes the latent space *equal*
  to gene space (``latent_dim == n_genes``); the flow runs directly on genes.

For each mode we run ``enhance(held_out_genes=...)`` end-to-end, score held-out
gene Pearson/Spearman/RMSE against the (un-masked) truth, and record train-time
(VAE pretrain, a no-op for ambient) + inference wall-clock plus a simple memory
proxy (trainable parameter count). The model is untrained (tiny, CPU) — the
deliverable is the harness + schema-valid JSON, not the metric values.

Usage:
    python scripts/ci/run_latent_vs_ambient_ablation.py \
        --out results/benchmark/latent_vs_ambient_ablation.json
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
from lumina_st.latents.tiny_vae import TinyVAE
from lumina_st.latents.vae_interface import IdentityLatentEncoder

# The two modes this ablation compares (issue #138).
MODES: tuple[str, ...] = ("latent", "ambient")

# A small held-out marker panel that lives inside the gene space.
HELD_OUT_GENES: tuple[str, ...] = ("CD8A", "EPCAM", "POSTN")


def make_synthetic_coad(
    n_genes: int = 24, n_cells: int = 64, seed: int = 0
) -> ad.AnnData:
    """Synthetic COAD-shaped AnnData in true gene space.

    Both modes consume the SAME gene matrix; the named held-out genes occupy the
    first columns so they can be masked and scored as a held-out-recovery panel.
    """
    rng = np.random.default_rng(seed)
    held = list(HELD_OUT_GENES)
    n_extra = n_genes - len(held)
    if n_extra < 1:
        raise ValueError(f"n_genes={n_genes} too small for {len(held)} held-out genes")
    genes = held + [f"GENE_{i:03d}" for i in range(n_extra)]
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.obs["cancer_type"] = ["COAD"] * n_cells
    return adata


def _tiny_config(latent_dim: int, seed: int) -> LuminaSTConfig:
    """A cheap untrained config: small width/depth, few sampling steps.

    ``latent_dim`` is the flow's working dimension — small for ``latent`` mode,
    equal to ``n_genes`` for ``ambient`` mode. Width (``hidden_size``), depth,
    and ``num_heads`` are held constant across modes so the only difference is
    where the flow operates.
    """
    return LuminaSTConfig(
        latent_dim=latent_dim,
        latent_encoder_backend="tiny_vae",
        hidden_size=16,
        depth=1,
        num_heads=2,
        cancer_types=["COAD"],
        sampling_method="euler",
        num_sampling_steps=8,
        apply_sparsity=False,
        seed=seed,
    )


def _trainable_param_count(module: torch.nn.Module) -> int:
    """Simple memory proxy: number of trainable parameters (no psutil)."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _run_mode(
    mode: str,
    adata: ad.AnnData,
    held_out: list[str],
    *,
    n_genes: int,
    latent_dim: int,
    vae_pretrain_steps: int,
    seed: int,
) -> dict:
    """Run one mode (``latent`` or ``ambient``) end-to-end and return a row.

    Both modes share the identical transformer width/depth and sampling config;
    only the encoder seam differs:

    * ``latent``  : config.latent_dim = ``latent_dim`` (small); the flow's VAE
      is a :class:`TinyVAE` compressing ``n_genes`` -> ``latent_dim``.
    * ``ambient`` : config.latent_dim = ``n_genes``; the encoder is an
      :class:`IdentityLatentEncoder`, so the flow runs directly on gene space.
    """
    truth = np.asarray(adata.X, dtype=np.float32).copy()
    var_names = list(adata.var_names)

    if mode == "latent":
        flow_dim = latent_dim
    elif mode == "ambient":
        flow_dim = n_genes
    else:  # pragma: no cover - guarded by MODES
        raise ValueError(f"unknown mode {mode!r}")

    config = _tiny_config(latent_dim=flow_dim, seed=seed)
    imputer = LuminaImputer.from_config(config)

    # Attach the encoder seam that defines this mode. ``from_config`` leaves
    # ``module.vae`` unset (None), which would force gene-space==latent-space;
    # we set it explicitly so each mode is faithful.
    train_time_s = 0.0
    if mode == "latent":
        torch.manual_seed(seed)
        vae = TinyVAE(input_dim=n_genes, latent_dim=latent_dim, hidden=32)
        # Best-effort tiny VAE pretrain so encode/decode is a real (if rough)
        # compression rather than random projection. Cheap + CPU + few steps.
        x_ref = torch.from_numpy(truth).float()
        if vae_pretrain_steps > 0:
            opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
            t0 = time.perf_counter()
            for _ in range(vae_pretrain_steps):
                loss = vae(x_ref)["loss"]
                opt.zero_grad()
                loss.backward()
                opt.step()
            train_time_s = time.perf_counter() - t0
        imputer.module.vae = vae
    else:
        # Identity over gene space: latent space *is* gene space.
        imputer.module.vae = IdentityLatentEncoder(latent_dim=n_genes)

    param_count = _trainable_param_count(imputer.module)

    t0 = time.perf_counter()
    try:
        enhanced = imputer.enhance(adata, held_out_genes=held_out, seed=seed)
        inference_s = time.perf_counter() - t0
        # ``enhance`` writes the gene-space reconstruction to layers['imputed']
        # whenever the decoded width matches n_vars (true for both modes here).
        imputed = ad.AnnData(X=np.asarray(enhanced.layers["imputed"], dtype=np.float32))
        imputed.var_names = var_names
        metrics = compute_imputation_metrics(
            truth=truth, imputed=imputed, held_out_genes=held_out
        )
        status = "ok"
    except Exception as exc:  # pragma: no cover - surfaced as a row status
        inference_s = time.perf_counter() - t0
        metrics = {}
        status = f"unavailable:{type(exc).__name__}: {exc}"

    return {
        "mode": mode,
        "flow_dim": int(flow_dim),
        "n_genes": int(n_genes),
        "status": status,
        "train_time_s": train_time_s,
        "inference_s": inference_s,
        "runtime_s": train_time_s + inference_s,
        "param_count": param_count,
        "metrics_json": metrics,
    }


def run_ablation(
    *,
    modes: Sequence[str] = MODES,
    n_genes: int = 24,
    latent_dim: int = 8,
    n_cells: int = 64,
    vae_pretrain_steps: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Run the latent vs ambient comparison and return one row per mode.

    Each row carries the mode, its flow working dimension, a ``status`` field,
    wall-clock (train + inference), a param-count memory proxy, and a
    ``metrics_json`` dict from ``compute_imputation_metrics`` (mean_pearson /
    mean_spearman / mean_rmse over the held-out genes).
    """
    adata = make_synthetic_coad(n_genes=n_genes, n_cells=n_cells, seed=seed)
    held_out = [g for g in HELD_OUT_GENES if g in adata.var_names]

    rows: list[dict] = []
    for mode in modes:
        rows.append(
            _run_mode(
                mode,
                adata,
                held_out,
                n_genes=n_genes,
                latent_dim=latent_dim,
                vae_pretrain_steps=vae_pretrain_steps,
                seed=seed,
            )
        )
    return rows


def _aggregate(rows: list[dict], *, n_genes: int, latent_dim: int, seed: int) -> dict:
    """JSON-serializable aggregation mirroring the guidance-sweep schema."""
    return {
        "schema_version": "1",
        "ablation": "latent_vs_ambient",
        "dataset": "synthetic-coad-latent-vs-ambient",
        "held_out_genes": list(HELD_OUT_GENES),
        "n_genes": n_genes,
        "latent_dim": latent_dim,
        "seed": seed,
        "n_rows": len(rows),
        "rows": rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="results/benchmark/latent_vs_ambient_ablation.json"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-genes", type=int, default=24)
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=8,
        help="Working dimension of the flow in 'latent' mode (small).",
    )
    parser.add_argument("--n-cells", type=int, default=64)
    parser.add_argument(
        "--vae-pretrain-steps",
        type=int,
        default=5,
        help="Tiny-VAE pretrain steps for 'latent' mode (0 disables; grid-shrink for tests).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Override the mode grid (default: latent ambient).",
    )
    args = parser.parse_args(argv)

    modes = tuple(args.modes) if args.modes else MODES

    rows = run_ablation(
        modes=modes,
        n_genes=args.n_genes,
        latent_dim=args.latent_dim,
        n_cells=args.n_cells,
        vae_pretrain_steps=args.vae_pretrain_steps,
        seed=args.seed,
    )
    aggregated = _aggregate(
        rows, n_genes=args.n_genes, latent_dim=args.latent_dim, seed=args.seed
    )

    out_arg = Path(args.out)
    out_path = (
        out_arg if out_arg.is_absolute() else Path(__file__).resolve().parents[2] / out_arg
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"[latent_vs_ambient] {len(rows)} modes -> {out_path}")

    manifest = RunManifest.create(
        run_id="latent-vs-ambient-ablation",
        config=_tiny_config(latent_dim=args.latent_dim, seed=args.seed),
        seed=args.seed,
        sweep_params={"mode": list(modes)},
    )
    manifest.to_json(out_path.with_name(out_path.stem + ".manifest.json"))

    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        print("[latent_vs_ambient] FAIL: no mode succeeded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
