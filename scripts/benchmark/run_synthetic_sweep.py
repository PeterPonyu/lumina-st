#!/usr/bin/env python3
"""
LuminaST synthetic configuration sweep.

Trains LuminaST end-to-end on the same synthetic reference + ST data across
several model configurations, then captures quality metrics, wall time, peak
GPU memory, and parameter count for every refined version. Outputs:

  results/benchmark/lumina_sweep_<TS>.json      (raw per-config records)
  results/benchmark/lumina_sweep_latest.json    (symlink-style copy of latest)
  results/benchmark/curves/<config>.json        (per-epoch loss curves)
  results/benchmark/enhanced/<config>.h5ad      (enhanced AnnData per config)

Run in the dl conda env (RTX 5090 sm_120 requires CUDA 13 wheels):

  conda run --no-capture-output -n dl python scripts/benchmark/run_synthetic_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.data_flow.generate_synthetic_st import generate_synthetic_reference_and_st

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer
from lumina_st.data.cancer_registry import CancerRegistry
from lumina_st.data.datasets import ReferenceAtlasDataset
from lumina_st.latents.tiny_vae import TinyVAE
from lumina_st.metrics.enhancement_evaluator import EnhancementEvaluator
from lumina_st.models.lumina_transformer import LuminaTransformer
from lumina_st.modules.lumina_flow_module import LuminaFlowModule


@dataclass
class SweepConfig:
    name: str
    latent_dim: int
    hidden_size: int
    depth: int
    num_heads: int
    max_epochs: int
    batch_size: int = 64

    def as_lumina(self, cancer_names: List[str]) -> LuminaSTConfig:
        return LuminaSTConfig(
            latent_dim=self.latent_dim,
            hidden_size=self.hidden_size,
            depth=self.depth,
            num_heads=self.num_heads,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            cancer_types=cancer_names[:1],
            vae_batch_key="cancer_type",  # synthetic fixture stores label here; #106
        )


DEFAULT_SWEEP: List[SweepConfig] = [
    SweepConfig("tiny",   latent_dim=16, hidden_size=32,  depth=2, num_heads=2, max_epochs=4),
    SweepConfig("small",  latent_dim=32, hidden_size=64,  depth=2, num_heads=2, max_epochs=4),
    SweepConfig("wide",   latent_dim=32, hidden_size=128, depth=2, num_heads=4, max_epochs=4),
    SweepConfig("deep",   latent_dim=32, hidden_size=64,  depth=4, num_heads=4, max_epochs=4),
]


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = torch.zeros(1, device="cuda")
        _ = torch.relu(probe)
        return torch.device("cuda")
    except Exception as exc:
        print(f"[WARN] CUDA probe failed ({exc}); falling back to CPU.")
        return torch.device("cpu")


def count_params(modules: List[torch.nn.Module]) -> int:
    seen, total = set(), 0
    for m in modules:
        for p in m.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
    return total


def run_one(
    cfg: SweepConfig,
    ref: sc.AnnData,
    target: sc.AnnData,
    cancer_names: List[str],
    device: torch.device,
    out_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    registry = CancerRegistry({c: i for i, c in enumerate(cancer_names)})
    lumina_cfg = cfg.as_lumina(cancer_names)

    dataset = ReferenceAtlasDataset(ref, lumina_cfg, registry)
    loader = DataLoader(dataset, batch_size=lumina_cfg.batch_size, shuffle=True, num_workers=0)

    transformer = LuminaTransformer(
        latent_dim=lumina_cfg.latent_dim,
        patch_size=1,
        hidden_size=lumina_cfg.hidden_size,
        depth=lumina_cfg.depth,
        num_heads=lumina_cfg.num_heads,
        mlp_ratio=4.0,
        num_classes=len(registry),
        class_dropout_prob=0.1,
    )

    vae = TinyVAE(input_dim=ref.n_vars, latent_dim=lumina_cfg.latent_dim).to(device)
    vae_opt = torch.optim.AdamW(vae.parameters(), lr=lumina_cfg.lr)
    x_ref = torch.as_tensor(np.asarray(ref.X), dtype=torch.float32, device=device)

    vae_curve: List[float] = []
    t0 = time.perf_counter()
    for epoch in range(min(5, lumina_cfg.max_epochs)):
        loss = vae(x_ref)["loss"]
        vae_opt.zero_grad()
        loss.backward()
        vae_opt.step()
        vae_curve.append(float(loss.item()))
    t_vae = time.perf_counter() - t0

    module = LuminaFlowModule(lumina_cfg, transformer, vae=vae).to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=lumina_cfg.lr)

    flow_curve: List[float] = []
    t0 = time.perf_counter()
    for epoch in range(lumina_cfg.max_epochs):
        last = None
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            z, _ = vae.encode_to_latent(x, y)
            loss_dict = module.transport.training_losses(transformer, z.detach(), {"y": y})
            loss = loss_dict["loss"]
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.item())
        flow_curve.append(last if last is not None else float("nan"))
    t_flow = time.perf_counter() - t0

    t0 = time.perf_counter()
    imputer = LuminaImputer(lumina_cfg, module)
    enhanced = imputer.enhance(target, cancer_type=cancer_names[0])
    t_enhance = time.perf_counter() - t0

    evaluator = EnhancementEvaluator(enhanced)
    metrics = evaluator.summary()

    peak_mb = None
    if device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    n_params = count_params([transformer, vae])

    enhanced_path = out_dir / "enhanced" / f"{cfg.name}.h5ad"
    enhanced_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced.write(enhanced_path)

    curve_path = out_dir / "curves" / f"{cfg.name}.json"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(json.dumps({"vae": vae_curve, "flow": flow_curve}, indent=2))

    record = {
        "config": asdict(cfg),
        "device": str(device),
        "n_params": n_params,
        "wall_seconds": {
            "vae_pretrain": t_vae,
            "flow_train": t_flow,
            "enhance": t_enhance,
            "total": t_vae + t_flow + t_enhance,
        },
        "peak_gpu_mem_mb": peak_mb,
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()},
        "data": {
            "ref_cells": int(ref.n_obs),
            "ref_genes": int(ref.n_vars),
            "st_cells": int(target.n_obs),
            "st_genes": int(target.n_vars),
            "cancer_types": cancer_names,
        },
        "enhanced_path": str(enhanced_path.relative_to(PROJECT_ROOT)),
        "loss_curve_path": str(curve_path.relative_to(PROJECT_ROOT)),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "benchmark")
    parser.add_argument("--ref-cells", type=int, default=1200)
    parser.add_argument("--st-cells", type=int, default=300)
    parser.add_argument("--n-genes", type=int, default=120)
    parser.add_argument("--n-cancers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"[bench] Device: {device}")
    print(f"[bench] Output dir: {args.out_dir}")

    ref, target, cancer_names = generate_synthetic_reference_and_st(
        n_ref_cells=args.ref_cells,
        n_st_cells=args.st_cells,
        n_genes=args.n_genes,
        n_cancer_types=args.n_cancers,
        seed=args.seed,
    )
    print(f"[bench] Synthetic ref={ref.shape} target={target.shape} cancers={cancer_names}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for cfg in DEFAULT_SWEEP:
        print(f"\n[bench] --- {cfg.name} ---")
        rec = run_one(cfg, ref, target, cancer_names, device, args.out_dir, args.seed)
        print(
            f"[bench]   total={rec['wall_seconds']['total']:.2f}s "
            f"peak_gpu_mb={rec['peak_gpu_mem_mb']} "
            f"params={rec['n_params']:,}"
        )
        for k, v in rec["metrics"].items():
            print(f"[bench]     {k}: {v}")
        records.append(rec)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_json = args.out_dir / f"lumina_sweep_{ts}.json"
    latest = args.out_dir / "lumina_sweep_latest.json"
    payload = {
        "timestamp": ts,
        "device": str(device),
        "data_settings": {
            "ref_cells": args.ref_cells,
            "st_cells": args.st_cells,
            "n_genes": args.n_genes,
            "n_cancers": args.n_cancers,
            "seed": args.seed,
        },
        "records": records,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    latest.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] Wrote {out_json}")
    print(f"[bench] Wrote {latest}")


if __name__ == "__main__":
    main()
