#!/usr/bin/env python3
"""
Render benchmark figures + Markdown report from a sweep JSON.

Reads results/benchmark/lumina_sweep_latest.json and writes:

  docs/benchmark/figures/metric_*.png
  docs/benchmark/figures/loss_curves.png
  docs/benchmark/figures/runtime.png
  docs/benchmark/figures/peak_gpu_mem.png
  docs/benchmark/figures/latent_umap.png      (if scanpy + umap available)
  docs/benchmark/summary.csv
  docs/benchmark/BENCHMARK_REPORT.md          (embeds tables + figures)

The figures directory is checked into git; raw JSON / enhanced AnnData are not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = PROJECT_ROOT / "results" / "benchmark" / "lumina_sweep_latest.json"
DEFAULT_DOCS = PROJECT_ROOT / "docs" / "benchmark"


METRIC_ORDER = [
    ("mean_pearson", "Per-gene Pearson (↑)"),
    ("mean_spearman", "Per-gene Spearman (↑)"),
    ("ari_enhanced_vs_original", "Leiden ARI (↑)"),
    ("nmi_enhanced_vs_original", "Leiden NMI (↑)"),
]


def _bar(ax, names: List[str], values: List[float], ylabel: str, title: str) -> None:
    bars = ax.bar(names, values, color="#4C72B0", edgecolor="#1f1f1f")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for b, v in zip(bars, values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.annotate(f"{v:.3g}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)


def render_metric_bars(records: List[Dict[str, Any]], fig_dir: Path) -> List[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = [r["config"]["name"] for r in records]
    paths: List[Path] = []
    for key, label in METRIC_ORDER:
        vals = [r["metrics"].get(key, float("nan")) for r in records]
        if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
            continue
        fig, ax = plt.subplots(figsize=(5, 3.2))
        _bar(ax, names, [float(v) if v is not None else float("nan") for v in vals], label, label)
        fig.tight_layout()
        p = fig_dir / f"metric_{key}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def render_runtime_and_mem(records: List[Dict[str, Any]], fig_dir: Path) -> Dict[str, Path]:
    names = [r["config"]["name"] for r in records]
    totals = [r["wall_seconds"]["total"] for r in records]
    flow = [r["wall_seconds"]["flow_train"] for r in records]
    vae = [r["wall_seconds"]["vae_pretrain"] for r in records]
    enh = [r["wall_seconds"]["enhance"] for r in records]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(names))
    ax.bar(x, vae, label="VAE pretrain", color="#55A868")
    ax.bar(x, flow, bottom=vae, label="Flow train", color="#4C72B0")
    ax.bar(x, enh, bottom=np.array(vae) + np.array(flow), label="Enhance", color="#C44E52")
    ax.set_xticks(x, names)
    ax.set_ylabel("Wall seconds")
    ax.set_title("Wall-clock breakdown per config")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for i, t in enumerate(totals):
        ax.annotate(f"{t:.1f}s", (i, t), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    runtime_path = fig_dir / "runtime.png"
    fig.savefig(runtime_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    peak = [r["peak_gpu_mem_mb"] if r["peak_gpu_mem_mb"] is not None else 0.0 for r in records]
    _bar(ax, names, peak, "Peak GPU MB", "Peak GPU memory per config")
    fig.tight_layout()
    mem_path = fig_dir / "peak_gpu_mem.png"
    fig.savefig(mem_path, dpi=150)
    plt.close(fig)

    return {"runtime": runtime_path, "mem": mem_path}


def render_loss_curves(records: List[Dict[str, Any]], fig_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for r in records:
        curve_path = PROJECT_ROOT / r["loss_curve_path"]
        if not curve_path.exists():
            continue
        curve = json.loads(curve_path.read_text())
        ax.plot(curve.get("flow", []), label=f"{r['config']['name']} (flow)", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow loss")
    ax.set_title("Flow training loss per config")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "loss_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def render_latent_umap(records: List[Dict[str, Any]], fig_dir: Path) -> Path | None:
    """UMAP comparison of original ST raw vs enhanced latent for the largest config."""
    import scanpy as sc

    try:
        rec = max(records, key=lambda r: r["n_params"])
        enhanced_path = PROJECT_ROOT / rec["enhanced_path"]
        if not enhanced_path.exists():
            return None
        a = sc.read_h5ad(enhanced_path)
        if "latent_enhanced" not in a.obsm:
            return None

        sc.pp.neighbors(a, use_rep="latent_enhanced", n_neighbors=15, key_added="enh")
        sc.tl.umap(a, neighbors_key="enh")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        color_key = "cancer_type" if "cancer_type" in a.obs else None
        sc.pl.umap(a, color=color_key, ax=ax, show=False, frameon=False, title=f"UMAP of enhanced latent ({rec['config']['name']})")
        fig.tight_layout()
        p = fig_dir / "latent_umap.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p
    except Exception as exc:
        print(f"[warn] UMAP rendering failed: {exc}")
        return None


def write_summary_csv(records: List[Dict[str, Any]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "name", "latent_dim", "hidden_size", "depth", "num_heads", "max_epochs",
        "n_params", "wall_total_s", "wall_flow_s", "wall_vae_s", "wall_enhance_s",
        "peak_gpu_mb",
        "mean_pearson", "mean_spearman", "ari", "nmi",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in records:
            c = r["config"]; m = r["metrics"]; t = r["wall_seconds"]
            w.writerow([
                c["name"], c["latent_dim"], c["hidden_size"], c["depth"], c["num_heads"], c["max_epochs"],
                r["n_params"],
                f"{t['total']:.3f}", f"{t['flow_train']:.3f}", f"{t['vae_pretrain']:.3f}", f"{t['enhance']:.3f}",
                r["peak_gpu_mem_mb"],
                m.get("mean_pearson"), m.get("mean_spearman"),
                m.get("ari_enhanced_vs_original"), m.get("nmi_enhanced_vs_original"),
            ])


def _md_metrics_table(records: List[Dict[str, Any]]) -> str:
    lines = [
        "| Config | latent | hidden | depth | heads | epochs | Pearson | Spearman | ARI | NMI |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        c = r["config"]; m = r["metrics"]
        def fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            return f"{float(v):.4f}"
        lines.append(
            f"| `{c['name']}` | {c['latent_dim']} | {c['hidden_size']} | {c['depth']} | {c['num_heads']} | {c['max_epochs']} | "
            f"{fmt(m.get('mean_pearson'))} | {fmt(m.get('mean_spearman'))} | {fmt(m.get('ari_enhanced_vs_original'))} | {fmt(m.get('nmi_enhanced_vs_original'))} |"
        )
    return "\n".join(lines)


def _md_resource_table(records: List[Dict[str, Any]]) -> str:
    lines = [
        "| Config | Params | Wall total (s) | Flow train (s) | VAE pretrain (s) | Enhance (s) | Peak GPU (MB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        c = r["config"]; t = r["wall_seconds"]
        peak = "—" if r["peak_gpu_mem_mb"] is None else f"{r['peak_gpu_mem_mb']:.1f}"
        lines.append(
            f"| `{c['name']}` | {r['n_params']:,} | {t['total']:.2f} | {t['flow_train']:.2f} | {t['vae_pretrain']:.2f} | {t['enhance']:.2f} | {peak} |"
        )
    return "\n".join(lines)


def write_report(
    payload: Dict[str, Any], fig_paths: Dict[str, Any], docs_dir: Path
) -> Path:
    records = payload["records"]
    docs_dir.mkdir(parents=True, exist_ok=True)
    fig_rel = lambda p: f"./figures/{p.name}" if p else None  # noqa: E731

    md = []
    md.append("# LuminaST Synthetic Benchmark Report")
    md.append("")
    md.append(f"- Generated: `{payload['timestamp']}`")
    md.append(f"- Device: `{payload['device']}`")
    md.append(f"- Synthetic data: ref {payload['data_settings']['ref_cells']} cells × {payload['data_settings']['n_genes']} genes, "
              f"target ST {payload['data_settings']['st_cells']} cells, "
              f"{payload['data_settings']['n_cancers']} cancer types, seed {payload['data_settings']['seed']}")
    md.append("")
    md.append("All runs use the same synthetic dataset and the same random seed; only the model configuration changes.")
    md.append("")

    md.append("## Quality metrics per refined version")
    md.append("")
    md.append(_md_metrics_table(records))
    md.append("")

    metric_imgs = [p for p in fig_paths.get("metrics", []) if p is not None]
    if metric_imgs:
        md.append("Per-metric comparison:")
        md.append("")
        for p in metric_imgs:
            md.append(f"![{p.stem}]({fig_rel(p)})")
        md.append("")

    md.append("## Resource usage per refined version")
    md.append("")
    md.append(_md_resource_table(records))
    md.append("")

    rt = fig_paths.get("runtime"); mem = fig_paths.get("mem")
    if rt:
        md.append(f"![runtime breakdown]({fig_rel(rt)})")
        md.append("")
    if mem:
        md.append(f"![peak GPU memory]({fig_rel(mem)})")
        md.append("")

    lc = fig_paths.get("loss_curves")
    if lc:
        md.append("## Training loss curves")
        md.append("")
        md.append(f"![flow loss]({fig_rel(lc)})")
        md.append("")

    umap = fig_paths.get("umap")
    if umap:
        md.append("## Enhanced-latent UMAP (largest config)")
        md.append("")
        md.append(f"![UMAP]({fig_rel(umap)})")
        md.append("")

    md.append("---")
    md.append("")
    md.append("Raw per-config JSON (loss curves, enhanced AnnData paths) lives under `results/benchmark/` "
              "(git-ignored). Re-run with:")
    md.append("")
    md.append("```bash")
    md.append("conda run --no-capture-output -n dl python scripts/benchmark/run_synthetic_sweep.py")
    md.append("conda run --no-capture-output -n dl python scripts/benchmark/make_plots.py")
    md.append("```")
    md.append("")

    out = docs_dir / "BENCHMARK_REPORT.md"
    out.write_text("\n".join(md))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS)
    args = parser.parse_args()

    payload = json.loads(args.sweep_json.read_text())
    records = payload["records"]

    fig_dir = args.docs_dir / "figures"
    metric_paths = render_metric_bars(records, fig_dir)
    rt_mem = render_runtime_and_mem(records, fig_dir)
    lc_path = render_loss_curves(records, fig_dir)
    umap_path = render_latent_umap(records, fig_dir)

    write_summary_csv(records, args.docs_dir / "summary.csv")

    out = write_report(
        payload,
        {
            "metrics": metric_paths,
            "runtime": rt_mem["runtime"],
            "mem": rt_mem["mem"],
            "loss_curves": lc_path,
            "umap": umap_path,
        },
        args.docs_dir,
    )
    print(f"[plots] Wrote {out}")
    print(f"[plots] Summary CSV: {args.docs_dir / 'summary.csv'}")
    print(f"[plots] Figures dir: {fig_dir}")


if __name__ == "__main__":
    main()
