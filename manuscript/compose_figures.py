#!/usr/bin/env python3
"""Compose multi-panel figures for the LuminaST manuscript.

Reads the synthetic benchmark JSONs under `lumina-st/results/benchmark/`
and renders each manuscript figure as a single PDF in `figures/` — the
gridspec composition pattern matches what reference spatial-omics papers
do (multi-panel figures assembled before LaTeX embed, not via LaTeX
subfigure layouts).

Every figure is a placeholder backed by synthetic data; replace the
input JSONs (or update the loaders below) when real data lands.

Build chain: `make figures` in this directory invokes this script.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
BENCH_DIR = PROJECT_ROOT / "results" / "benchmark"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


PLACEHOLDER_BANNER = "synthetic placeholder — real data pending"


def _load_or_warn(path: Path) -> dict | None:
    if not path.exists():
        print(f"[compose] WARN: missing benchmark artifact {path.relative_to(PROJECT_ROOT)}; "
              f"skipping this figure", file=sys.stderr)
        return None
    return json.loads(path.read_text())


def _is_synthetic(payload: dict | None) -> bool:
    """Heuristic: payload provenance indicates synthetic source."""
    if payload is None:
        return True
    txt = json.dumps(payload)[:4096]
    return "synthetic" in txt.lower()


def _annotate_placeholder(ax) -> None:
    """Stamp a small banner so reviewers can never confuse synthetic with real."""
    ax.text(
        0.99, 0.01, PLACEHOLDER_BANNER,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=6, style="italic", color="grey",
    )


def _bold_panel_label(ax, letter: str) -> None:
    """Bold lowercase panel label at the top-left of each panel.

    Mirrors the field-conventional style documented in
    `docs/REFERENCE_FIGURE_STYLE.md`. Letter is rendered outside the data
    axes so it doesn't collide with subplot content.
    """
    ax.text(
        -0.10, 1.05, letter,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=12, fontweight="bold",
    )


def _stamp_provenance(fig, source_path: Path, data_card_id: str | None) -> None:
    """Foot-of-figure provenance stamp so reviewers know which JSON + data card backed the figure."""
    try:
        rel = source_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = source_path
    card = f"  ·  data_card_id={data_card_id}" if data_card_id else ""
    fig.text(
        0.99, 0.005,
        f"source: {rel}{card}",
        ha="right", va="bottom",
        fontsize=5.5, style="italic", color="dimgrey",
        family="monospace",
    )


# -- Figure 1: held-out panel (Pearson / Spearman / RMSE per adapter) -------


def compose_heldout(synthetic_smoke: dict, out_path: Path, source_path: Path, data_card_id: str | None = None) -> None:
    panels = synthetic_smoke["panels"]
    key = next(iter(panels))
    methods = panels[key]
    method_names = sorted(methods.keys())

    metric_keys = ("mean_pearson", "mean_spearman", "mean_rmse")
    metric_labels = ("mean Pearson ↑", "mean Spearman ↑", "mean RMSE ↓")

    fig = plt.figure(figsize=(12, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.35)
    for mi, (mk, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        ax = fig.add_subplot(gs[0, mi])
        vals: list[float] = []
        labels: list[str] = []
        for m in method_names:
            v = methods[m].get("metrics", {}).get(mk, float("nan"))
            vals.append(float(v) if v is not None else float("nan"))
            labels.append(m)
        finite = [(l, v) for l, v in zip(labels, vals) if not math.isnan(v)]
        if not finite:
            ax.text(0.5, 0.5, "no available results",
                    transform=ax.transAxes, ha="center", va="center")
        else:
            xs = np.arange(len(finite))
            bars = ax.bar(xs, [v for _, v in finite], color="steelblue", alpha=0.85)
            ax.set_xticks(xs)
            ax.set_xticklabels([l for l, _ in finite], rotation=30, ha="right")
            ax.set_ylabel(mlabel)
            for b, v in zip(bars, [v for _, v in finite]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        _bold_panel_label(ax, chr(ord('a') + mi))
        ax.set_title(mlabel, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        _annotate_placeholder(ax)

    fig.suptitle("Held-out marker-panel recovery (TME-immune-stromal panel)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _stamp_provenance(fig, source_path, data_card_id)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[compose] {out_path.relative_to(PROJECT_ROOT)}")


# -- Figure 2: sparsity sweep ---------------------------------------------


def compose_sparsity(sparsity_sweep: dict, out_path: Path, source_path: Path, data_card_id: str | None = None) -> None:
    rows = sparsity_sweep["rows"]
    # Aggregate by (method, fraction)
    fractions = sorted({r["fraction"] for r in rows})
    methods = sorted({r["method"] for r in rows})

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 2, wspace=0.32, width_ratios=[1.0, 1.4])

    # (a) Observed sparsity vs fraction (one line; method-invariant)
    ax_a = fig.add_subplot(gs[0, 0])
    by_fraction: dict[float, list[float]] = {}
    for r in rows:
        by_fraction.setdefault(r["fraction"], []).append(r["sparsity_observed_thinned"])
    fs = sorted(by_fraction.keys(), reverse=True)
    ys = [np.mean(by_fraction[f]) for f in fs]
    ax_a.plot(fs, ys, marker="o", color="darkorange")
    ax_a.set_xlabel("detection-rate fraction")
    ax_a.set_ylabel("observed sparsity (fraction of zeros)")
    _bold_panel_label(ax_a, "a")
    ax_a.set_title("Observed sparsity grows monotonically as fraction drops", fontsize=10)
    ax_a.invert_xaxis()
    ax_a.grid(linestyle=":", alpha=0.5)
    _annotate_placeholder(ax_a)

    # (b) Per-method mean Pearson at each fraction (only ok rows)
    ax_b = fig.add_subplot(gs[0, 1])
    method_colors = plt.colormaps["tab10"](np.linspace(0, 1, len(methods)))
    for m, color in zip(methods, method_colors):
        m_rows = [r for r in rows if r["method"] == m and r["status"] == "ok"]
        if not m_rows:
            continue
        f_to_pearson: dict[float, float] = {}
        for r in m_rows:
            mp = r.get("metrics_json", {}).get("mean_pearson")
            if mp is not None and not math.isnan(mp):
                f_to_pearson[r["fraction"]] = mp
        if not f_to_pearson:
            continue
        sorted_f = sorted(f_to_pearson.keys(), reverse=True)
        ax_b.plot(
            sorted_f, [f_to_pearson[f] for f in sorted_f],
            marker="o", color=color, label=m,
        )
    ax_b.set_xlabel("detection-rate fraction")
    ax_b.set_ylabel("mean Pearson on held-out panel")
    _bold_panel_label(ax_b, "b")
    ax_b.set_title("Per-method recovery vs detection rate", fontsize=10)
    ax_b.invert_xaxis()
    ax_b.grid(linestyle=":", alpha=0.5)
    ax_b.legend(loc="best", fontsize=8, framealpha=0.9)
    _annotate_placeholder(ax_b)

    fig.suptitle("Sparsity / detection-rate sweep", fontsize=11, y=1.02)
    fig.tight_layout()
    _stamp_provenance(fig, source_path, data_card_id)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[compose] {out_path.relative_to(PROJECT_ROOT)}")


# -- Figure 3: conformal calibration ---------------------------------------


def compose_conformal(out_path: Path, data_card_id: str | None = None) -> None:
    """Compose the conformal-calibration figure from a fresh deterministic run.

    Avoids requiring a pre-stored JSON — the calibrator is fast enough to
    re-run inline here. Keeps the figure pipeline self-contained.
    """
    try:
        import anndata as ad
        from lumina_st.benchmarks import AdapterInput, ConformalCalibrator
        from lumina_st.benchmarks.adapters import KNNAdapter, MeanAdapter
    except ImportError as exc:
        print(f"[compose] WARN: lumina_st not importable ({exc}); "
              f"skipping conformal figure", file=sys.stderr)
        return

    rng = np.random.default_rng(7)
    X = rng.poisson(3.0, size=(400, 30)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE_{i:03d}" for i in range(30)]
    held_out = ["GENE_005", "GENE_010", "GENE_015", "GENE_020"]
    inp = AdapterInput(input_h5ad=adata, held_out_genes=held_out, seed=42)

    rows: list[tuple[str, float, float, float, float]] = []
    for base, base_name in [(MeanAdapter(), "mean"), (KNNAdapter(k=10), "knn")]:
        for alpha in (0.05, 0.10, 0.20):
            calibrator = ConformalCalibrator(base=base, alpha=alpha)
            r = calibrator.run(inp)
            m = r.metrics_json
            rows.append((
                base_name, alpha,
                m["nominal_coverage"],
                m["empirical_coverage"],
                m["mean_interval_width"],
            ))

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 2, wspace=0.35)

    # (a) Empirical vs nominal coverage
    ax_a = fig.add_subplot(gs[0, 0])
    method_to_marker = {"mean": "o", "knn": "s"}
    for base_name in {r[0] for r in rows}:
        sub = [r for r in rows if r[0] == base_name]
        ax_a.plot(
            [r[2] for r in sub], [r[3] for r in sub],
            marker=method_to_marker[base_name], linestyle="-",
            label=f"conformal[{base_name}]",
        )
    ax_a.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.4,
              label="perfect calibration")
    ax_a.set_xlim(0.7, 1.0)
    ax_a.set_ylim(0.7, 1.0)
    ax_a.set_xlabel("nominal coverage  (1 − α)")
    ax_a.set_ylabel("empirical coverage on held-out cells")
    _bold_panel_label(ax_a, "a")
    ax_a.set_title("Empirical vs nominal coverage", fontsize=10)
    ax_a.legend(fontsize=8, loc="lower right")
    ax_a.grid(linestyle=":", alpha=0.5)
    _annotate_placeholder(ax_a)

    # (b) Interval width vs α
    ax_b = fig.add_subplot(gs[0, 1])
    for base_name in {r[0] for r in rows}:
        sub = [r for r in rows if r[0] == base_name]
        alphas = [r[1] for r in sub]
        widths = [r[4] for r in sub]
        order = np.argsort(alphas)
        ax_b.plot(
            np.array(alphas)[order], np.array(widths)[order],
            marker=method_to_marker[base_name], linestyle="-",
            label=f"conformal[{base_name}]",
        )
    ax_b.set_xlabel("miscoverage level α")
    ax_b.set_ylabel("mean interval width")
    _bold_panel_label(ax_b, "b")
    ax_b.set_title("Interval width vs α", fontsize=10)
    ax_b.legend(fontsize=8, loc="best")
    ax_b.grid(linestyle=":", alpha=0.5)
    _annotate_placeholder(ax_b)

    fig.suptitle("Conformal prediction-interval calibration",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    # The conformal figure re-runs the calibrator inline; the "source" is the
    # ConformalCalibrator class itself, not a benchmark JSON.
    _stamp_provenance(fig, Path("lumina_st.benchmarks.uncertainty.ConformalCalibrator"), data_card_id)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[compose] {out_path.relative_to(PROJECT_ROOT)}")


# -- Figure 4: leave-one-context CV ----------------------------------------


def compose_loo(loo: dict, out_path: Path, source_path: Path, data_card_id: str | None = None) -> None:
    per_adapter = loo["per_adapter"]
    methods = sorted(per_adapter.keys())

    fig = plt.figure(figsize=(11, 4.5))
    gs = fig.add_gridspec(1, len(methods), wspace=0.3)

    for mi, m in enumerate(methods):
        ax = fig.add_subplot(gs[0, mi])
        folds = per_adapter[m]["folds"]
        contexts = [f["test_context"] for f in folds]
        per_fold_pearson = [
            f.get("result", {}).get("metrics_json", {}).get("mean_pearson", float("nan"))
            for f in folds
        ]
        # Bootstrap CI from summary
        summary = per_adapter[m].get("summary", {}).get("per_metric", {}).get("mean_pearson", {})
        ci_lo = summary.get("lower", float("nan"))
        ci_hi = summary.get("upper", float("nan"))
        point = summary.get("point", float("nan"))

        xs = np.arange(len(contexts))
        bars = ax.bar(xs, per_fold_pearson, color="steelblue", alpha=0.85)
        ax.axhline(point, color="darkred", linestyle="--", alpha=0.7,
                   label=f"mean = {point:.3f}")
        if not math.isnan(ci_lo) and not math.isnan(ci_hi):
            ax.axhspan(ci_lo, ci_hi, alpha=0.15, color="darkred",
                       label=f"bootstrap 95% CI")
        ax.set_xticks(xs)
        ax.set_xticklabels(contexts, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("mean Pearson (held-out panel)")
        _bold_panel_label(ax, chr(ord('a') + mi))
        ax.set_title(m, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        _annotate_placeholder(ax)

    fig.suptitle("Leave-one-context cross-validation across cancer contexts",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _stamp_provenance(fig, source_path, data_card_id)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[compose] {out_path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heldout-json", type=Path,
        default=BENCH_DIR / "synthetic_smoke.json",
        help="Benchmark JSON for the held-out panel figure.",
    )
    parser.add_argument(
        "--sparsity-json", type=Path,
        default=BENCH_DIR / "sparsity_sweep.json",
        help="Benchmark JSON for the sparsity sweep figure.",
    )
    parser.add_argument(
        "--loo-json", type=Path,
        default=BENCH_DIR / "leave_one_context.json",
        help="Benchmark JSON for the leave-one-context figure.",
    )
    parser.add_argument(
        "--data-card-id", type=str, default=None,
        help="Optional data card id stamped on every figure's provenance footer.",
    )
    args = parser.parse_args()

    print(f"[compose] benchmark dir: {BENCH_DIR}")
    print(f"[compose] figures out:    {FIG_DIR}")
    if args.data_card_id:
        print(f"[compose] data_card_id:   {args.data_card_id}")

    smoke = _load_or_warn(args.heldout_json)
    if smoke:
        compose_heldout(smoke, FIG_DIR / "fig_heldout_panel.pdf", args.heldout_json, args.data_card_id)

    sparsity = _load_or_warn(args.sparsity_json)
    if sparsity:
        compose_sparsity(sparsity, FIG_DIR / "fig_sparsity_sweep.pdf", args.sparsity_json, args.data_card_id)

    compose_conformal(FIG_DIR / "fig_conformal_calibration.pdf", args.data_card_id)

    loo = _load_or_warn(args.loo_json)
    if loo:
        compose_loo(loo, FIG_DIR / "fig_leave_one_context.pdf", args.loo_json, args.data_card_id)

    print("[compose] all figures composed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
