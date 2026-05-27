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

# Round 10 W002 — stable method → color map shared across all 4 figures.
from compose_palette import color_for, palette_for  # noqa: E402

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

    # Round 10 W003a — 2 rows × 3 cols: top row aggregate bars, bottom
    # row spans all 3 cols and shows per-gene Pearson heatmap so a
    # reviewer can see WHICH markers each adapter recovers well.
    fig = plt.figure(figsize=(12, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.5, wspace=0.35)
    for mi, (mk, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        ax = fig.add_subplot(gs[0, mi])
        vals: list[float] = []
        labels: list[str] = []
        for m in method_names:
            v = methods[m].get("metrics", {}).get(mk, float("nan"))
            vals.append(float(v) if v is not None else float("nan"))
            labels.append(m)
        finite = [(lbl, v) for lbl, v in zip(labels, vals) if not math.isnan(v)]
        if not finite:
            ax.text(0.5, 0.5, "no available results",
                    transform=ax.transAxes, ha="center", va="center")
        else:
            xs = np.arange(len(finite))
            colors = palette_for([lbl for lbl, _ in finite])
            bars = ax.bar(xs, [v for _, v in finite], color=colors, alpha=0.9)
            ax.set_xticks(xs)
            ax.set_xticklabels([lbl for lbl, _ in finite], rotation=30, ha="right")
            ax.set_ylabel(mlabel)
            for b, v in zip(bars, [v for _, v in finite]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        _bold_panel_label(ax, chr(ord('a') + mi))
        ax.set_title(mlabel, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        _annotate_placeholder(ax)

    # Panel d — per-gene Pearson heatmap (gene × always-available adapter)
    ax_d = fig.add_subplot(gs[1, :])
    available = [m for m in method_names
                 if methods[m].get("status", "ok") == "ok"
                 and methods[m].get("metrics", {}).get("per_gene_pearson")]
    if not available:
        ax_d.text(0.5, 0.5, "no per-gene Pearson data available",
                  transform=ax_d.transAxes, ha="center", va="center")
        ax_d.axis("off")
    else:
        # Gene set = union of keys across adapters, sorted for stable order.
        gene_set: set[str] = set()
        for m in available:
            gene_set.update(methods[m]["metrics"]["per_gene_pearson"].keys())
        genes = sorted(gene_set)
        matrix = np.full((len(available), len(genes)), np.nan, dtype=np.float64)
        for ai, m in enumerate(available):
            pgp = methods[m]["metrics"]["per_gene_pearson"]
            for gi, g in enumerate(genes):
                v = pgp.get(g)
                if v is not None:
                    try:
                        matrix[ai, gi] = float(v)
                    except (TypeError, ValueError):
                        pass
        im = ax_d.imshow(matrix, aspect="auto", cmap="viridis", vmin=-0.2, vmax=1.0)
        ax_d.set_yticks(np.arange(len(available)))
        ax_d.set_yticklabels(available)
        ax_d.set_xticks(np.arange(len(genes)))
        ax_d.set_xticklabels(genes, rotation=45, ha="right", fontsize=8)
        for ai in range(matrix.shape[0]):
            for gi in range(matrix.shape[1]):
                v = matrix[ai, gi]
                if not math.isnan(v):
                    ax_d.text(gi, ai, f"{v:.2f}", ha="center", va="center",
                              color="white" if v < 0.55 else "black", fontsize=7)
        fig.colorbar(im, ax=ax_d, fraction=0.025, pad=0.02, label="per-gene Pearson ↑")
        ax_d.set_title("Per-gene Pearson by adapter (held-out marker panel)", fontsize=10)
    _bold_panel_label(ax_d, "d")
    _annotate_placeholder(ax_d)

    fig.suptitle("Held-out marker-panel recovery (TME-immune-stromal panel)",
                 fontsize=11)
    _stamp_provenance(fig, source_path, data_card_id)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[compose] {out_path.relative_to(PROJECT_ROOT)}")


# -- Figure 2: sparsity sweep ---------------------------------------------


def compose_sparsity(sparsity_sweep: dict, out_path: Path, source_path: Path, data_card_id: str | None = None) -> None:
    rows = sparsity_sweep["rows"]
    # Aggregate by method (fractions are read directly from rows below).
    methods = sorted({r["method"] for r in rows})

    fig = plt.figure(figsize=(11, 4.2), constrained_layout=True)
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
    method_colors = palette_for(methods)
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

    fig.suptitle("Sparsity / detection-rate sweep", fontsize=11)
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

    # Round 10 W003b — 3 panels now: (a) empirical vs nominal, (b)
    # interval width vs α, (c) coverage-gap heatmap (empirical − nominal)
    # so under/over-coverage by adapter × α is directly readable.
    fig = plt.figure(figsize=(14, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.4)
    method_to_marker = {"mean": "o", "knn": "s"}
    base_names_sorted = sorted({r[0] for r in rows})

    # (a) Empirical vs nominal coverage
    ax_a = fig.add_subplot(gs[0, 0])
    for base_name in base_names_sorted:
        sub = [r for r in rows if r[0] == base_name]
        ax_a.plot(
            [r[2] for r in sub], [r[3] for r in sub],
            marker=method_to_marker.get(base_name, "o"), linestyle="-",
            color=color_for(base_name),
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
    for base_name in base_names_sorted:
        sub = [r for r in rows if r[0] == base_name]
        alphas = [r[1] for r in sub]
        widths = [r[4] for r in sub]
        order = np.argsort(alphas)
        ax_b.plot(
            np.array(alphas)[order], np.array(widths)[order],
            marker=method_to_marker.get(base_name, "o"), linestyle="-",
            color=color_for(base_name),
            label=f"conformal[{base_name}]",
        )
    ax_b.set_xlabel("miscoverage level α")
    ax_b.set_ylabel("mean interval width")
    _bold_panel_label(ax_b, "b")
    ax_b.set_title("Interval width vs α", fontsize=10)
    ax_b.legend(fontsize=8, loc="best")
    ax_b.grid(linestyle=":", alpha=0.5)
    _annotate_placeholder(ax_b)

    # (c) Coverage-gap heatmap (empirical − nominal) over (α, adapter)
    ax_c = fig.add_subplot(gs[0, 2])
    alphas_sorted = sorted({r[1] for r in rows})
    gap = np.full((len(base_names_sorted), len(alphas_sorted)), np.nan, dtype=np.float64)
    for r in rows:
        ai = base_names_sorted.index(r[0])
        aj = alphas_sorted.index(r[1])
        gap[ai, aj] = r[3] - r[2]   # empirical − nominal
    im = ax_c.imshow(gap, aspect="auto", cmap="RdBu_r", vmin=-0.10, vmax=0.10)
    ax_c.set_xticks(np.arange(len(alphas_sorted)))
    ax_c.set_xticklabels([f"{a:g}" for a in alphas_sorted])
    ax_c.set_yticks(np.arange(len(base_names_sorted)))
    ax_c.set_yticklabels(base_names_sorted)
    ax_c.set_xlabel("miscoverage level α")
    for ai in range(gap.shape[0]):
        for aj in range(gap.shape[1]):
            v = gap[ai, aj]
            if not math.isnan(v):
                ax_c.text(aj, ai, f"{v:+.3f}", ha="center", va="center",
                          color="black" if abs(v) < 0.04 else "white", fontsize=8)
    fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04,
                 label="empirical − nominal (0 = calibrated)")
    _bold_panel_label(ax_c, "c")
    ax_c.set_title("Coverage gap by adapter × α", fontsize=10)
    _annotate_placeholder(ax_c)

    fig.suptitle("Conformal prediction-interval calibration",
                 fontsize=11)
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

    # Round 10 W003c — per-adapter panels (a, b, c, ...) plus an aggregate
    # summary panel as the last subplot (grouped bars across adapters per
    # context) so a reviewer can read both per-method and cross-method
    # patterns from the same figure.
    n_panels = len(methods) + 1
    fig = plt.figure(figsize=(min(2.6 * n_panels, 14.0), 4.5), constrained_layout=True)
    gs = fig.add_gridspec(1, n_panels, wspace=0.3)

    # Collect per-method per-context Pearson values once for the aggregate.
    per_method_context: dict[str, dict[str, float]] = {}

    for mi, m in enumerate(methods):
        ax = fig.add_subplot(gs[0, mi])
        folds = per_adapter[m]["folds"]
        contexts = [f["test_context"] for f in folds]
        per_fold_pearson = [
            f.get("result", {}).get("metrics_json", {}).get("mean_pearson", float("nan"))
            for f in folds
        ]
        per_method_context[m] = {c: float(v) for c, v in zip(contexts, per_fold_pearson)
                                 if v is not None and not math.isnan(float(v))}
        # Bootstrap CI from summary
        summary = per_adapter[m].get("summary", {}).get("per_metric", {}).get("mean_pearson", {})
        ci_lo = summary.get("lower", float("nan"))
        ci_hi = summary.get("upper", float("nan"))
        point = summary.get("point", float("nan"))

        xs = np.arange(len(contexts))
        adapter_color = color_for(m)
        ax.bar(xs, per_fold_pearson, color=adapter_color, alpha=0.9)
        ax.axhline(point, color="darkred", linestyle="--", alpha=0.7,
                   label=f"mean = {point:.3f}")
        if not math.isnan(ci_lo) and not math.isnan(ci_hi):
            ax.axhspan(ci_lo, ci_hi, alpha=0.15, color="darkred",
                       label="bootstrap 95% CI")
        ax.set_xticks(xs)
        ax.set_xticklabels(contexts, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("mean Pearson (held-out panel)")
        _bold_panel_label(ax, chr(ord('a') + mi))
        ax.set_title(m, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        _annotate_placeholder(ax)

    # Aggregate summary panel — grouped bars across adapters per context
    ax_sum = fig.add_subplot(gs[0, len(methods)])
    all_contexts: list[str] = []
    for m in methods:
        for c in per_method_context[m]:
            if c not in all_contexts:
                all_contexts.append(c)
    if all_contexts:
        x_pos = np.arange(len(all_contexts))
        group_width = 0.8
        bar_w = group_width / max(len(methods), 1)
        for j, m in enumerate(methods):
            ys = [per_method_context[m].get(c, np.nan) for c in all_contexts]
            ax_sum.bar(x_pos + (j - (len(methods) - 1) / 2) * bar_w, ys,
                       width=bar_w * 0.95, color=color_for(m), label=m, alpha=0.9)
        ax_sum.set_xticks(x_pos)
        ax_sum.set_xticklabels(all_contexts, rotation=20, ha="right", fontsize=8)
        ax_sum.set_ylabel("mean Pearson (held-out panel)")
        ax_sum.legend(fontsize=7, loc="best", framealpha=0.9)
        ax_sum.grid(axis="y", linestyle=":", alpha=0.5)
    else:
        ax_sum.text(0.5, 0.5, "no available results",
                    transform=ax_sum.transAxes, ha="center", va="center")
    _bold_panel_label(ax_sum, chr(ord('a') + len(methods)))
    ax_sum.set_title("All adapters per context", fontsize=10)
    _annotate_placeholder(ax_sum)

    fig.suptitle("Leave-one-context cross-validation across cancer contexts",
                 fontsize=11)
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
