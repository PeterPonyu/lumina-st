#!/usr/bin/env python3
"""Multi-dataset gene-recovery boxplots + cross-metric rank-aggregation (#266).

Consumes the aggregated benchmark bundle (the ``panels`` schema emitted by
``lumina_st.benchmarks.runner.aggregate_results``). For each recovery metric it
draws one boxplot per method, where the box is built from that method's
per-panel per-gene values (``per_gene_pearson`` / ``per_gene_spearman`` /
``per_gene_rmse`` pooled across panels). It also emits a CSV/markdown summary of
each method's mean rank across metrics (rank aggregation), so a single number
captures "best overall" without picking a favourite metric.

This is a read-only consumer: every value is read verbatim from each method's
``metrics`` block; nothing is recomputed.

Usage::

    python scripts/visualize/fig_recovery_boxplots.py \
        --results results.json --out fig_recovery_boxplots.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make _figbase importable

from _figbase import (  # noqa: E402
    BASELINE_COLOR,
    FOCAL_COLOR,
    FOCAL_METHOD,
    get_pyplot,
    iter_ok_records,
    load_results,
    order_methods,
    require_records,
    save_fig,
    save_table,
)

# (per-gene metric key, display label, higher-is-better) in display order.
METRICS: tuple[tuple[str, str, bool], ...] = (
    ("per_gene_pearson", "PCC", True),
    ("per_gene_spearman", "Spearman", True),
    ("per_gene_rmse", "RMSE", False),
)


def collect_per_gene(bundle: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Return ``{method: {per_gene_key: [pooled per-gene values across panels]}}``."""
    out: dict[str, dict[str, list[float]]] = {}
    for _panel, method, record in iter_ok_records(bundle):
        metrics = record.get("metrics", {})
        for key, _label, _hib in METRICS:
            per_gene = metrics.get(key)
            if not isinstance(per_gene, dict):
                continue
            vals = [
                float(v)
                for v in per_gene.values()
                if isinstance(v, (int, float)) and float(v) == float(v)
            ]
            if vals:
                out.setdefault(method, {}).setdefault(key, []).extend(vals)
    return out


def rank_aggregation(
    per_gene: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    """Mean rank of each method across metrics (rank 1 = best per metric)."""
    methods = order_methods(list(per_gene))
    # metric_key -> {method: mean value}
    means: dict[str, dict[str, float]] = {}
    for key, _label, _hib in METRICS:
        means[key] = {}
        for m in methods:
            vals = per_gene.get(m, {}).get(key)
            if vals:
                means[key][m] = sum(vals) / len(vals)

    # metric_key -> {method: rank}
    ranks: dict[str, dict[str, int]] = {}
    for key, _label, higher_is_better in METRICS:
        scored = means[key]
        ordered = sorted(scored, key=lambda m: scored[m], reverse=higher_is_better)
        ranks[key] = {m: i + 1 for i, m in enumerate(ordered)}

    rows: list[dict[str, Any]] = []
    for m in methods:
        method_ranks = [ranks[key].get(m) for key, _l, _h in METRICS]
        present = [r for r in method_ranks if r is not None]
        if not present:
            continue
        row: dict[str, Any] = {"method": m}
        for (key, label, _hib), r in zip(METRICS, method_ranks):
            row[f"rank_{label}"] = r if r is not None else "nan"
        row["mean_rank"] = sum(present) / len(present)
        rows.append(row)
    rows.sort(key=lambda r: r["mean_rank"])
    return rows


def render_boxplots(per_gene: dict[str, dict[str, list[float]]], title: str):
    """Render one boxplot axes per metric and return the Figure."""
    plt = get_pyplot()
    methods = order_methods(list(per_gene))
    present = [
        (key, label, hib)
        for key, label, hib in METRICS
        if any(per_gene.get(m, {}).get(key) for m in methods)
    ]
    require_records(present, what="recovery boxplots (no per-gene metric present)")

    fig, axes = plt.subplots(
        1, len(present), figsize=(max(4.0, 3.2 * len(present)), 4.6), squeeze=False
    )
    for ax, (key, label, hib) in zip(axes[0], present):
        data = [per_gene.get(m, {}).get(key, []) for m in methods]
        positions = list(range(len(methods)))
        bp = ax.boxplot(
            [d if d else [0.0] for d in data],
            positions=positions,
            patch_artist=True,
            widths=0.6,
        )
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(FOCAL_COLOR if m == FOCAL_METHOD else BASELINE_COLOR)
            patch.set_alpha(0.85)
        arrow = "↑" if hib else "↓"
        ax.set_title(f"{label}  ({arrow} better)", fontsize=11)
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    axes[0][0].set_ylabel("per-gene metric value", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def render_from_bundle(
    results_path: str | Path,
    out_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """Load, render the boxplots, and emit the rank-aggregation table."""
    bundle = load_results(results_path)
    per_gene = collect_per_gene(bundle)
    require_records(list(per_gene), what="recovery boxplots")
    out = Path(out_path)

    rows = rank_aggregation(per_gene)
    columns = ["method", *[f"rank_{label}" for _k, label, _h in METRICS], "mean_rank"]
    save_table(
        rows,
        out.with_name(out.stem + "_rank_aggregation.csv"),
        columns=columns,
        title="Cross-metric rank aggregation (mean rank, lower is better)",
    )

    fig = render_boxplots(per_gene, title or "Multi-dataset gene-recovery distribution")
    return save_fig(fig, out, dpi=dpi)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Aggregated benchmark results JSON.")
    parser.add_argument("--out", required=True, help="Output figure path (e.g. .png).")
    parser.add_argument("--title", default=None, help="Override the figure title.")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = render_from_bundle(args.results, args.out, title=args.title, dpi=args.dpi)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
