#!/usr/bin/env python3
"""Runtime + peak-memory scalability curves vs #cells / panel size (#267/#288).

Consumes the aggregated benchmark bundle. For each method it plots wall-clock
runtime (record-level ``runtime_s``) and peak memory against a scale axis read
from each record's ``metrics``:

* x-axis #cells  ← ``metrics["n_cells"]`` (the held-out scorer records it).
* panel size     ← ``metrics["n_genes_scored"]`` (used as the marker/secondary
  grouping; reported in the companion table).
* peak memory    ← first present of ``metrics["peak_memory_mb"]`` /
  ``metrics["peak_gpu_mem_mb"]`` (megabytes); omitted from the memory axis when
  no record carries it (the table still lists runtime).

A companion CSV/markdown table lists every (method, panel) point so the raw
scalability numbers are inspectable. Read-only: nothing is recomputed.

Usage::

    python scripts/visualize/fig_runtime_memory.py \
        --results results.json --out fig_runtime_memory.png
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

_MEMORY_KEYS = ("peak_memory_mb", "peak_gpu_mem_mb")


def _memory_mb(metrics: dict[str, Any]) -> float | None:
    for key in _MEMORY_KEYS:
        v = metrics.get(key)
        if isinstance(v, (int, float)) and float(v) == float(v):
            return float(v)
    return None


def collect_points(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """One scalability point per (panel, method): n_cells, runtime_s, memory."""
    points: list[dict[str, Any]] = []
    for panel, method, record in iter_ok_records(bundle):
        metrics = record.get("metrics", {})
        n_cells = metrics.get("n_cells")
        runtime = record.get("runtime_s")
        if not isinstance(runtime, (int, float)):
            continue
        points.append(
            {
                "panel": panel,
                "method": method,
                "n_cells": int(n_cells) if isinstance(n_cells, (int, float)) else None,
                "n_genes_scored": metrics.get("n_genes_scored"),
                "runtime_s": float(runtime),
                "peak_memory_mb": _memory_mb(metrics),
            }
        )
    return points


def render_curves(points: list[dict[str, Any]], title: str):
    """Render runtime + (when available) peak-memory curves vs #cells."""
    plt = get_pyplot()
    methods = order_methods(list({p["method"] for p in points}))
    has_cells = any(p["n_cells"] is not None for p in points)
    has_mem = any(p["peak_memory_mb"] is not None for p in points)

    n_axes = 2 if has_mem else 1
    fig, axes = plt.subplots(1, n_axes, figsize=(5.4 * n_axes, 4.4), squeeze=False)
    runtime_ax = axes[0][0]
    mem_ax = axes[0][1] if has_mem else None

    for method in methods:
        color = FOCAL_COLOR if method == FOCAL_METHOD else BASELINE_COLOR
        mpts = sorted(
            (p for p in points if p["method"] == method),
            key=lambda p: (p["n_cells"] if p["n_cells"] is not None else 0),
        )
        xs = [p["n_cells"] for p in mpts] if has_cells else list(range(len(mpts)))
        runtime_ax.plot(
            xs, [p["runtime_s"] for p in mpts], marker="o", color=color, label=method
        )
        if mem_ax is not None:
            mem_pts = [(x, p["peak_memory_mb"]) for x, p in zip(xs, mpts) if p["peak_memory_mb"] is not None]
            if mem_pts:
                mx, my = zip(*mem_pts)
                mem_ax.plot(mx, my, marker="s", color=color, label=method)

    xlabel = "#cells" if has_cells else "panel index"
    runtime_ax.set_xlabel(xlabel, fontsize=10)
    runtime_ax.set_ylabel("runtime (s)", fontsize=10)
    runtime_ax.set_title("Runtime scalability", fontsize=11)
    if mem_ax is not None:
        mem_ax.set_xlabel(xlabel, fontsize=10)
        mem_ax.set_ylabel("peak memory (MB)", fontsize=10)
        mem_ax.set_title("Memory scalability", fontsize=11)
    for ax in axes[0]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    runtime_ax.legend(fontsize=8, frameon=False)
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
    bundle = load_results(results_path)
    points = require_records(
        collect_points(bundle), what="runtime/memory scalability (no ok record with runtime_s)"
    )
    out = Path(out_path)

    columns = ["method", "panel", "n_cells", "n_genes_scored", "runtime_s", "peak_memory_mb"]
    save_table(
        points,
        out.with_name(out.stem + "_table.csv"),
        columns=columns,
        title="Runtime + peak-memory scalability points",
    )

    fig = render_curves(points, title or "Runtime & memory scalability")
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
