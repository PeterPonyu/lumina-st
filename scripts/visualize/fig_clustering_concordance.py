#!/usr/bin/env python3
"""Clustering-uplift bars: raw vs enhanced × ARI/AMI/NMI/homogeneity (#287).

Consumes the aggregated benchmark bundle. For each clustering metric it draws a
paired bar (raw-vs-GT and enhanced-vs-GT) and annotates the signed uplift
(``Δ = enhanced − raw``), reading verbatim from the
``lumina_st.metrics.enhancement_evaluator`` keys carried in a method's
``metrics`` block:

* ``<m>_raw_vs_gt``       — clustering agreement of the raw/observed embedding vs GT.
* ``<m>_enhanced_vs_gt``  — agreement of the enhanced embedding vs GT.
* ``<m>_delta_over_raw``  — signed Δ (used verbatim when present, else computed).

for ``m`` in {``ari``, ``ami``, ``nmi``, ``homogeneity``}. A companion
CSV/markdown table lists raw / enhanced / Δ per metric. By default the focal
method (``lumina``) is plotted; ``--method`` overrides. Read-only.

Usage::

    python scripts/visualize/fig_clustering_concordance.py \
        --results results.json --out fig_clustering_concordance.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make _figbase importable

from _figbase import (  # noqa: E402
    FOCAL_METHOD,
    get_pyplot,
    iter_ok_records,
    load_results,
    require_records,
    save_fig,
    save_table,
)

CLUSTERING_METRICS: tuple[tuple[str, str], ...] = (
    ("ari", "ARI"),
    ("ami", "AMI"),
    ("nmi", "NMI"),
    ("homogeneity", "Homogeneity"),
)

RAW_COLOR = "#95a5a6"  # grey — the raw/observed baseline
ENHANCED_COLOR = "#c0392b"  # warm red — the enhanced (focal) result


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and float(value) == float(value):
        return float(value)
    return None


def collect_concordance(
    bundle: dict[str, Any], method: str
) -> list[dict[str, Any]]:
    """Mean raw / enhanced / Δ per clustering metric for *method*, pooled across panels."""
    # metric_key -> {"raw": [...], "enhanced": [...], "delta": [...]}
    acc: dict[str, dict[str, list[float]]] = {
        m: {"raw": [], "enhanced": [], "delta": []} for m, _l in CLUSTERING_METRICS
    }
    for _panel, rec_method, record in iter_ok_records(bundle):
        if rec_method != method:
            continue
        metrics = record.get("metrics", {})
        for m, _label in CLUSTERING_METRICS:
            raw = _finite(metrics.get(f"{m}_raw_vs_gt"))
            enh = _finite(metrics.get(f"{m}_enhanced_vs_gt"))
            if raw is None and enh is None:
                continue
            delta = _finite(metrics.get(f"{m}_delta_over_raw"))
            if delta is None and raw is not None and enh is not None:
                delta = enh - raw
            if raw is not None:
                acc[m]["raw"].append(raw)
            if enh is not None:
                acc[m]["enhanced"].append(enh)
            if delta is not None:
                acc[m]["delta"].append(delta)

    rows: list[dict[str, Any]] = []
    for m, label in CLUSTERING_METRICS:
        bucket = acc[m]
        if not bucket["raw"] and not bucket["enhanced"]:
            continue
        rows.append(
            {
                "metric": label,
                "raw_vs_gt": _mean(bucket["raw"]),
                "enhanced_vs_gt": _mean(bucket["enhanced"]),
                "delta_over_raw": _mean(bucket["delta"]),
            }
        )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def render_bars(rows: list[dict[str, Any]], method: str, title: str):
    """Paired raw-vs-enhanced bars per metric with signed Δ annotations."""
    plt = get_pyplot()
    labels = [r["metric"] for r in rows]
    x = list(range(len(rows)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(5.0, 1.6 * len(rows)), 4.6))
    ax.bar(
        [i - width / 2 for i in x],
        [r["raw_vs_gt"] for r in rows],
        width,
        color=RAW_COLOR,
        edgecolor="black",
        linewidth=0.6,
        label="raw vs GT",
    )
    ax.bar(
        [i + width / 2 for i in x],
        [r["enhanced_vs_gt"] for r in rows],
        width,
        color=ENHANCED_COLOR,
        edgecolor="black",
        linewidth=0.6,
        label=f"{method} enhanced vs GT",
    )
    for i, r in zip(x, rows):
        delta = r["delta_over_raw"]
        if delta == delta:  # not NaN
            ax.annotate(
                f"Δ {delta:+.3f}",
                xy=(i, max(r["raw_vs_gt"], r["enhanced_vs_gt"], 0.0)),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#c0392b" if delta >= 0 else "#1f618d",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("clustering agreement vs ground truth", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def render_from_bundle(
    results_path: str | Path,
    out_path: str | Path,
    *,
    method: str = FOCAL_METHOD,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    bundle = load_results(results_path)
    rows = require_records(
        collect_concordance(bundle, method),
        what=f"clustering concordance for method {method!r} (no ARI/AMI/NMI/homogeneity keys)",
    )
    out = Path(out_path)
    save_table(
        rows,
        out.with_name(out.stem + "_table.csv"),
        columns=["metric", "raw_vs_gt", "enhanced_vs_gt", "delta_over_raw"],
        title=f"Clustering uplift ({method}): raw vs enhanced with signed Δ",
    )
    fig = render_bars(rows, method, title or f"Clustering uplift — {method}")
    return save_fig(fig, out, dpi=dpi)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Aggregated benchmark results JSON.")
    parser.add_argument("--out", required=True, help="Output figure path (e.g. .png).")
    parser.add_argument(
        "--method", default=FOCAL_METHOD, help=f"Method to plot (default: {FOCAL_METHOD})."
    )
    parser.add_argument("--title", default=None, help="Override the figure title.")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = render_from_bundle(
        args.results, args.out, method=args.method, title=args.title, dpi=args.dpi
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
