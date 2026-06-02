#!/usr/bin/env python3
"""Render the multi-method gene-recovery comparison panel from a benchmark bundle.

This script consumes the aggregated results JSON emitted by
``lumina_st.benchmarks.runner.aggregate_results`` / ``write_results_json`` and
renders a publication-style comparison panel: one sub-axes per recovery metric
(PCC / Spearman / RMSE / Jaccard / SSIM where present), with one bar per method
inside each axes. The LuminaST method is colour-highlighted and ordered first;
the remaining baselines and competitor adapters follow a stable ordering.

It is a *read-only* consumer of the results contract — it never recomputes a
metric. The metric values are taken verbatim from each method's
``metrics`` block. Per-gene metric dictionaries, when present, are used to draw
standard-error error bars.

Usage::

    python scripts/visualize/fig_method_comparison.py \
        --results results.json \
        --output figure.png \
        --panel "synthetic-coad/tme-immune-stromal"

If ``--panel`` is omitted the script aggregates (means) each metric across all
panels per method, which is the matched-panel summary view.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe backend; must precede pyplot import.
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


@dataclass(frozen=True)
class MetricSpec:
    """Describes one recovery metric: where to read it and how to display it."""

    key: str  # key inside a method's ``metrics`` dict
    label: str  # axis title shown to the reader
    higher_is_better: bool  # arrow direction annotated on the axis
    per_gene_key: str | None = None  # per-gene dict key for error bars, if any


# Metric registry, in display order. Only metrics actually present in the
# bundle are rendered, so a results file without SSIM simply omits that axes.
METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("mean_pearson", "PCC", True, "per_gene_pearson"),
    MetricSpec("mean_spearman", "Spearman", True, "per_gene_spearman"),
    MetricSpec("mean_rmse", "RMSE", False, "per_gene_rmse"),
    MetricSpec("zero_pattern_jaccard", "Jaccard (zero-pattern)", True, None),
    MetricSpec("mean_ssim", "SSIM", True, None),
)

# Preferred left-to-right method ordering: the focal method first, then the
# in-repo simple baselines, then the heavier competitor adapters. Any method
# not listed here is appended afterwards in alphabetical order, so the figure
# never silently drops a method the bundle contains.
PREFERRED_METHOD_ORDER: tuple[str, ...] = (
    "lumina",
    "mean",
    "knn",
    "spatial_neighbor_avg",
    "reference_regression",
    "spaim",
    "tissue",
    "stmcdi",
    "cellt",
)

FOCAL_METHOD = "lumina"
FOCAL_COLOR = "#c0392b"  # warm red — the focal method stands out
BASELINE_COLOR = "#4c72b0"  # muted blue for every other method


def load_results(results_path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an aggregated benchmark results bundle."""
    path = Path(results_path)
    data: dict[str, Any] = json.loads(path.read_text())
    if "panels" not in data or not isinstance(data["panels"], dict):
        raise ValueError(
            f"{path} is not a valid results bundle: missing a 'panels' mapping."
        )
    if not data["panels"]:
        raise ValueError(f"{path} contains no panels to plot.")
    return data


def order_methods(methods: list[str]) -> list[str]:
    """Return methods in the preferred display order, unknown ones appended."""
    known = [m for m in PREFERRED_METHOD_ORDER if m in methods]
    extra = sorted(m for m in methods if m not in PREFERRED_METHOD_ORDER)
    return known + extra


def _sem(values: list[float]) -> float:
    """Standard error of the mean over finite values (0.0 if < 2 samples)."""
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    n = len(finite)
    if n < 2:
        return 0.0
    mean = sum(finite) / n
    var = sum((v - mean) ** 2 for v in finite) / (n - 1)
    return math.sqrt(var / n)


def extract_method_metrics(
    bundle: dict[str, Any],
    panel: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Collapse the bundle into ``{method: {metric_key: {"value", "err"}}}``.

    Only methods with ``status == "ok"`` are kept. When ``panel`` is given only
    that panel is read; otherwise each metric is averaged across every panel in
    which the method ran (the matched-panel summary view). Error bars come from
    the per-gene metric dictionaries when available, else 0.
    """
    panels: dict[str, Any] = bundle["panels"]
    if panel is not None:
        if panel not in panels:
            raise KeyError(
                f"Panel {panel!r} not in bundle. Available: {sorted(panels)}"
            )
        selected = {panel: panels[panel]}
    else:
        selected = panels

    # method -> metric_key -> list of (value, err) collected across panels
    acc: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for panel_methods in selected.values():
        for method, record in panel_methods.items():
            if record.get("status") != "ok":
                continue
            metrics = record.get("metrics", {})
            for spec in METRIC_SPECS:
                raw = metrics.get(spec.key)
                if raw is None:
                    continue
                value = float(raw)
                if not math.isfinite(value):
                    continue
                err = 0.0
                if spec.per_gene_key is not None:
                    per_gene = metrics.get(spec.per_gene_key)
                    if isinstance(per_gene, dict):
                        err = _sem([float(v) for v in per_gene.values()])
                acc.setdefault(method, {}).setdefault(spec.key, []).append((value, err))

    out: dict[str, dict[str, dict[str, float]]] = {}
    for method, by_metric in acc.items():
        out[method] = {}
        for key, pairs in by_metric.items():
            values = [v for v, _ in pairs]
            errs = [e for _, e in pairs]
            mean_value = sum(values) / len(values)
            # Across multiple panels, combine per-gene SEMs in quadrature and
            # divide by the panel count; a single panel keeps its own SEM.
            mean_err = math.sqrt(sum(e**2 for e in errs)) / len(errs)
            out[method][key] = {"value": mean_value, "err": mean_err}
    if not out:
        raise ValueError(
            "No method with status 'ok' and a finite metric was found in the bundle."
        )
    return out


def render_comparison_panel(
    method_metrics: dict[str, dict[str, dict[str, float]]],
    title: str = "Multi-method gene recovery",
) -> Figure:
    """Render the grouped comparison panel and return the matplotlib Figure."""
    methods = order_methods(list(method_metrics))

    # Keep only metrics that at least one method reports, preserving registry order.
    present_specs = [
        spec
        for spec in METRIC_SPECS
        if any(spec.key in method_metrics[m] for m in methods)
    ]
    if not present_specs:
        raise ValueError("None of the known recovery metrics are present in the bundle.")

    colors = [FOCAL_COLOR if m == FOCAL_METHOD else BASELINE_COLOR for m in methods]
    x_positions = list(range(len(methods)))

    n_axes = len(present_specs)
    fig, axes = plt.subplots(
        1,
        n_axes,
        figsize=(max(4.0, 3.1 * n_axes), 4.6),
        squeeze=False,
    )
    flat_axes = list(axes[0])

    for ax, spec in zip(flat_axes, present_specs):
        values = [method_metrics[m].get(spec.key, {}).get("value", math.nan) for m in methods]
        errs = [method_metrics[m].get(spec.key, {}).get("err", 0.0) for m in methods]
        has_err = any(e > 0 for e in errs)
        ax.bar(
            x_positions,
            values,
            color=colors,
            yerr=errs if has_err else None,
            capsize=3 if has_err else 0,
            edgecolor="black",
            linewidth=0.6,
            error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
        )
        arrow = "↑" if spec.higher_is_better else "↓"
        ax.set_title(f"{spec.label}  ({arrow} better)", fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        ax.margins(x=0.05)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    flat_axes[0].set_ylabel("metric value", fontsize=10)

    # Shared legend distinguishing the focal method from the baselines.
    focal_handle = Patch(facecolor=FOCAL_COLOR, edgecolor="black", linewidth=0.6)
    baseline_handle = Patch(facecolor=BASELINE_COLOR, edgecolor="black", linewidth=0.6)
    fig.legend(
        [focal_handle, baseline_handle],
        [f"{FOCAL_METHOD} (this work)", "baseline / competitor"],
        loc="upper right",
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def render_from_bundle(
    results_path: str | Path,
    output_path: str | Path,
    panel: str | None = None,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    """End-to-end: load a bundle, render the panel, and save it to ``output_path``."""
    bundle = load_results(results_path)
    method_metrics = extract_method_metrics(bundle, panel=panel)
    panel_label = panel if panel is not None else "all panels (mean)"
    fig = render_comparison_panel(
        method_metrics,
        title=title or f"Multi-method gene recovery — {panel_label}",
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", required=True, help="Path to the aggregated benchmark results JSON."
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the rendered figure (e.g. .png/.pdf)."
    )
    parser.add_argument(
        "--panel",
        default=None,
        help="Specific '<dataset>/<panel>' key to plot; default aggregates all panels.",
    )
    parser.add_argument("--title", default=None, help="Override the figure title.")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = render_from_bundle(
        results_path=args.results,
        output_path=args.output,
        panel=args.panel,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
