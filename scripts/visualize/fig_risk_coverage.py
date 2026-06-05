#!/usr/bin/env python3
"""Selective-prediction risk-coverage / calibration figure (#291).

Consumes the aggregated benchmark bundle and renders the uncertainty
calibration of each method that reports it. Two modes, auto-selected per record:

* Risk-coverage curve — when a record's ``metrics`` carries a ``risk_coverage``
  array of ``{"coverage": c, "risk": r}`` points (or parallel ``coverage`` /
  ``risk`` lists), the curve is drawn directly.
* Calibration point — otherwise, when a record carries the conformal-interval
  keys from ``lumina_st.benchmarks.uncertainty`` (``empirical_coverage`` and
  ``nominal_coverage``), a single (nominal, empirical) calibration point is
  plotted against the y=x perfect-calibration diagonal.

A companion CSV/markdown table lists nominal vs empirical coverage (and interval
width when present) per method. Read-only: nothing is recomputed.

Usage::

    python scripts/visualize/fig_risk_coverage.py \
        --results results.json --out fig_risk_coverage.png
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

_COVERAGE_KEYS = ("risk_coverage", "empirical_coverage")


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and float(value) == float(value):
        return float(value)
    return None


def _risk_coverage_curve(metrics: dict[str, Any]) -> list[tuple[float, float]] | None:
    """Extract a sorted ``[(coverage, risk), ...]`` curve if the record has one."""
    rc = metrics.get("risk_coverage")
    points: list[tuple[float, float]] = []
    if isinstance(rc, list):
        for item in rc:
            if isinstance(item, dict):
                c = _finite(item.get("coverage"))
                r = _finite(item.get("risk"))
                if c is not None and r is not None:
                    points.append((c, r))
    elif isinstance(rc, dict):
        cov = rc.get("coverage")
        risk = rc.get("risk")
        if isinstance(cov, list) and isinstance(risk, list) and len(cov) == len(risk):
            for c, r in zip(cov, risk):
                fc, fr = _finite(c), _finite(r)
                if fc is not None and fr is not None:
                    points.append((fc, fr))
    if not points:
        return None
    points.sort(key=lambda p: p[0])
    return points


def collect(bundle: dict[str, Any]) -> tuple[dict[str, list[tuple[float, float]]], list[dict[str, Any]]]:
    """Return ``({method: risk_coverage_curve}, calibration_table_rows)``."""
    curves: dict[str, list[tuple[float, float]]] = {}
    rows: list[dict[str, Any]] = []
    for _panel, method, record in iter_ok_records(bundle, metric_keys=_COVERAGE_KEYS):
        metrics = record.get("metrics", {})
        curve = _risk_coverage_curve(metrics)
        if curve is not None:
            curves.setdefault(method, [])
            curves[method] = curve
        empirical = _finite(metrics.get("empirical_coverage"))
        nominal = _finite(metrics.get("nominal_coverage"))
        if empirical is not None or nominal is not None:
            rows.append(
                {
                    "method": method,
                    "nominal_coverage": nominal if nominal is not None else float("nan"),
                    "empirical_coverage": empirical if empirical is not None else float("nan"),
                    "mean_interval_width": _finite(metrics.get("mean_interval_width")) or float("nan"),
                    "calibration_gap": (empirical - nominal)
                    if (empirical is not None and nominal is not None)
                    else float("nan"),
                }
            )
    return curves, rows


def render(curves: dict[str, list[tuple[float, float]]], cal_rows: list[dict[str, Any]], title: str):
    """Risk-coverage curves and/or calibration scatter vs the y=x diagonal."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    has_curve = bool(curves)

    if has_curve:
        for method in order_methods(list(curves)):
            color = FOCAL_COLOR if method == FOCAL_METHOD else BASELINE_COLOR
            xs = [c for c, _r in curves[method]]
            ys = [r for _c, r in curves[method]]
            ax.plot(xs, ys, marker="o", color=color, label=method)
        ax.set_xlabel("coverage (fraction of cells retained)", fontsize=10)
        ax.set_ylabel("selective risk", fontsize=10)
        ax.set_title("Risk-coverage", fontsize=11)
    else:
        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="perfect calibration")
        for method in order_methods(list({r["method"] for r in cal_rows})):
            color = FOCAL_COLOR if method == FOCAL_METHOD else BASELINE_COLOR
            pts = [
                (r["nominal_coverage"], r["empirical_coverage"])
                for r in cal_rows
                if r["method"] == method
                and r["nominal_coverage"] == r["nominal_coverage"]
                and r["empirical_coverage"] == r["empirical_coverage"]
            ]
            if pts:
                nx, ey = zip(*pts)
                ax.scatter(nx, ey, color=color, s=60, edgecolor="black", linewidth=0.6, label=method)
        ax.set_xlabel("nominal coverage (1 − α)", fontsize=10)
        ax.set_ylabel("empirical coverage", fontsize=10)
        ax.set_title("Coverage calibration", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9, frameon=False)
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
    curves, cal_rows = collect(bundle)
    require_records(
        list(curves) or cal_rows,
        what="risk-coverage / calibration (no empirical_coverage or risk_coverage)",
    )
    out = Path(out_path)
    if cal_rows:
        save_table(
            cal_rows,
            out.with_name(out.stem + "_calibration.csv"),
            columns=[
                "method",
                "nominal_coverage",
                "empirical_coverage",
                "mean_interval_width",
                "calibration_gap",
            ],
            title="Selective-prediction coverage calibration",
        )
    fig = render(curves, cal_rows, title or "Selective prediction: risk-coverage / calibration")
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
