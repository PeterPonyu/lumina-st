#!/usr/bin/env python3
"""Shared framework for the LuminaST figure/table generators.

Every generator under ``scripts/visualize`` is a *read-only* consumer of the
aggregated benchmark results JSON emitted by
``lumina_st.benchmarks.runner.aggregate_results`` / ``write_results_json``. That
bundle has the shape::

    {
      "schema_version": "1",
      "panels": {
        "<dataset>/<panel>": {
          "<method>": {
            "status": "ok" | "unavailable:<reason>" | "error:<reason>",
            "runtime_s": <float>,
            "metrics": { ... open metric dict ... },
            "provenance": { ... },
          },
          ...
        },
        ...
      }
    }

This module centralises four concerns the template
(``fig_method_comparison.py``) re-implements ad hoc, so the rest of the
generators stay small and consistent:

* :func:`load_results` — load + validate the ``panels`` schema (clear error on
  malformed input, the schema guard the tests exercise).
* :func:`order_methods` — stable focal-method-first ordering.
* :func:`iter_ok_records` — yield ``(panel, method, record)`` for ``status ==
  "ok"`` rows only, the "no data / synthetic-smoke" guard living in
  :func:`require_records`.
* :func:`save_fig` / :func:`save_table` — write a matplotlib figure (optional,
  guarded so headless CI can skip it) or a table as CSV + markdown (always
  producible, no matplotlib needed).

Matplotlib is imported lazily through :func:`get_pyplot` with the ``Agg``
backend forced first, so importing this module never requires a display and the
table path works even if matplotlib is absent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Method ordering (shared with the comparison template).
# ---------------------------------------------------------------------------

FOCAL_METHOD = "lumina"
FOCAL_COLOR = "#c0392b"  # warm red — the focal method stands out
BASELINE_COLOR = "#4c72b0"  # muted blue for every other method

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


class SchemaError(ValueError):
    """Raised when a results bundle does not match the expected ``panels`` schema."""


# ---------------------------------------------------------------------------
# Loading + schema validation.
# ---------------------------------------------------------------------------


def load_results(results_path: str | Path) -> dict[str, Any]:
    """Load and validate an aggregated benchmark results bundle.

    Validates the top-level ``panels`` mapping and that each panel maps method
    names to record dicts. Raises :class:`SchemaError` with an actionable
    message on any structural violation so a malformed bundle fails loudly
    rather than producing a silently-empty figure.
    """
    path = Path(results_path)
    try:
        data: Any = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise SchemaError(f"{path} must contain a JSON object at the top level.")
    panels = data.get("panels")
    if not isinstance(panels, dict):
        raise SchemaError(
            f"{path} is not a valid results bundle: missing a 'panels' mapping. "
            "Expected the schema emitted by lumina_st.benchmarks.runner.aggregate_results."
        )
    if not panels:
        raise SchemaError(f"{path} contains no panels to plot.")
    for panel_key, methods in panels.items():
        if not isinstance(methods, dict):
            raise SchemaError(
                f"{path}: panel {panel_key!r} must map method names to record dicts, "
                f"got {type(methods).__name__}."
            )
        for method, record in methods.items():
            if not isinstance(record, dict):
                raise SchemaError(
                    f"{path}: record for {panel_key!r}/{method!r} must be a dict, "
                    f"got {type(record).__name__}."
                )
    return data


def order_methods(methods: list[str]) -> list[str]:
    """Return methods in the preferred display order; unknown ones appended sorted."""
    known = [m for m in PREFERRED_METHOD_ORDER if m in methods]
    extra = sorted(m for m in methods if m not in PREFERRED_METHOD_ORDER)
    return known + extra


# ---------------------------------------------------------------------------
# Record iteration + "no data" guard.
# ---------------------------------------------------------------------------


def iter_ok_records(
    bundle: dict[str, Any],
    *,
    metric_keys: tuple[str, ...] | None = None,
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield ``(panel, method, record)`` for every ``status == "ok"`` record.

    When *metric_keys* is given, only records whose ``metrics`` block contains
    at least one of those keys are yielded — so a generator that needs e.g.
    ``empirical_coverage`` transparently skips rows that never measured it.
    """
    for panel, methods in bundle["panels"].items():
        for method, record in methods.items():
            if record.get("status") != "ok":
                continue
            if metric_keys is not None:
                metrics = record.get("metrics", {})
                if not any(k in metrics for k in metric_keys):
                    continue
            yield panel, method, record


def require_records(
    records: list[Any],
    *,
    what: str,
) -> list[Any]:
    """Guard against empty/synthetic-smoke input.

    Raises :class:`SchemaError` when no usable record survived filtering, with a
    message naming *what* was being looked for, so a real-metrics drop-in that
    silently lacks a field fails loudly instead of emitting an empty artifact.
    """
    if not records:
        raise SchemaError(
            f"No usable records for {what}. The bundle has no 'ok' method with the "
            "required metric(s); cannot render. (On synthetic-smoke data, populate "
            "the relevant metric keys first.)"
        )
    return records


# ---------------------------------------------------------------------------
# Output helpers.
# ---------------------------------------------------------------------------


def get_pyplot():
    """Import matplotlib.pyplot with the headless ``Agg`` backend forced.

    Kept lazy so importing ``_figbase`` (and the table-only code paths) never
    requires matplotlib or a display. Raises a clear error if matplotlib is
    unavailable, so the table path can be used independently.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SchemaError(
            "matplotlib is required to render figures but is not installed; "
            "the CSV/markdown table outputs do not need it."
        ) from exc
    matplotlib.use("Agg")  # headless / CI-safe; must precede pyplot import.
    import matplotlib.pyplot as plt

    return plt


def save_fig(fig: Any, output_path: str | Path, *, dpi: int = 200) -> Path:
    """Save a matplotlib figure to *output_path*, creating parent dirs."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt = get_pyplot()
    plt.close(fig)
    return out


def save_table(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    columns: list[str] | None = None,
    title: str | None = None,
) -> tuple[Path, Path]:
    """Write *rows* as both CSV and a GitHub-flavoured markdown table.

    *output_path* is treated as the CSV path; the markdown file is written with
    a ``.md`` suffix beside it. Returns ``(csv_path, md_path)``. Always
    producible without matplotlib.
    """
    if not rows:
        raise SchemaError("save_table received no rows to write.")
    if columns is None:
        # Preserve first-seen key order across all rows.
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)

    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})

    md_path = csv_path.with_suffix(".md")
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = [_fmt_cell(row.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def _fmt_cell(value: Any) -> str:
    """Format one markdown cell: trim floats, leave everything else as ``str``."""
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.4g}"
    return str(value)
