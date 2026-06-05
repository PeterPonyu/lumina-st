#!/usr/bin/env python3
"""Held-out gene-recovery table stratified by expression & sparsity bin (#270/#292).

Emits CSV + markdown only (no figure needed). For each method it stratifies the
per-gene recovery scores into bins by:

* expression level — read from each gene's ``per_gene_mean_expression`` if the
  bundle carries it, else estimated from the gene's own per-gene RMSE magnitude
  is NOT used; instead, when no expression annotation is present every gene
  falls in the single ``all`` expression bin (the table degrades gracefully).
* detection sparsity — read from ``per_gene_detection_fraction`` when present
  (fraction of cells in which the gene is detected); otherwise the single
  ``all`` sparsity bin is used.

Within each (expression-bin, sparsity-bin) cell we report mean PCC / Spearman /
RMSE and the number of genes, pooled across panels. This is the table backbone:
when the real bundle ships ``per_gene_mean_expression`` /
``per_gene_detection_fraction`` the strata auto-populate; the synthetic-smoke
path with neither still renders an ``all/all`` row per method.

Usage::

    python scripts/visualize/table_heldout_stratified.py \
        --results results.json --out table_heldout_stratified.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make _figbase importable

from _figbase import (  # noqa: E402
    iter_ok_records,
    load_results,
    order_methods,
    require_records,
    save_table,
)

# Bin edges (right-closed). A gene with annotation value v lands in the first
# bin whose upper edge it does not exceed.
EXPRESSION_BINS: tuple[tuple[str, float], ...] = (
    ("low", 0.33),
    ("mid", 0.66),
    ("high", float("inf")),
)
SPARSITY_BINS: tuple[tuple[str, float], ...] = (
    ("dense", 0.33),
    ("intermediate", 0.66),
    ("sparse", float("inf")),
)


def _bin_label(value: float | None, bins: tuple[tuple[str, float], ...]) -> str:
    if value is None:
        return "all"
    for label, upper in bins:
        if value <= upper:
            return label
    return bins[-1][0]


def _gene_annotation(metrics: dict[str, Any], key: str, gene: str) -> float | None:
    ann = metrics.get(key)
    if isinstance(ann, dict) and gene in ann:
        v = ann[gene]
        if isinstance(v, (int, float)) and float(v) == float(v):
            return float(v)
    return None


def build_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate per-gene scores into ``(method, expr_bin, sparsity_bin)`` rows."""
    # (method, expr_bin, sparsity_bin) -> {metric: [values]}
    acc: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for _panel, method, record in iter_ok_records(bundle):
        metrics = record.get("metrics", {})
        pcc = metrics.get("per_gene_pearson")
        if not isinstance(pcc, dict):
            continue
        spearman = metrics.get("per_gene_spearman", {})
        rmse = metrics.get("per_gene_rmse", {})
        for gene in pcc:
            expr = _gene_annotation(metrics, "per_gene_mean_expression", gene)
            spars = _gene_annotation(metrics, "per_gene_detection_fraction", gene)
            cell = (method, _bin_label(expr, EXPRESSION_BINS), _bin_label(spars, SPARSITY_BINS))
            bucket = acc.setdefault(cell, {"pcc": [], "spearman": [], "rmse": []})
            for store, src in (("pcc", pcc), ("spearman", spearman), ("rmse", rmse)):
                v = src.get(gene) if isinstance(src, dict) else None
                if isinstance(v, (int, float)) and float(v) == float(v):
                    bucket[store].append(float(v))

    rows: list[dict[str, Any]] = []
    for (method, expr_bin, sparsity_bin), bucket in acc.items():
        n = len(bucket["pcc"])
        if n == 0:
            continue
        rows.append(
            {
                "method": method,
                "expression_bin": expr_bin,
                "sparsity_bin": sparsity_bin,
                "n_genes": n,
                "mean_pcc": _mean(bucket["pcc"]),
                "mean_spearman": _mean(bucket["spearman"]),
                "mean_rmse": _mean(bucket["rmse"]),
            }
        )

    method_rank = {m: i for i, m in enumerate(order_methods(list({r["method"] for r in rows})))}
    expr_rank = {label: i for i, (label, _u) in enumerate(EXPRESSION_BINS)}
    expr_rank["all"] = -1
    spars_rank = {label: i for i, (label, _u) in enumerate(SPARSITY_BINS)}
    spars_rank["all"] = -1
    rows.sort(
        key=lambda r: (
            method_rank.get(r["method"], 99),
            expr_rank.get(r["expression_bin"], 99),
            spars_rank.get(r["sparsity_bin"], 99),
        )
    )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def render_from_bundle(
    results_path: str | Path,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> tuple[Path, Path]:
    bundle = load_results(results_path)
    rows = require_records(
        build_rows(bundle), what="stratified held-out recovery table (no per-gene PCC present)"
    )
    columns = [
        "method",
        "expression_bin",
        "sparsity_bin",
        "n_genes",
        "mean_pcc",
        "mean_spearman",
        "mean_rmse",
    ]
    return save_table(
        rows,
        out_path,
        columns=columns,
        title=title or "Held-out gene recovery stratified by expression & sparsity",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Aggregated benchmark results JSON.")
    parser.add_argument("--out", required=True, help="Output CSV path (markdown written beside it).")
    parser.add_argument("--title", default=None, help="Override the table title.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    csv_path, md_path = render_from_bundle(args.results, args.out, title=args.title)
    print(f"wrote {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
