"""Benchmark runner: drive a set of adapters over one or more marker panels and
emit a results JSON.

The output JSON schema is keyed by (dataset, panel, method) and includes per-
gene metrics + adapter provenance, so downstream figure scripts read it
without re-running anything. Schema is documented in
`benchmark_contracts/luminast_external_methods.json` and asserted by
`tests/benchmarks/test_runner.py`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import anndata as ad

from .contract import AdapterInput, AdapterResult, BaseAdapter
from .panels import MarkerPanel


def run_panel(
    adapters: Sequence[BaseAdapter],
    adata: ad.AnnData,
    panel: MarkerPanel,
    seed: int = 0,
    dataset_name: str = "unknown",
    cancer_type: str | None = None,
    observed_layer: str | None = None,
    truth_layer: str | None = None,
) -> list[AdapterResult]:
    """Run every adapter on the same held-out marker panel and return results."""
    inp = AdapterInput(
        input_h5ad=adata,
        held_out_genes=list(panel.genes),
        observed_layer=observed_layer,
        truth_layer=truth_layer,
        seed=seed,
        cancer_type=cancer_type,
        extra={"dataset": dataset_name, "panel": panel.name},
    )
    return [adapter.run(inp) for adapter in adapters]


def aggregate_results(
    results_by_panel: dict[tuple[str, str], list[AdapterResult]],
) -> dict:
    """Aggregate adapter results into a JSON-serializable nested dict.

    Returns:
        {
          "panels": {
            "<dataset>/<panel>": {
              "<method>": {
                "status": "ok" | "unavailable:...",
                "runtime_s": ..., "metrics": {...}, "provenance": {...}
              }, ...
            }, ...
          },
          "schema_version": "1"
        }
    """
    out: dict = {"schema_version": "1", "panels": {}}
    for (dataset, panel_name), results in results_by_panel.items():
        key = f"{dataset}/{panel_name}"
        out["panels"][key] = {}
        for r in results:
            out["panels"][key][r.method] = {
                "status": r.status,
                "runtime_s": r.runtime_s,
                "metrics": r.metrics_json,
                "provenance": asdict(r.provenance),
            }
    return out


def write_results_json(aggregated: dict, output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(aggregated, indent=2))
    return p
