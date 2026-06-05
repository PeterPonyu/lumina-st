"""Benchmark runner: drive a set of adapters over one or more marker panels and
emit a results JSON.

The output JSON schema is keyed by (dataset, panel, method) and includes per-
gene metrics + adapter provenance, so downstream figure scripts read it
without re-running anything. The schema is defined by the dataclasses in
`contract.py` and asserted by `tests/benchmarks/test_runner.py`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import anndata as ad

from .contract import (
    AdapterInput,
    AdapterResult,
    BaseAdapter,
    ProtocolParityError,
    TaskBoundaryError,
    TaskType,
)
from .panels import MarkerPanel


def enforce_protocol_parity(
    adapters: Sequence[BaseAdapter],
    inp: AdapterInput,
) -> None:
    """Centrally enforce the #307/#309 protocol gates before any adapter runs.

    Parity is enforced HERE, in the runner/contract layer, not per-adapter, so
    every adapter is provably compared under one held-out protocol:

    * Held-out gene protocol parity (#307): every adapter is scored on the SAME
      ``AdapterInput`` (same held-out split, same scoring genes, same masking),
      so the protocol signature is identical by construction. We assert the
      signature is well-formed (non-empty held-out set) so an empty "score every
      gene" run cannot masquerade as a held-out comparison.
    * Encoder-leakage guard (#307): the shared input is checked once — if a
      held-out gene could structurally leak into the encoder, no adapter runs.
    * Task-boundary separation (#309): ``run_panel`` is the gene-recovery runner,
      so the input and every adapter must declare ``GENE_RECOVERY``; a denoising
      / pathway-aggregate adapter is rejected up front instead of being silently
      scored as recovery.
    """
    if inp.task_type != TaskType.GENE_RECOVERY:
        raise TaskBoundaryError(
            f"run_panel is the gene-recovery runner but the input is tagged "
            f"{inp.task_type.value!r}. Denoising / pathway-aggregate tasks need their "
            "own runner+scorer; they cannot be reported as gene recovery (#309)."
        )
    if not inp.held_out_genes:
        raise ProtocolParityError(
            "run_panel requires a non-empty held-out gene set so every adapter is "
            "compared on an identical held-out split (#307)."
        )

    # Encoder-leakage guard on the shared input (#307) — fail-closed for all.
    inp.assert_no_encoder_leakage()

    mismatched = [a.name for a in adapters if a.task_type != TaskType.GENE_RECOVERY]
    if mismatched:
        raise TaskBoundaryError(
            "These adapters do not produce gene-recovery results and cannot be run by "
            f"run_panel: {mismatched}. Their results would be scored under the wrong "
            "task (#309)."
        )


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
    """Run every adapter on the same held-out marker panel and return results.

    All adapters are handed the SAME ``AdapterInput`` and the #307/#309 gates are
    enforced centrally via :func:`enforce_protocol_parity` before any adapter
    runs, so the comparison is provably under one held-out protocol.
    """
    inp = AdapterInput(
        input_h5ad=adata,
        held_out_genes=list(panel.genes),
        observed_layer=observed_layer,
        truth_layer=truth_layer,
        seed=seed,
        cancer_type=cancer_type,
        extra={"dataset": dataset_name, "panel": panel.name},
    )
    enforce_protocol_parity(adapters, inp)
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
