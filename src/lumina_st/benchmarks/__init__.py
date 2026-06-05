"""Benchmark contract and adapters for LuminaST.

This package implements the shared adapter contract defined in
`contract.py`. Every external imputation method
(SpaIM, TISSUE, CellT, stMCDI, ...) and every internal baseline (mean, kNN,
LuminaST itself) must subclass BaseAdapter and honor the same input/output
schema so they can be run from one runner and compared on shared metrics.
"""

from .contract import (
    AdapterInput,
    AdapterResult,
    BaseAdapter,
    Provenance,
    compute_imputation_metrics,
)
from .panels import REGISTERED_PANELS, MarkerPanel, get_panel
from .runner import aggregate_results, run_panel, write_results_json
from .sparsity import (
    SparsityPoint,
    aggregate_sparsity_results,
    downsample_detection_rate,
    run_sparsity_sweep,
)
from .uncertainty import ConformalCalibrator
from .cross_validation import (
    CVFold,
    aggregate_cv_results,
    bootstrap_ci,
    run_cv,
    stateless_factory,
    summarize_cv,
)
from .foundation_provenance import (
    FOUNDATION_BACKENDS,
    FoundationRerunError,
    assert_not_claimed_rerun,
    is_reference_reported,
    mark_reference_reported,
    require_reference_reported,
)

__all__ = [
    "AdapterInput",
    "AdapterResult",
    "BaseAdapter",
    "Provenance",
    "compute_imputation_metrics",
    "MarkerPanel",
    "REGISTERED_PANELS",
    "get_panel",
    "run_panel",
    "aggregate_results",
    "write_results_json",
    "SparsityPoint",
    "downsample_detection_rate",
    "run_sparsity_sweep",
    "aggregate_sparsity_results",
    "ConformalCalibrator",
    "CVFold",
    "run_cv",
    "summarize_cv",
    "bootstrap_ci",
    "stateless_factory",
    "aggregate_cv_results",
    "FOUNDATION_BACKENDS",
    "FoundationRerunError",
    "mark_reference_reported",
    "is_reference_reported",
    "require_reference_reported",
    "assert_not_claimed_rerun",
]
