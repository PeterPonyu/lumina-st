"""LuminaST data subpackage: registries, datasets, and schema validation."""

from __future__ import annotations

from .cancer_registry import CancerRegistry
from .count_type import (
    CountType,
    assert_raw_counts,
    infer_count_type,
    is_raw_counts,
)
from .dataset_registry import (
    DATASET_REGISTRY,
    DatasetSpec,
    LoaderKind,
    UrlStatus,
    all_ids,
    get,
    issues_covered,
    unverified_specs,
    verified_specs,
)
from .validation import AnnDataSchemaValidator

__all__ = [
    "CancerRegistry",
    "CountType",
    "assert_raw_counts",
    "infer_count_type",
    "is_raw_counts",
    "DATASET_REGISTRY",
    "DatasetSpec",
    "LoaderKind",
    "UrlStatus",
    "all_ids",
    "get",
    "issues_covered",
    "unverified_specs",
    "verified_specs",
    "AnnDataSchemaValidator",
]
