"""
Machine-readable dataset registry for LuminaST ingestion.

This is the single source of truth that enumerates every spatial-transcriptomics
(and reference scRNA) dataset tracked for LuminaST, mapping each onto the repo's
actual AnnData input contract (see ``AnnDataSchemaValidator.validate_spatial_data``
in ``validation.py`` and ``SpatialTranscriptomicsDataset`` /
``ReferenceAtlasDataset`` in ``datasets.py``).

It complements — does not replace — :class:`CancerRegistry` (which maps cancer
*names* to integer guidance indices). Here we record dataset *provenance*: id,
platform, accession/URL (with verification status), citation key, contract
mapping, and the raw-count policy.

Policy
------
* **Never fabricate URLs.** Where a download hotlink is unconfirmed the spec is
  flagged ``UrlStatus.UNVERIFIED`` and only the canonical landing page is stored.
  The unified fetcher (``scripts/data/fetch_datasets.py``) refuses to download
  such datasets and points at the tracking issue instead.
* **Raw integer counts only.** Every loader must yield raw counts in ``.X``
  (cells/spots x genes); normalized objects, WSIs, FASTQs, and BAMs are rejected.

Canonical source of truth for accessions: ``ST_research/datasets/DATASET_REGISTRY.md``
(dated 2026-05-28/29). Per-dataset evidence lives in the linked GitHub issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class UrlStatus(str, Enum):
    """Verification status of a dataset's download URL."""

    VERIFIED = "verified"          # link returned HTTP 200 / confirmed artifact
    UNVERIFIED = "unverified"      # only the canonical page is known; hotlink unconfirmed


class LoaderKind(str, Enum):
    """How a dataset is materialised into a schema-valid AnnData."""

    HTTP = "http"                  # direct download of a verified artifact URL
    PAGE_BUNDLE = "page_bundle"    # bundle pulled from a 10x dataset "Download" tab
    SQUIDPY = "squidpy"            # squidpy one-liner (sq.datasets.*)
    SCANPY = "scanpy"              # scanpy one-liner (sc.datasets.visium_sge / *)
    HUGGINGFACE = "huggingface"    # huggingface_hub.hf_hub_download
    CENSUS = "census"             # CZ CELLxGENE Census API (cellxgene_census)


@dataclass(frozen=True)
class DatasetSpec:
    """Provenance + contract mapping for one LuminaST-tracked dataset."""

    id: str
    name: str
    issues: Tuple[int, ...]
    platform: str
    tissue: str
    cancer_type: str               # CancerRegistry token (or UNKNOWN fallback)
    accession: str
    url: str                       # canonical landing page OR verified artifact URL
    url_status: UrlStatus
    loader: LoaderKind
    download_artifact: str         # filename / loader call that carries raw counts
    citation_key: str
    raw_count_policy: str
    contract_mapping: str
    tier: str = "A"               # A = registry recommended; B = squidpy/scanpy native
    is_reference: bool = False     # True for scRNA reference atlas (no .obsm['spatial'])
    notes: str = ""

    @property
    def verified(self) -> bool:
        return self.url_status is UrlStatus.VERIFIED


# ---------------------------------------------------------------------------
# Registry — every dataset tracked for LuminaST ingestion.
# Issue numbers reference https://github.com/PeterPonyu/lumina-st/issues/<n>.
# ---------------------------------------------------------------------------
DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "hest1k": DatasetSpec(
        id="hest1k",
        name="HEST-1k (counts)",
        issues=(54, 70),
        platform="Visium v1/v2 + legacy ST",
        tissue="multi-organ H&E + ST (1,229 samples)",
        cancer_type="MIXED",
        accession="HuggingFace MahmoodLab/hest",
        url="https://huggingface.co/datasets/MahmoodLab/hest",
        url_status=UrlStatus.VERIFIED,
        loader=LoaderKind.HUGGINGFACE,
        download_artifact="st/<sample>.h5ad (per-sample raw-count AnnData)",
        citation_key="jaume2024hest",
        raw_count_policy="raw transcript counts already in .X + (x,y) coords; no restore needed.",
        contract_mapping="SpatialTranscriptomicsDataset; per-sample cancer_type varies "
        "(COAD/PRAD confirmed) -> map to CancerRegistry token or UNKNOWN. Also seeds the "
        "ReferenceAtlasDataset pool.",
        tier="A",
        notes="Registry first-pull. Pan-organ pool backs the pan-cancer claim.",
    ),
    "her2st": DatasetSpec(
        id="her2st",
        name="her2st (counts)",
        issues=(55,),
        platform="legacy ST array (NOT Visium)",
        tissue="HER2+ breast (36 sections, 8 patients)",
        cancer_type="BRCA",
        accession="Zenodo 3957257",
        url="https://zenodo.org/records/3957257",
        url_status=UrlStatus.VERIFIED,
        loader=LoaderKind.PAGE_BUNDLE,
        download_artifact="count-matrices.zip (37.2 MB; .tsv.gz per section) from the Zenodo record",
        citation_key="andersson2021her2st",
        raw_count_policy="integer [n_spots]x[n_genes] count matrices; build AnnData per section.",
        contract_mapping="SpatialTranscriptomicsDataset; array coords -> .obsm['spatial']; "
        ".obs['cancer_type']='BRCA'.",
        tier="A",
        notes="Smallest cancer ST -> fast smoke target for held-out-gene recovery.",
    ),
    "visiumhd_crc": DatasetSpec(
        id="visiumhd_crc",
        name="Visium HD CRC",
        issues=(56,),
        platform="Visium HD (8 um bins)",
        tissue="colorectal cancer",
        cancer_type="COAD",
        accession="10x Genomics dataset page",
        url="https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc",
        url_status=UrlStatus.VERIFIED,
        loader=LoaderKind.PAGE_BUNDLE,
        download_artifact="Visium_HD_Human_Colon_Cancer_binned_outputs.tar.gz (15.9 GB) "
        "-> square_008um/filtered_feature_bc_matrix.h5",
        citation_key="10x_visiumhd_crc",
        raw_count_policy="raw UMIs in square_008um/filtered_feature_bc_matrix.h5.",
        contract_mapping="SpatialTranscriptomicsDataset @ 8 um bins; .obs['cancer_type']='COAD'.",
        tier="A",
        notes="Flagship high-resolution COAD target. Pull binned_outputs.tar.gz from the "
        "page; do NOT hardcode a guessed cf.10xgenomics.com h5 hotlink (caveat #1).",
    ),
    "visiumhd_mousebrain": DatasetSpec(
        id="visiumhd_mousebrain",
        name="Visium HD Mouse Brain",
        issues=(57,),
        platform="Visium HD (8 um bins)",
        tissue="mouse brain (H&E)",
        cancer_type="UNKNOWN",
        accession="10x Genomics dataset page",
        url="https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-brain-he",
        url_status=UrlStatus.VERIFIED,
        loader=LoaderKind.PAGE_BUNDLE,
        download_artifact="Visium_HD_Mouse_Brain_binned_outputs.tar.gz (4.6 GB) "
        "-> square_008um/filtered_feature_bc_matrix.h5",
        citation_key="10x_visiumhd_mousebrain",
        raw_count_policy="raw UMIs in square_008um/filtered_feature_bc_matrix.h5.",
        contract_mapping="SpatialTranscriptomicsDataset @ 8 um bins; .obs['cancer_type']='UNKNOWN'.",
        tier="A",
        notes="Dense non-cancer HD slice -> cross-tissue enhancement stress test.",
    ),
}


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------
def all_ids() -> List[str]:
    """All registered dataset ids, in declaration order."""
    return list(DATASET_REGISTRY.keys())


def get(dataset_id: str) -> DatasetSpec:
    """Return the spec for ``dataset_id`` or raise ``KeyError`` with the valid ids."""
    if dataset_id not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset id {dataset_id!r}. Known ids: {', '.join(all_ids())}"
        )
    return DATASET_REGISTRY[dataset_id]


def verified_specs() -> List[DatasetSpec]:
    """Specs whose download URL is verified (safe to fetch)."""
    return [s for s in DATASET_REGISTRY.values() if s.verified]


def unverified_specs() -> List[DatasetSpec]:
    """Specs with an UNVERIFIED hotlink — never auto-download these."""
    return [s for s in DATASET_REGISTRY.values() if not s.verified]


def issues_covered() -> List[int]:
    """Sorted, de-duplicated list of every GitHub issue referenced by the registry."""
    seen: set[int] = set()
    for spec in DATASET_REGISTRY.values():
        seen.update(spec.issues)
    return sorted(seen)
