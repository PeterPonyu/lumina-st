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
    "visiumhd_tonsil": DatasetSpec(
        id="visiumhd_tonsil",
        name="Visium HD Tonsil",
        issues=(58,),
        platform="Visium HD (8 um bins)",
        tissue="tonsil (fresh frozen)",
        cancer_type="UNKNOWN",
        accession="10x Genomics dataset page",
        url="https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-tonsil-fresh-frozen",
        url_status=UrlStatus.VERIFIED,
        loader=LoaderKind.PAGE_BUNDLE,
        download_artifact="Visium_HD_Human_Tonsil_Fresh_Frozen_binned_outputs.tar.gz (17.4 GB) "
        "-> square_008um/filtered_feature_bc_matrix.h5",
        citation_key="10x_visiumhd_tonsil",
        raw_count_policy="raw UMIs in square_008um/filtered_feature_bc_matrix.h5.",
        contract_mapping="SpatialTranscriptomicsDataset @ 8 um bins; .obs['cancer_type']='UNKNOWN'.",
        tier="A",
        notes="Immune-rich HD slice. Registry size estimate (2-4 GB) is far too low (17.4 GB).",
    ),
    "xenium_skin": DatasetSpec(
        id="xenium_skin",
        name="Xenium Human Skin (377-gene panel)",
        issues=(59,),
        platform="Xenium",
        tissue="skin (multi-tissue + cancer panel; non-diseased)",
        cancer_type="UNKNOWN",
        accession="10x Genomics dataset page (377-gene panel)",
        url="https://www.10xgenomics.com/datasets/human-skin-data-xenium-human-multi-tissue-and-cancer-panel",
        url_status=UrlStatus.UNVERIFIED,
        loader=LoaderKind.PAGE_BUNDLE,
        download_artifact="cell_feature_matrix.h5 inside ..._outs.zip (raw per-cell counts)",
        citation_key="10x_xenium_skin",
        raw_count_policy="raw per-cell counts in cell_feature_matrix.h5; drop control/negative probes.",
        contract_mapping="SpatialTranscriptomicsDataset; per-cell centroids -> .obsm['spatial']; "
        ".obs['cancer_type']='UNKNOWN' (sample is non-diseased skin).",
        tier="A",
        notes="UNVERIFIED hotlink (caveat #2): the registry page link 404s; the correct page "
        "adds the '-1-standard' suffix. Resolve from the Download tab. Upgraded by #66.",
    ),
    "xenium_breast": DatasetSpec(
        id="xenium_breast",
        name="Xenium Breast Cancer (Janesick, 313-gene panel)",
        issues=(60,),
        platform="Xenium",
        tissue="breast cancer",
        cancer_type="BRCA",
        accession="10x demo Xenium_V1_human_Breast (313-gene panel)",
        url="https://www.10xgenomics.com/datasets/ffpe-human-breast-using-the-entire-sample-area-1-standard",
        url_status=UrlStatus.VERIFIED,
        loader=LoaderKind.PAGE_BUNDLE,
        download_artifact="Xenium_V1_human_Breast outputs "
        "(Xenium_FFPE_Human_Breast_Cancer_Rep1_cell_feature_matrix.h5, 12 MB; full outs ~9.86 GB)",
        citation_key="janesick2023xenium",
        raw_count_policy="raw per-cell counts in cell_feature_matrix.h5; drop control/negative probes.",
        contract_mapping="SpatialTranscriptomicsDataset; per-cell centroids -> .obsm['spatial']; "
        ".obs['cancer_type']='BRCA'.",
        tier="A",
        notes="Canonical Xenium breast demo. Upgraded to 5K panel by #65.",
    ),
    "dlpfc_maynard2021": DatasetSpec(
        id="dlpfc_maynard2021",
        name="DLPFC 12-sample Visium (Maynard 2021 / spatialLIBD)",
        issues=(62,),
        platform="Visium v1 (spot)",
        tissue="human DLPFC (non-diseased brain); manual L1-L6 + WM labels",
        cancer_type="UNKNOWN",
        accession="Maynard 2021 Nat. Neurosci. (DOI 10.1038/s41593-020-00787-0); spatialLIBD",
        url="https://research.libd.org/spatialLIBD/",
        url_status=UrlStatus.VERIFIED,
        loader=LoaderKind.PAGE_BUNDLE,
        download_artifact="spatialLIBD::fetch_data('spe') -> SpatialExperiment (counts assay = "
        "raw integer UMIs); convert SPE->AnnData via zellkonverter/anndata2ri.",
        citation_key="maynard2021dlpfc",
        raw_count_policy="counts assay = raw integer UMIs; use it, not logcounts.",
        contract_mapping="SpatialTranscriptomicsDataset; spatialCoords()->.obsm['spatial']; "
        "manual layer label in .obs; .obs['cancer_type']='UNKNOWN'.",
        tier="A",
        notes="Gold-standard manual-layer imputation benchmark (layer-ARI, held-out-gene recovery).",
    ),
    "cytassist_ffpe_breast": DatasetSpec(
        id="cytassist_ffpe_breast",
        name="Visium CytAssist FFPE Human Breast Cancer",
        issues=(63,),
        platform="Visium CytAssist (FFPE, spot)",
        tissue="breast cancer",
        cancer_type="BRCA",
        accession="10x CytAssist_FFPE_Human_Breast_Cancer (Space Ranger v2.0.0)",
        url="https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5",
        url_status=UrlStatus.VERIFIED,
        loader=LoaderKind.HTTP,
        download_artifact="CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5 "
        "(33 MB, HTTP 200) + ..._spatial.tar.gz (34.5 MB)",
        citation_key="10x_cytassist_ffpe_breast",
        raw_count_policy="raw integer UMI feature-barcode matrix; histology via _spatial.tar.gz.",
        contract_mapping="SpatialTranscriptomicsDataset; coords from _spatial.tar.gz -> "
        ".obsm['spatial']; .obs['cancer_type']='BRCA'.",
        tier="A",
        notes="Lowest-friction Visium breast target (direct .h5, no bin-unpacking).",
    ),
    "visiumhd_breast": DatasetSpec(
        id="visiumhd_breast",
        name="Visium HD Human Breast Cancer (FFPE, IF)",
        issues=(64,),
        platform="Visium HD (8 um bins)",
        tissue="breast cancer (FFPE)",
        cancer_type="BRCA",
        accession="10x Visium_HD_Human_Breast_Cancer_FFPE (Space Ranger v3.1.2)",
        url="https://cf.10xgenomics.com/samples/spatial-exp/3.1.2/Visium_HD_Human_Breast_Cancer_FFPE/Visium_HD_Human_Breast_Cancer_FFPE_binned_outputs.tar.gz",
        url_status=UrlStatus.VERIFIED,
        loader=LoaderKind.HTTP,
        download_artifact="Visium_HD_Human_Breast_Cancer_FFPE_binned_outputs.tar.gz (6.75 GB) "
        "-> binned_outputs/square_008um/filtered_feature_bc_matrix.h5",
        citation_key="10x_visiumhd_breast",
        raw_count_policy="raw integer UMIs in square_008um/filtered_feature_bc_matrix.h5.",
        contract_mapping="SpatialTranscriptomicsDataset @ 8 um bins; .obs['cancer_type']='BRCA'.",
        tier="A",
        notes="Same-tissue multi-resolution breast (legacy ST <-> Visium <-> HD <-> Xenium). "
        "Use binned_outputs.tar.gz, not a guessed h5 hotlink.",
    ),
    "xenium_prime_breast": DatasetSpec(
        id="xenium_prime_breast",
        name="Xenium Prime 5K Human Breast Cancer (~5,100-gene panel)",
        issues=(65, 184),
        platform="Xenium Prime (single-cell)",
        tissue="breast cancer (FFPE; 699,110 cells)",
        cancer_type="BRCA",
        accession="10x Xenium Prime FFPE Human Breast Cancer (2024-10-24, XOA v3.x)",
        url="https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-breast-cancer",
        url_status=UrlStatus.UNVERIFIED,
        loader=LoaderKind.PAGE_BUNDLE,
        download_artifact="cell_feature_matrix.zarr.zip (subset) / cell_feature_matrix.h5 "
        "(full Output Bundle .zip) — confirm filename on Download tab",
        citation_key="10x_xenium_prime_breast",
        raw_count_policy="raw per-cell integer counts; keep genes only (drop control/negative "
        "probes); read via spatialdata_io.xenium() or scanpy on cell_feature_matrix.h5.",
        contract_mapping="SpatialTranscriptomicsDataset; (x_centroid,y_centroid)->.obsm['spatial']; "
        ".obs['cancer_type']='BRCA'.",
        tier="A",
        notes="UNVERIFIED Output-Bundle hotlink — pull from Download tab. ~16x panel vs #60. "
        "Part of the Prime 5K cohort (#184).",
    ),
    "xenium_prime_skin_melanoma": DatasetSpec(
        id="xenium_prime_skin_melanoma",
        name="Xenium Prime 5K Human Skin / Dermal Melanoma (5,006-gene panel)",
        issues=(66, 184),
        platform="Xenium Prime (single-cell)",
        tissue="primary dermal melanoma (FFPE; 112,551 cells)",
        cancer_type="SKCM",
        accession="10x Xenium Prime FFPE Human Skin (2024-07-31, PREVIEW DATA)",
        url="https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-skin",
        url_status=UrlStatus.UNVERIFIED,
        loader=LoaderKind.PAGE_BUNDLE,
        download_artifact="cell_feature_matrix.zarr.zip (subset) / cell_feature_matrix.h5 "
        "(full Output Bundle .zip) — confirm filename on Download tab",
        citation_key="10x_xenium_prime_skin",
        raw_count_policy="raw per-cell integer counts; keep genes only (5 genes excluded in dev "
        "panel: ADGRG1, NMBR, OSMR, OSTF1, RGS8). Verify Preview bundle completeness first.",
        contract_mapping="SpatialTranscriptomicsDataset; (x_centroid,y_centroid)->.obsm['spatial']; "
        ".obs['cancer_type']='SKCM' (requires registry config; UNKNOWN fallback otherwise).",
        tier="A",
        notes="PREVIEW DATA + UNVERIFIED hotlink. Real melanoma upgrade of non-diseased #59. "
        "Part of the Prime 5K cohort (#184). SKCM is not in default_pan_cancer.",
    ),
    "xenium_prime_ovarian": DatasetSpec(
        id="xenium_prime_ovarian",
        name="Xenium Prime 5K Human Ovarian Cancer (~5,100-gene panel)",
        issues=(67, 184),
        platform="Xenium Prime (single-cell, XOA v3.2)",
        tissue="ovarian cancer (FFPE; 407,124 cells; pathologist-annotated H&E)",
        cancer_type="OV",
        accession="10x Xenium Prime FFPE Human Ovarian Cancer (2024-12-17, XOA v3.2)",
        url="https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-ovarian-cancer",
        url_status=UrlStatus.UNVERIFIED,
        loader=LoaderKind.PAGE_BUNDLE,
        download_artifact="cell_feature_matrix.zarr.zip (subset) / cell_feature_matrix.h5 "
        "(full Output Bundle .zip) — confirm filename on Download tab",
        citation_key="10x_xenium_prime_ovarian",
        raw_count_policy="raw per-cell integer counts; keep genes only (drop control/negative probes).",
        contract_mapping="SpatialTranscriptomicsDataset; (x_centroid,y_centroid)->.obsm['spatial']; "
        ".obs['cancer_type']='OV'; optional region label from annotated H&E in .obs.",
        tier="A",
        notes="UNVERIFIED Output-Bundle hotlink — pull from Download tab. First ovarian-cancer "
        "slice in the set; ships pathology annotations. Part of the Prime 5K cohort (#184).",
    ),
    "slideseqv2_cerebellum": DatasetSpec(
        id="slideseqv2_cerebellum",
        name="Slide-seqV2 cerebellum (B3)",
        issues=(68,),
        platform="Slide-seqV2 (bead array)",
        tissue="mouse cerebellum (41,786 beads x 4,000 genes)",
        cancer_type="UNKNOWN",
        accession="squidpy figshare mirror 28242783",
        url="https://squidpy.readthedocs.io/en/stable/api/squidpy.datasets.slideseqv2.html",
        url_status=UrlStatus.VERIFIED,
        loader=LoaderKind.SQUIDPY,
        download_artifact="squidpy.datasets.slideseqv2() -> AnnData (raw UMI in .X)",
        citation_key="stickels2021slideseqv2",
        raw_count_policy="raw UMI already in .X; no restore needed.",
        contract_mapping="SpatialTranscriptomicsDataset; bead xy -> .obsm['spatial']; "
        ".obs['cancer_type']='UNKNOWN' (sparsity stress test).",
        tier="B",
        notes="Primary sparse raw-count imputation target. Loader verified in dl env "
        "(squidpy 1.6.5). Shared cache with factorgraph-st / niche-lens-st.",
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
