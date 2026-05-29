# LuminaST — Dataset Literature Links

Source paper / portal for every dataset enumerated in
[`docs/DATASETS.md`](../docs/DATASETS.md) and the machine-readable
[`src/lumina_st/data/dataset_registry.py`](../src/lumina_st/data/dataset_registry.py).
`citation_key` matches the `DatasetSpec.citation_key` field in that registry.

> Canonical source of truth for accessions: `ST_research/datasets/DATASET_REGISTRY.md`
> (2026-05-28/29). Where a download hotlink is unconfirmed the link below is the
> **canonical landing page**, carrying the same `⚠️ UNVERIFIED` flag as the registry —
> never a guessed hotlink.

| `citation_key` | Dataset | Issue(s) | Source paper / portal |
|----------------|---------|----------|-----------------------|
| `jaume2024hest` | HEST-1k | #54, #70 | Jaume et al., *HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis*, NeurIPS 2024 — [arXiv:2406.16192](https://arxiv.org/abs/2406.16192); data: [huggingface.co/datasets/MahmoodLab/hest](https://huggingface.co/datasets/MahmoodLab/hest) |
| `andersson2021her2st` | her2st | #55 | Andersson et al., *Spatial deconvolution of HER2-positive breast cancer delineates tumor-associated cell type interactions*, Nat. Commun. 2021 — [doi:10.1038/s41467-021-26271-2](https://doi.org/10.1038/s41467-021-26271-2); data: [Zenodo 3957257](https://zenodo.org/records/3957257) |
| `10x_visiumhd_crc` | Visium HD CRC | #56 | 10x Genomics dataset — [Visium HD CytAssist, Human CRC](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc) |
| `10x_visiumhd_mousebrain` | Visium HD Mouse Brain | #57 | 10x Genomics dataset — [Visium HD CytAssist, Mouse Brain (H&E)](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-brain-he) |

## Related infrastructure

- **#71** — *publish/register real LuminaST checkpoints*: the checkpoint download CLI
  (`src/lumina_st/cli/download.py`) is infra-adjacent to dataset ingestion; its
  fabricated `spatial-omics/lumina-st` HF URLs are flagged in `docs/DATASETS.md`
  §"UNVERIFIED-URL caveats" (caveat #3) and tracked separately.
- **#182** — meta-issue requesting these 05-29 cards/fetchers/literature links be
  consolidated into draft PR #61 (this PR).
