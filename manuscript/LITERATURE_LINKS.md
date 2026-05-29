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
| `10x_visiumhd_tonsil` | Visium HD Tonsil | #58 | 10x Genomics dataset — [Visium HD CytAssist, Human Tonsil (FF)](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-tonsil-fresh-frozen) |
| `10x_xenium_skin` | Xenium Human Skin (377-gene) | #59 | 10x Genomics dataset — [Human Skin, Xenium Multi-Tissue & Cancer Panel](https://www.10xgenomics.com/datasets/human-skin-data-xenium-human-multi-tissue-and-cancer-panel) · ⚠️ UNVERIFIED hotlink (page link 404s; correct page adds `-1-standard`) |
| `janesick2023xenium` | Xenium Breast Cancer (Janesick) | #60 | Janesick et al., *High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis*, Nat. Commun. 2023 — [doi:10.1038/s41467-023-43458-x](https://doi.org/10.1038/s41467-023-43458-x); demo: 10x `Xenium_V1_human_Breast` |
| `maynard2021dlpfc` | DLPFC 12-sample Visium (spatialLIBD) | #62 | Maynard et al., *Transcriptome-scale spatial gene expression in the human dorsolateral prefrontal cortex*, Nat. Neurosci. 2021 — [doi:10.1038/s41593-020-00787-0](https://doi.org/10.1038/s41593-020-00787-0); pkg: [spatialLIBD](https://research.libd.org/spatialLIBD/) |
| `10x_cytassist_ffpe_breast` | Visium CytAssist FFPE Breast | #63 | 10x Genomics dataset — [Human Breast Cancer, CytAssist FFPE (2-standard)](https://www.10xgenomics.com/datasets/gene-and-protein-expression-library-of-human-breast-cancer-cytassist-ffpe-2-standard) |
| `10x_visiumhd_breast` | Visium HD Breast Cancer FFPE | #64 | 10x Genomics dataset — [Visium HD, Human Breast Cancer FFPE (IF)](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-breast-cancer-ffpe-if) |
| `10x_xenium_prime_breast` | Xenium Prime 5K Breast | #65, #184 | 10x Genomics dataset — [Xenium Prime FFPE Human Breast Cancer](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-breast-cancer) · ⚠️ UNVERIFIED Output-Bundle hotlink |
| `10x_xenium_prime_skin` | Xenium Prime 5K Skin / Melanoma | #66, #184 | 10x Genomics dataset (PREVIEW) — [Xenium Prime FFPE Human Skin](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-skin) · ⚠️ UNVERIFIED hotlink + Preview Data caveat |

## Related infrastructure

- **#71** — *publish/register real LuminaST checkpoints*: the checkpoint download CLI
  (`src/lumina_st/cli/download.py`) is infra-adjacent to dataset ingestion; its
  fabricated `spatial-omics/lumina-st` HF URLs are flagged in `docs/DATASETS.md`
  §"UNVERIFIED-URL caveats" (caveat #3) and tracked separately.
- **#182** — meta-issue requesting these 05-29 cards/fetchers/literature links be
  consolidated into draft PR #61 (this PR).
