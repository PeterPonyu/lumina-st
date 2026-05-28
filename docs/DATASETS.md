# LuminaST — Dataset Integration Guide

> Source of truth: the canonical, audit-corrected
> [`ST_research/datasets/DATASET_REGISTRY.md`](../../../datasets/DATASET_REGISTRY.md)
> (dated 2026-05-28). All figures, accessions, and links below are taken from that
> registry — **not** from earlier uncorrected reports. Where the registry flags a
> URL as `⚠️ UNVERIFIED`, this guide carries the same flag and points at the
> canonical landing page instead of a guessed hotlink.

## Method recap (why these datasets)

LuminaST performs **latent enhancement + guided imputation on single spatial
transcriptomics (ST) slices**. Given one schema-valid slice it (1) encodes observed
genes to a latent embedding, (2) runs conditional flow matching with cancer-type
classifier-free guidance, and (3) decodes back with sparsity-aware post-processing —
returning an AnnData carrying `.obsm['latent_enhanced']` and `.layers['imputed']`
(see `LuminaImputer.enhance()` in `src/lumina_st/core/lumina_imputer.py`). Because the
method operates per-slice and benefits from dense measurements and histology pairing,
it prefers **high-resolution, histology-paired data delivered as raw integer counts in
`.h5ad`** (no normalized objects, WSIs, FASTQs, or BAMs).

## Input contract (what a loader must produce)

A LuminaST-ready slice is an AnnData that passes
`AnnDataSchemaValidator.validate_spatial_data` (`src/lumina_st/data/validation.py`):

| Field | Requirement |
|-------|-------------|
| `.X` | raw integer counts (cells/spots × genes); not normalized. |
| `.obsm['spatial']` | 2-D `np.ndarray`, ≥ 2 columns (x, y). **Required.** |
| `.obs['cancer_type']` | optional but recommended; maps to a `CancerRegistry` index for guidance. |
| `.var['impute_mask']` | optional boolean mask of genes to impute (defaults to all). |

These objects feed `SpatialTranscriptomicsDataset` (`src/lumina_st/data/datasets.py`)
and the high-level `LuminaImputer.enhance(st_adata, cancer_type=...)` API. The runnable
end-to-end reference today is `scripts/e2e/enhance_real_st.py`
(`--reference <atlas.h5ad> --target <slice.h5ad> --cancer <TYPE>`); the package CLI
entry point is `python -m lumina_st.cli run-enhance --target <slice.h5ad>`.

> **Planned ingestion surface:** a Python-native `scripts/data/fetch_datasets.py`
> (squidpy/scanpy one-line loaders + documented external downloads; `--list`/`--dry-run`;
> never fabricate URLs or write placeholder bytes) plus optional per-dataset cards under
> `docs/data_cards/<id>.yaml`. The repo `.gitignore` ignores all of `data/`, so raw
> downloads stay local and only the doc/cards are tracked. Each row below has a tracking
> issue (see the PR checklist) following this contract.

## Recommended datasets

From the registry's per-repo plan for **lumina-st** (high-resolution + histology-paired,
single-slice raw counts). Legend: ✅ verified link · ⚠️ UNVERIFIED (canonical page given,
guessed hotlink not confirmed).

| # | Dataset | Accession / ID | Platform | Tissue / Disease | Raw-count size | Source | How it plugs into LuminaST |
|---|---------|----------------|----------|------------------|----------------|--------|----------------------------|
| 1 | HEST-1k (counts) | HF `MahmoodLab/hest` (1,229 samples) | Visium v1/v2 + legacy ST | multi-organ H&E + ST | 8–12 GB (`st/*.h5ad`) | ✅ [huggingface.co/datasets/MahmoodLab/hest](https://huggingface.co/datasets/MahmoodLab/hest) | Each `st/*.h5ad` is already a per-slice raw-count + spatial object → direct enhancement targets; the pan-organ pool also seeds the reference atlas for the latent encoder. |
| 2 | her2st (counts) | Zenodo **3957257** | legacy ST array (not Visium) | HER2+ breast | 37.2 MB (36 sections) | ✅ [zenodo.org/records/3957257](https://zenodo.org/records/3957257) (`count-matrices.zip`) | Small, sparse breast-cancer slices → ideal fast smoke target for guided imputation + held-out-gene recovery benchmarks. |
| 3 | Visium HD CRC | 10x dataset page | Visium HD (8 µm bins) | colorectal cancer | 4–10 GB (`binned_outputs.tar.gz`) | ✅ [10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc) | Flagship high-resolution COAD target (matches the `COAD` registry token); dense 8 µm bins exercise latent enhancement at near-single-cell resolution. |
| 4 | Visium HD Mouse Brain | 10x dataset page | Visium HD (8 µm) | mouse brain (H&E) | 4–7 GB | ✅ [10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-brain-he](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-brain-he) | Dense non-cancer HD slice → cross-tissue enhancement stress test (maps to `UNKNOWN` guidance token). |
| 5 | Visium HD Tonsil | 10x dataset page | Visium HD (8 µm) | tonsil (fresh frozen) | 2–4 GB | ✅ [10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-tonsil-fresh-frozen](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-tonsil-fresh-frozen) | Immune-rich HD slice; smallest HD bundle → fastest HD ingestion smoke. |
| 9 | Xenium Human Skin | 10x dataset page (377-gene panel) | Xenium | skin (multi-tissue + cancer panel) | 0.5–3 GB | ⚠️ **UNVERIFIED** hotlink — page: [10xgenomics.com/datasets/human-skin-data-xenium-human-multi-tissue-and-cancer-panel](https://www.10xgenomics.com/datasets/human-skin-data-xenium-human-multi-tissue-and-cancer-panel) | Single-cell-resolution targeted panel → imputation from a sparse 377-gene panel; histology-paired (LuminaST's preferred regime). |
| 11 | Xenium Breast Cancer (Janesick) | 10x demo `Xenium_V1_human_Breast` (313-gene panel) | Xenium | breast cancer | 0.4–1.5 GB (full outs 8–9 GB) | ✅ 10x `Xenium_V1_human_Breast` demo | Canonical Xenium breast demo → panel imputation and cross-platform (Visium ↔ Xenium) enhancement experiments. |

**Suggested first-pull order (registry):** HEST-1k → 1 Visium HD → 1 Xenium. Total
network ≈ 12–20 GB.

## Local resources

- **Canonical dataset registry (source of truth):**
  `/home/zeyufu/Desktop/ST_research/datasets/DATASET_REGISTRY.md`
- **Verified papers (8 PDFs + indices):** `/home/zeyufu/Desktop/ST_research/references/`
  — includes `DATASET_CATALOG.md` (network-verified loader calls in the `dl` conda env)
  and `ST_omics_DL_Papers_Index_2025_2026.md`.
- **Corrected provenance + download commands:** `st_dataset_provenance_and_policy.md`
  (gemini brain dir) — concrete `curl` commands; 3 URLs flagged `⚠️ UNVERIFIED`.
  *(Not under `ST_research/` on this checkout; cite by canonical name.)*
- **Suggested local cache path:** `~/Desktop/ST_research/data_cache/raw/<dataset_slug>/`
  (raw counts + spatial metadata only). The repo `.gitignore` ignores all of `data/`.
- **Audit trail:** `/home/zeyufu/Desktop/ST_research/audits/` (`findings_*.md` + audit summary).

## ⚠️ UNVERIFIED-URL caveats (resolve before wiring a downloader)

1. **Visium HD CRC (#3):** do **not** hardcode a guessed
   `cf.10xgenomics.com/...filtered_feature_bc_matrix.h5`; pull `binned_outputs.tar.gz`
   from the dataset page instead.
2. **Xenium Human Skin (#9):** the guessed
   `cf.10xgenomics.com/samples/xenium/2.0.0/xenium_human_skin/...` hotlink is
   unconfirmed → obtain the output bundle from the skin dataset page.
3. **Repo integrity — `src/lumina_st/cli/download.py`:** the VAE/diffusion checkpoint
   URLs point at `huggingface.co/datasets/spatial-omics/lumina-st/...`; per the verified
   catalog that HF org/dataset **does not exist** and no checkpoints have been published
   for this package yet. Treat those URLs as fabricated placeholders — do not rely on
   them for data; the fix is tracked separately from dataset ingestion.

## Ingestion roadmap

1. **HEST-1k first** — already AnnData-native; validates the input contract end-to-end
   with the least glue code, and doubles as the reference-atlas pool.
2. **One Visium HD** (CRC) — exercises the high-resolution target path
   (`binned_outputs.tar.gz` → schema-valid `.h5ad`).
3. **One Xenium** (Breast, verified) — exercises sparse targeted-panel imputation.
4. Backfill the remaining HD (Brain, Tonsil) and Xenium Skin (after resolving the
   ⚠️ hotlink), each landing a loader, a tiny smoke test, and a row in this table.

Per-dataset tracking issues are linked from the integration PR's checklist.
