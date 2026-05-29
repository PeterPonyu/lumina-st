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

## Extended validation datasets

> Added 2026-05-28 after an independent **raw-integer-count re-verification** of every
> dataset above (web checks against the 10x dataset pages, the Zenodo API, and the HEST
> paper/HF card). Tracking issues #54–#60 carry the per-dataset evidence comments;
> expansion issues #62–#64 are below. All download URLs returned **HTTP 200** on 2026-05-28.

### Raw-count verification verdicts (existing datasets)

| # | Dataset | Issue | Verdict | Raw-count artifact (verified) |
|---|---------|-------|---------|-------------------------------|
| 1 | HEST-1k | #54 | ✅ CONFIRMED | `st/*.h5ad` — raw transcript counts in `.X` + (x,y) coords (HEST paper, [arXiv 2406.16192](https://arxiv.org/abs/2406.16192)). |
| 2 | her2st | #55 | ✅ CONFIRMED | `count-matrices.zip` (37.24 MB) — integer `[n_spots]×[n_genes]` `.tsv.gz` per section ([Zenodo 3957257](https://zenodo.org/records/3957257)). |
| 3 | Visium HD CRC | #56 | ✅ CONFIRMED | `Visium_HD_Human_Colon_Cancer_binned_outputs.tar.gz` (**15.9 GB**) → `square_008um/filtered_feature_bc_matrix.h5` raw UMIs. Resolves UNVERIFIED caveat #1. |
| 4 | Visium HD Mouse Brain | #57 | ✅ CONFIRMED | `Visium_HD_Mouse_Brain_binned_outputs.tar.gz` (**4.6 GB**) → `square_008um/filtered_feature_bc_matrix.h5`. |
| 5 | Visium HD Tonsil | #58 | ✅ CONFIRMED | `Visium_HD_Human_Tonsil_Fresh_Frozen_binned_outputs.tar.gz` (**17.4 GB**) → `square_008um/filtered_feature_bc_matrix.h5`. *(registry "2–4 GB" estimate is far too low.)* |
| 9 | Xenium Human Skin | #59 | ⚠️ PARTIAL | raw `cell_feature_matrix.h5` inside `…_outs.zip` (11.8 GB) is fine, **but the issue/registry page link 404s** — correct page adds the `-1-standard` suffix; the public sample is *non-diseased* skin (→ `UNKNOWN`). |
| 11 | Xenium Breast (Janesick) | #60 | ✅ CONFIRMED | `Xenium_FFPE_Human_Breast_Cancer_Rep1_cell_feature_matrix.h5` (12 MB) raw per-cell counts (full `_outs.zip` 9.86 GB). |

**Net:** 6/7 ✅ raw counts confirmed; 1 ⚠️ (#59 Xenium Skin) needs only a **link correction** to
`…/datasets/human-skin-data-xenium-human-multi-tissue-and-cancer-panel-1-standard` — no replacement
dataset required. No dataset failed (no ❌).

### New expansion datasets (verified raw counts)

Three additional raw-count, single-slice, histology-paired targets that close validation
gaps for LuminaST's latent-enhancement / guided-imputation method:

| Dataset | Issue | Platform | Tissue | Raw-count artifact (HTTP 200) | Why it fits LuminaST |
|---------|-------|----------|--------|-------------------------------|----------------------|
| DLPFC 12-sample Visium (Maynard 2021 / spatialLIBD) | #62 | Visium v1 | human DLPFC (brain) | `SpatialExperiment` `counts` assay = raw integer UMIs (`spatialLIBD::fetch_data("spe")`) + manual L1–L6+WM labels | **Gold-standard imputation benchmark**: expert layer labels → layer-ARI / held-out-gene scoring; serial replicates → enhancement-consistency checks. |
| Visium CytAssist FFPE Human Breast Cancer | #63 | Visium CytAssist (FFPE) | breast cancer | `CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5` (33 MB) | Low-friction single-`.h5` breast Visium slice → cross-platform breast (legacy ST ↔ Visium ↔ Xenium). `BRCA`. |
| Visium HD Human Breast Cancer (FFPE, IF) | #64 | Visium HD (8 µm) | breast cancer | `Visium_HD_Human_Breast_Cancer_FFPE_binned_outputs.tar.gz` (6.75 GB) → `square_008um/filtered_feature_bc_matrix.h5` | Same-tissue **multi-resolution** breast (HD ↔ Visium ↔ Xenium ↔ legacy ST); near-single-cell HD enhancement. `BRCA`. |

The canonical [`DATASET_REGISTRY.md`](../../../datasets/DATASET_REGISTRY.md) still governs
accessions; the tables above record the 2026-05-28 raw-count re-verification and the focused
expansion set.

## Consolidated dataset framework (2026-05-29)

> Added 2026-05-29 per meta-issue **#182**, which asks to fold the 05-29 cards, the fetch
> surface, and the literature links into this draft PR. This section consolidates **all**
> tracked LuminaST datasets — the recommended set (#54–#60), the expansion set (#62–#64), the
> next-generation **Xenium Prime 5K cancer cohort** (#65–#67, consolidated by **#184**), and
> the squidpy/scanpy-native **Tier B** loaders (#68–#70) — behind one machine-readable registry
> and one offline-safe fetch CLI.

### Machine-readable registry + unified fetcher

| Surface | Path | Role |
|---------|------|------|
| Dataset registry | [`src/lumina_st/data/dataset_registry.py`](../src/lumina_st/data/dataset_registry.py) | One `DatasetSpec` per dataset: id, platform, accession/URL (with `UrlStatus`), citation key, **raw-count policy**, and **contract mapping** onto `SpatialTranscriptomicsDataset` / `ReferenceAtlasDataset`. The single source of truth that backs every card below. |
| Unified fetcher | [`scripts/data/fetch_datasets.py`](../scripts/data/fetch_datasets.py) | CLI dispatching per-dataset fetch from the registry. `--list` / `--help` are offline (exit 0); real downloads are gated behind `--download` and only run for **verified** URLs. `UrlStatus.UNVERIFIED` datasets print `URL UNVERIFIED - see issue #N` and refuse to download. |
| Literature links | [`manuscript/LITERATURE_LINKS.md`](../manuscript/LITERATURE_LINKS.md) | Source paper / portal for every dataset, keyed by `citation_key`. |

```bash
python scripts/data/fetch_datasets.py --list                       # all datasets, offline
python scripts/data/fetch_datasets.py --dataset her2st --dry-run   # describe, no download
python scripts/data/fetch_datasets.py --dataset her2st --download  # real fetch (verified only)
python scripts/data/fetch_datasets.py --dataset xenium_prime_breast  # refuses: URL UNVERIFIED (#65/#184)
```

Raw downloads land under `~/Desktop/ST_research/data_cache/raw/<id>/`; the repo `.gitignore`
ignores all of `data/`, so only the docs + registry are tracked.

### Next-generation Xenium Prime 5K cancer cohort (#184 — upgrades #65–#67)

State-of-the-art high-plex (~5k-gene) FFPE imaging ST. All three carry an **⚠️ UNVERIFIED**
Output-Bundle hotlink (canonical page given; pull the bundle from the Download tab — never
hardcode a guessed `cf.10xgenomics.com/...` URL).

| Dataset | Issue(s) | Panel / cells | `cancer_type` | Raw-count artifact | Why it fits LuminaST |
|---------|----------|---------------|---------------|--------------------|----------------------|
| Xenium Prime 5K Breast Cancer | #65, #184 | ~5,100 genes · 699,110 cells | `BRCA` | `cell_feature_matrix.h5` / `.zarr.zip` (Output Bundle) | ~16× panel vs the 313-gene #60 demo; richest Xenium panel-imputation + cross-platform breast target. |
| Xenium Prime 5K Skin / Dermal Melanoma | #66, #184 | 5,006 genes · 112,551 cells | `SKCM` | `cell_feature_matrix.h5` / `.zarr.zip` (PREVIEW) | Real melanoma upgrade of the non-diseased 377-gene #59 skin; fast Prime smoke target. **Preview Data** — verify bundle completeness first. `SKCM` is not in `default_pan_cancer` → needs registry config or `UNKNOWN`. |
| Xenium Prime 5K Ovarian Cancer | #67, #184 | ~5,100 genes · 407,124 cells | `OV` | `cell_feature_matrix.h5` / `.zarr.zip` (Output Bundle) | First ovarian slice in the set; ships a pathologist-annotated H&E for region-aware evaluation. |

All three: `X` ← raw per-cell integer counts (drop control/negative probes); `obsm['spatial']`
← `(x_centroid, y_centroid)`; read via `spatialdata_io.xenium()` or scanpy on
`cell_feature_matrix.h5`.

### Tier B — squidpy / scanpy-native loaders (#68–#70)

Native one-liner loaders (run in the `dl` conda env: scanpy 1.10.4 / squidpy 1.6.5);
all verified to load on 2026-05-28.

| # | Dataset | Issue | Loader | Raw counts | `cancer_type` | Role |
|---|---------|-------|--------|-----------|---------------|------|
| B3 | Slide-seqV2 cerebellum | #68 | `sq.datasets.slideseqv2()` | ✅ raw UMI in `.X` | `UNKNOWN` | Primary **sparse** raw-count imputation target (41,786 beads × 4,000 genes). |
| B5 | sc_mouse_cortex reference | #69 | `sq.datasets.sc_mouse_cortex()` → `adata.raw.to_adata()` | ✅ raw in `.raw` (restore to `.X`) | `UNKNOWN` | **VAE reference atlas** (not spatial → `ReferenceAtlasDataset`). CZ CELLxGENE Census = real-human-cancer upgrade path. |
| B6 | Visium breast cancer Block A (2 serial sections) | #70 | `sc.datasets.visium_sge('V1_Breast_Cancer_Block_A_Section_1' / '..._Section_2')` | ✅ raw UMI in `.X` | `BRCA` | Real cancer ST target; serial sections → paired-slice experiments. HEST-1k (#54) backs the pan-cancer claim. |

> **CancerRegistry note (#70):** `BRCA` and `OV` are present in
> `configs/stpainter_registry.yaml`; `SKCM` (#66) is not. `CancerRegistry.default_pan_cancer`
> falls back to `UNKNOWN` for any unlisted token, so loaders either load a registry file that
> defines the token or accept the documented `UNKNOWN` mapping. No code change is forced here.

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

1. **HEST-1k first (#54)** — already AnnData-native; validates the input contract end-to-end
   with the least glue code, and doubles as the reference-atlas pool.
2. **Tier B smoke loaders (#68 Slide-seqV2, #69 sc_mouse_cortex, #70 Visium breast)** —
   squidpy/scanpy one-liners; the fastest path to an end-to-end `enhance()` run and the
   VAE reference atlas (#69) + a real cancer ST target (#70).
3. **One Visium HD (#56 CRC)** — exercises the high-resolution target path
   (`binned_outputs.tar.gz` → schema-valid `.h5ad`).
4. **One Xenium (#60 Breast, verified)** — exercises sparse targeted-panel imputation;
   then the low-friction Visium breast `.h5` (#63) and the manual-layer DLPFC benchmark (#62).
5. **Backfill HD (#57 Brain, #58 Tonsil, #64 HD Breast)** and **Xenium Skin (#59,** after
   resolving the ⚠️ hotlink).
6. **Next-gen Xenium Prime 5K cohort (#184 → #65 Breast, #66 Skin/Melanoma, #67 Ovarian)** —
   highest-plex targets; gated on resolving each ⚠️ UNVERIFIED Output-Bundle link from the
   Download tab.

Each step lands a registry entry (already done — see `dataset_registry.py`), a loader, a tiny
smoke test, and a row in the tables above. Per-dataset tracking issues are linked from the
integration PR's checklist. Infra adjacency: **#71** (publish/register real checkpoints) governs
`src/lumina_st/cli/download.py` and is tracked separately from dataset ingestion.
