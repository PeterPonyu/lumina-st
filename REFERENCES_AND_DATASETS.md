# LuminaST — references (with code) & datasets

Consolidated reference + dataset index. Paper DOIs verified via Crossref and code
repositories via the GitHub API on 2026-06-09. See `manuscript/refs.bib`,
`docs/DATASETS.md`, `manuscript/LITERATURE_LINKS.md`, and the benchmark adapters under
`src/lumina_st/benchmarks/adapters/`.

## Reference papers & method baselines (with public code)

| Role | Method | Venue / year | DOI | Code |
|------|--------|--------------|-----|------|
| Primary | SpaIM — single-cell ST imputation via style transfer | Nature Communications 2025 | `10.1038/s41467-025-63185-9` | https://github.com/QSong-github/SpaIM |
| Baseline ⚙ | TISSUE — uncertainty-calibrated ST prediction | Nature Methods 2024 | `10.1038/s41592-024-02184-y` | https://github.com/sunericd/TISSUE |
| Baseline ⚙ | Tangram — deep learning + alignment of scRNA & ST | — | `10.1038/s41592-021-01264-7` | https://github.com/broadinstitute/Tangram |
| Baseline ⚙ | gimVI — joint model of scRNA & ST for imputation | — | arXiv 1905.02269 | https://github.com/scverse/scvi-tools |
| Baseline ⚙ | NovoSpaRc — de novo spatial reconstruction | — | `10.1038/s41586-019-1773-3` | https://github.com/rajewsky-lab/novosparc |
| Baseline ⚙ | stMCDI — conditional diffusion imputation | — | — | https://github.com/lllxxyyy-lxy/stMCDI |
| Baseline ⚙ | stDiff — diffusion-based ST imputation | — | `10.1093/bib/bbae171` | github.com/hannshu/stDiff (⚠️ link 404 — resolve current location) |
| Baseline ⚙ | CellT | — | — | https://github.com/wehos/CellT |

Additional imputation baselines in `manuscript/refs.bib`: SpaGE `10.1093/nar/gkaa740` ·
stPlus `10.1093/bioinformatics/btab298` · SpatialScope `10.1038/s41467-023-43629-w` ·
scVI (Lopez 2018, Nature Methods). Optional foundation-model encoders (seam, not deps):
scGPT (https://github.com/bowang-lab/scGPT) + scGPT-spatial · Nicheformer
(https://github.com/theislab/nicheformer) · UCE (https://github.com/snap-stanford/UCE) ·
Geneformer (https://huggingface.co/ctheodoris/Geneformer).

## Datasets (audited registry — `docs/DATASETS.md`)

- HEST-1k (`MahmoodLab/hest`); her2st (Zenodo **3957257**); Visium HD CRC / mouse brain / tonsil / breast
- Xenium: Human Skin, Breast (Janesick), Prime 5K (breast/skin/ovarian); Visium CytAssist FFPE breast; DLPFC (spatialLIBD)
- Tier B (squidpy): Slide-seqV2 cerebellum, sc_mouse_cortex (VAE reference atlas), Visium breast Block A serial

> Verification: SpaIM + TISSUE + Tangram + NovoSpaRc DOIs confirmed in Crossref; SpaIM /
> TISSUE / Tangram / scvi-tools / NovoSpaRc / stMCDI / CellT / scGPT / Nicheformer / UCE
> repos live via GitHub API. her2st Zenodo 3957257 confirmed accessible (2026-06-09).
> stDiff (hannshu/stDiff) returned GitHub 404 — likely renamed/moved.
