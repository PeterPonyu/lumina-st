# LuminaST vs stPainter Baseline — Refactor Mapping & Leakage Audit

**Project**: LuminaST (Illuminating Latent Priors for Pan-Cancer Spatial Transcriptomics Enhancement)
**Baseline**: `../baselines/stPainter-original/`
**Date of mapping**: 2026-05-21
**Status**: Initial skeleton (heavy refactor + rename in progress)

## Purpose
This document is the living audit artifact. It proves that the final LuminaST source contains **no textual or structural leakage** from the public stPainter repository while preserving the exact scientific method (two-stage VAE + conditional latent flow-matching for guided ST imputation).

## High-Level Rename Table

| Original Concept / File                  | LuminaST Equivalent (new)                          | Rationale / Improvement |
|------------------------------------------|----------------------------------------------------|---------------------------|
| `stPainter` (brand)                      | `LuminaST` / `lumina_st`                           | Full rebrand for new paper series |
| `train_vae.py` + `VAEModule`             | `scripts/train_latent_encoder.py` + `LatentEncoderModule` | Clearer name, Pydantic config |
| `train_diffsuion.py` (typo kept)         | `scripts/train_latent_flow.py`                     | Fixed spelling, modern CLI |
| `impute_stPainter.py`                    | `scripts/run_enhancement.py` + `LuminaImputer.enhance()` | User-facing verb "enhance" |
| `DiffusionModule` + `impute()`           | `LuminaFlowModule` + `impute_expression()`         | Consistent with Aether3D naming |
| `GiT` / "Gene Diffusion Transformer"     | `LatentVelocityNet` (in `models/velocity_net.py`) | No "GiT", no paper phrase |
| `STDataset` / `SCDataset`                | `SpatialTranscriptomicsDataset`, `ReferenceAtlasDataset` | Explicit, no acronym in public API |
| `TUMOR_TO_IDX` (hard-coded 21 cancers)   | `lumina_st/data/cancer_registry.py` + YAML       | Extensible, no magic numbers |
| `CalculateMetrics`                       | `EnhancementEvaluator` + metric registry         | Pluggable, paper-agnostic |
| `src/transport/` (duplicated)            | `lumina_st/flow/` (clean re-type)                | Deduped within project; same clean copy in Aether3D |
| `src/models/commons.py` + `git.py`       | `lumina_st/models/{embeddings,blocks,velocity_net}.py` | Fresh docstrings, modern PyTorch (SDPA, compile) |
| argparse + raw dicts                     | `LuminaSTConfig` (Pydantic v2) + Hydra/OmegaConf optional | Type-safe, serializable, reproducible |
| `process_data.py`                        | `lumina_st/data/processors.py` + CLI             | Reusable functions, no side effects |

## Leakage Prevention (Automated + Manual)

Run this from repo root before every PR:
```bash
rg --type py --type md --glob '!BASELINE_COMPARISON.md' \
   'DeepSpatial|stPainter|yyh030806|Gene Diffusion Transformer|10.64898/2026.02.11.704553' \
   src/ docs/ scripts/ experiments/
```
**Must return zero matches** (except the citation block in top README).

## Scientific Equivalence Checklist (to be filled Phase 5)

- [ ] VAE latent (50/100 dim) reconstruction Pearson > baseline on COAD test set
- [ ] Conditional flow-matching imputation SSIM / gene Pearson parity
- [ ] Clustering ARI/NMI on enhanced latent matches or exceeds original
- [ ] Sparsity post-processing logic preserved (but configurable)
- [ ] CFG (classifier-free guidance) behavior identical

Any numerical deviation > 2% will be documented with seed, batch size, and torch version.

## Files Intentionally Not Ported (or heavily rewritten)
- Original `figure/overview.png` — new branded figures will be generated
- All docstrings and comments — 100% rewritten
- `parsing.py` (15 argparse groups) — replaced by Pydantic + YAML examples in `configs/`

## Next Audit Milestones
- After Phase 1 (flow primitives): numerical unit-test parity
- After Phase 2 (full LuminaST): full tutorial reproduction + new experiment
- Pre-paper submission: final `rg` scan + human review of every public string

**Signed off by**: (to be filled at each milestone)

This document + the pristine `baselines/stPainter-original/` together constitute the complete change audit for the new LuminaST publication.
