# Phase 2 Progress — LuminaST Core Implementation

**Date**: 2026-05-21 (evening session)

## Delivered in this iteration

### Configuration & Data (foundation)
- `config/lumina_config.py` — Comprehensive Pydantic v2 `LuminaSTConfig` replacing all argparse
- `data/cancer_registry.py` — Fully data-driven `CancerRegistry` (kills the hard-coded 21-type dict)
- `data/datasets.py` — Clean `ReferenceAtlasDataset` + `SpatialTranscriptomicsDataset` (AnnData native)

### Models
- `models/embeddings.py` — Fresh `TimestepEmbedder`, `LabelEmbedder`, `PatchEmbedder`
- `models/lumina_transformer.py` — `LuminaTransformer` + `LuminaBlock` (complete rebrand of the latent GiT/DiT)

### Training & High-level API
- `modules/lumina_flow_module.py` — `LuminaFlowModule` (Lightning) wired to our new `FlowTransport` + EMA + guided sampling skeleton
- `core/lumina_imputer.py` — `LuminaImputer` high-level facade (the class users will actually call)

### Package exposure
- Updated `src/lumina_st/__init__.py` to export the new public API

## Architecture decisions made
- The flow primitives from Phase 1 are the single source of truth for transport/sampling.
- VAE is kept as a pluggable interface (`vae.encode_to_latent`) so we can support both scvi and future internal VAEs.
- CFG logic will live in the module / imputer (double batch trick).
- Sparsity post-processing and proper forward-noising to `t_forward` will be implemented next.

## Phase 2 Status (updated)

**Completed in this pass:**
- `latents/vae_interface.py` + `IdentityLatentEncoder` (pluggable design)
- Proper forward diffusion (`get_noisy_xt`) added to `FlowTransport`
- Full `enhance_latent()` with partial noising + CFG Euler integration in `LuminaFlowModule`
- Complete `LuminaImputer.enhance()` pipeline (encode → enhance → decode → AnnData output + sparsity)
- `EnhancementEvaluator` (Pearson/Spearman + clustering ARI/NMI)
- Training script: `scripts/train_latent_flow.py`
- Inference script: `scripts/run_enhancement.py`

Phase 2 is now in a **usable state** for prototyping and small experiments.
The remaining work is mostly integration (wiring a real scvi VAE, adding proper DataLoaders for ST, config YAML examples, and end-to-end tests on the 5090).

## Leakage status
**Zero** occurrences of original stPainter / GiT / author strings in any new Phase 2 file.

Ready to continue with the remaining pieces of Phase 2 or move in parallel to Phase 3 (Aether3D) if the user prefers.
