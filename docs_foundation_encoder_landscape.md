# LuminaST Round 6: foundation-model latent encoder branch

Branch: `feature/round6-foundation-encoder-20260526`

## Goal

Wire spatial/single-cell foundation models as optional LuminaST latent encoders
without taking hard dependencies on large checkpoints or model-specific packages.
This branch provides the safe integration seam; actual scGPT/Nicheformer weights,
vocabularies, and fine-tuning recipes remain explicit downstream choices.

## Current adapter contract

- `lumina_st.latents.FoundationLatentEncoder` wraps any callable or `nn.Module`
  that returns a cell embedding tensor or a dict containing one of:
  `cell_emb`, `embedding`, `embeddings`, `latent`, or `z`.
- The adapter projects the foundation embedding to `LuminaSTConfig.latent_dim`.
- It exposes the same `LatentEncoder` API used by TinyVAE/scVI:
  `encode_to_latent()` and `decode_from_latent()`.
- A linear decoder is provided only as a smoke/readout path; paper claims need
  calibrated training before use.

## Prioritized options from online scan

| Option | Why it matters for LuminaST | Branch stance |
| --- | --- | --- |
| Nicheformer | Official repo for a foundation model trained on single-cell and spatial omics; Nature Methods paper reports SpatialCorpus-110M and spatial-context tasks. | Preferred spatial-first backend; optional import check only. |
| scGPT-spatial | Continual pretraining of scGPT for ST; repo advertises SpatialHuman30M, Visium/Visium HD/Xenium/MERFISH, missing-gene imputation, and deconvolution. | Strong drop-in candidate for held-out gene recovery. |
| SAGE-FM | Lightweight GCN spatial foundation model trained on HEST1k/Visium-style subgraphs with embedding export scripts. | Candidate for graph/neighborhood-aware embeddings when transcript-only transformers underperform. |
| SpatialPEFT | Parameter-efficient fine-tuning framework for spatial transcriptomics foundation models. | Candidate for local GPU fine-tuning without full-model updates. |
| STFM_BASELINE / spatialFM-embedding | Community repos for comparing/extracting ST foundation embeddings. | Use as benchmark harness references, not core dependency. |
| Geneformer / UCE / scFoundation | Strong single-cell foundation baselines but not always spatial-native. | Secondary baselines; use to test if spatial-specific pretraining really helps. |

## Next acceptance gate

1. Choose a backend (`nicheformer` or `scgpt`) and freeze a model/vocabulary
   license note.
2. Add a small adapter wrapper in user code that returns cell embeddings.
3. Train/calibrate the readout decoder on one held-out gene panel.
4. Compare against TinyVAE/scVI latent encoders using the existing held-out gene
   tests and benchmark contracts.
