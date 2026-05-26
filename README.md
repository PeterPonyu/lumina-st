# LuminaST

**Target API for schema-aware spatial transcriptomics enhancement and imputation**

LuminaST is a local research package for spatial transcriptomics (ST) enhancement and imputation experiments. Current evidence supports synthetic/local-small smoke validation; pan-cancer, zero-shot, platform-transfer, and baseline-superiority claims remain gated by `../docs/CLAIM_LEDGER.md`.

Given a schema-valid ST slice, the target API returns:
- an enhanced latent embedding for downstream evaluation;
- an imputed gene-expression layer for benchmarked analyses.

Do not treat these outputs as biologically superior or analysis-ready until the corresponding claim-ledger rows have local benchmark, data-card, and figure evidence.

## Why "LuminaST"?

The name evokes a "light" (lumen) for examining weak or sparse spatial measurements. The current package is a benchmark-controlled implementation surface, not yet evidence for broad biological improvement.

## Installation (Planned)

```bash
pip install lumina-st
# or for development
git clone https://github.com/<your-org>/lumina-st
cd lumina-st
pip install -e ".[dev,scvi]"
```

## Quick Start (Target API)

```python
from lumina_st import LuminaImputer
from lumina_st.config import LuminaSTConfig
import scanpy as sc

cfg = LuminaSTConfig(
    latent_dim=50,
    cancer_type="COAD",           # or list for pan-cancer
    guidance_scale=3.0,
    checkpoint="checkpoints/lumina_50.ckpt"
)

imputer = LuminaImputer(cfg)
st_slice = sc.read_h5ad("my_xenium_coad.h5ad")

enhanced = imputer.enhance(st_slice)          # returns AnnData with .obsm['latent_enhanced'], .layers['imputed']
```

## Architecture (High Level)

1. **Latent Encoder** — Lightweight VAE (scVI-style or internal) for reference-atlas latent experiments → 50/100-dim latents.
2. **Conditional Flow Matching** — `LatentVelocityNet` (DiT-style transformer with class conditioning + CFG) learns the velocity field of the latent distribution.
3. **Guided Imputation** — For a new ST observation, encode observed genes → partial noise forward to `t_forward` → conditional reverse sampling (with cancer-type guidance) → decode + sparsity-aware post-processing.

This architecture defines the local LuminaST implementation surface. Publication claims are controlled by the claim ledger and must be validated with local benchmarks before manuscript use.

## Prior Art and Audit Boundary

LuminaST is an independent package and manuscript track for spatial-transcriptomics enhancement. The public stPainter preprint and repository are treated as prior art for the general research problem and for audit comparison only; LuminaST's user-facing API, documentation, validation plan, figures, and manuscript claims must be written from local evidence.

See [BASELINE_COMPARISON.md](./BASELINE_COMPARISON.md) and the immutable audit clone in `../baselines/stPainter-original/` for traceability and leakage checks. Claims graduate to the manuscript only through the project claim ledger and reproducible benchmark artifacts, not by inheriting claims from the reference work.

**Prior-art citation** (to appear in final paper):
> Yang, Y. et al. "Enhancing Pan-cancer Spatial Transcriptomics at Single-cell Resolution with stPainter." bioRxiv (2026).

## Status (2026-05-21)

- **Phase 0** — Skeletons, audit baselines, git repos, data docs, 5090 verification: **complete**
- **Phase 1** — Clean-room `flow/` primitives (Linear/GVP/VP paths, FlowTransport, FlowSampler, ODE/SDE integrators, CFG-ready): **complete** — 4 unit tests passing, zero original strings.

Next: Phase 2 (LuminaST high-level API, latent encoder, transformer, guided imputation).

Target: two independent publication-grade packages and manuscript tracks once the claim ledger, benchmark contracts, data cards, and reproducible figure artifacts support the intended claims.

## License

MIT (to be confirmed — matching the clean spirit of the baselines).

## Contact & Contributing

(Internal lab project — details to be added after first working prototype.)

---

*Part of the Lumina / Aether spatial omics re-implementation program, 2026.*

### Optional foundation-model latent encoders

Round 6 adds a lightweight adapter seam for spatial/single-cell foundation
models. `lumina_st.latents.FoundationLatentEncoder` can wrap scGPT-spatial,
Nicheformer, SAGE-FM, Geneformer/UCE-style embedding callables, project their
cell embeddings to LuminaST latent space, and keep the downstream flow API
unchanged. These backends are optional: no heavyweight checkpoint dependency is
installed by default, and the bundled linear decoder is a smoke/readout path
rather than paper-ready calibrated imputation. See
`docs_foundation_encoder_landscape.md`.
