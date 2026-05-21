# LuminaST

**Illuminating Latent Priors for High-Fidelity Pan-Cancer Spatial Transcriptomics at Single-Cell Resolution**

LuminaST is a deep generative framework that learns **universal latent priors** from massive pan-cancer single-cell RNA atlases and uses them to **enhance and impute** noisy or sparse spatial transcriptomics (ST) data — without retraining per tumor type.

Given a new ST slice (any cancer, any platform), LuminaST returns:
- An enhanced latent embedding (superior for fine-grained cell typing and spatial domain detection)
- A high-quality imputed gene expression matrix (ready for differential expression, ligand-receptor analysis, etc.)

## Why "LuminaST"?

The name evokes a "light" (lumen) that illuminates the true biological signals hidden inside imperfect spatial measurements. The priors act as a denoising beacon learned once from the collective diversity of human cancers.

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

1. **Latent Encoder** — Lightweight VAE (scVI-style or internal) pretrained on pan-cancer scRNA atlas → 50/100-dim latents.
2. **Conditional Flow Matching** — `LatentVelocityNet` (DiT-style transformer with class conditioning + CFG) learns the velocity field of the latent distribution.
3. **Guided Imputation** — For a new ST observation, encode observed genes → partial noise forward to `t_forward` → conditional reverse sampling (with cancer-type guidance) → decode + sparsity-aware post-processing.

This is the exact scientific contribution of the stPainter line of work, fully rebranded, deduplicated, and modernized for new publications.

## Relationship to Baseline

LuminaST is a **heavy refactor + complete rebrand** of the public `stPainter` repository (Yang et al., bioRxiv 2026). The core mathematical machinery (VAE + conditional flow matching on latents + guided partial denoising) is faithfully preserved and numerically validated, while **every identifier, docstring, config system, and narrative** has been freshly authored under the LuminaST brand.

See [BASELINE_COMPARISON.md](./BASELINE_COMPARISON.md) and the immutable audit clone in `../baselines/stPainter-original/` for the complete change log and leakage audit.

**Citation of the inspirational work** (will appear in final paper):
> Yang, Y. et al. "Enhancing Pan-cancer Spatial Transcriptomics at Single-cell Resolution with stPainter." bioRxiv (2026).

We cite it as prior art and inspiration only. All new code, experiments, and claims are original to the LuminaST project.

## Status (2026-05-21)

- **Phase 0** — Skeletons, audit baselines, git repos, data docs, 5090 verification: **complete**
- **Phase 1** — Clean-room `flow/` primitives (Linear/GVP/VP paths, FlowTransport, FlowSampler, ODE/SDE integrators, CFG-ready): **complete** — 4 unit tests passing, zero original strings.

Next: Phase 2 (LuminaST high-level API, latent encoder, transformer, guided imputation).

Target: two independent publication-grade packages + two new bioRxiv preprints with fresh 2026 experiments.

## License

MIT (to be confirmed — matching the clean spirit of the baselines).

## Contact & Contributing

(Internal lab project — details to be added after first working prototype.)

---

*Part of the Lumina / Aether spatial omics re-implementation program, 2026.*
