# LuminaST

**Target API for schema-aware spatial transcriptomics enhancement and imputation**

LuminaST is a local research package for spatial transcriptomics (ST) enhancement and imputation experiments. Current evidence supports synthetic/local-small smoke validation; pan-cancer, zero-shot, platform-transfer, and baseline-superiority claims remain gated by [`docs/CLAIM_LEDGER.md`](./docs/CLAIM_LEDGER.md).

Given a schema-valid ST slice, the target API returns:
- an enhanced latent embedding for downstream evaluation;
- an imputed gene-expression layer for benchmarked analyses.

Do not treat these outputs as biologically superior or analysis-ready until the corresponding claim-ledger rows have local benchmark, data-card, and figure evidence.

## Why "LuminaST"?

The name evokes a "light" (lumen) for examining weak or sparse spatial measurements. The current package is a benchmark-controlled implementation surface, not yet evidence for broad biological improvement.

## Installation

This research package is not published on PyPI yet. Install from a clone or a GitHub branch/commit:

```bash
git clone https://github.com/PeterPonyu/lumina-st
cd lumina-st
pip install -e ".[dev,scvi]"
# or: pip install "git+https://github.com/PeterPonyu/lumina-st.git"
```

## Quick Start (Target API)

```python
from lumina_st import LuminaImputer
from lumina_st.config import LuminaSTConfig
import scanpy as sc

st_slice = sc.read_h5ad("my_xenium_coad.h5ad")

# Without a VAE checkpoint, enhance() runs in latent space, so latent_dim
# must equal the input gene-space width. To enhance gene-space data of a
# different width, attach a VAE via LuminaImputer.from_checkpoint(...).
cfg = LuminaSTConfig(
    latent_dim=st_slice.n_vars,
    cancer_types=["COAD"],
    seed=42,
    guidance_scale=3.0,
)

imputer = LuminaImputer.from_config(cfg)
enhanced = imputer.enhance(st_slice)          # returns AnnData with .obsm['latent_enhanced'], .layers['imputed']
```

## Architecture (High Level)

1. **Latent Encoder** — Lightweight VAE (scVI-style or internal) for reference-atlas latent experiments → 50/100-dim latents.
2. **Conditional Flow Matching** — `LuminaTransformer` (DiT-style transformer with class conditioning + CFG, in `src/lumina_st/models/lumina_transformer.py`) learns the velocity field of the latent distribution.
3. **Guided Imputation** — For a new ST observation, encode observed genes → partial noise forward to `t_forward` → conditional reverse sampling (with cancer-type guidance) → decode + sparsity-aware post-processing.

This architecture defines the local LuminaST implementation surface. Publication claims are controlled by the claim ledger and must be validated with local benchmarks before manuscript use.

## Prior Art and Audit Boundary

LuminaST is an independent package and manuscript track for spatial-transcriptomics enhancement. A public prior-art preprint and repository in this problem area are treated as prior art for the general research problem and for audit comparison only; LuminaST's user-facing API, documentation, validation plan, figures, and manuscript claims must be written from local evidence.

An immutable audit clone of the prior-art repository may be placed at `baselines/prior-art-original/` for side-by-side auditing; it is **external and optional** — it is not part of this repository and is not required to use LuminaST. Claims graduate to the manuscript only through the project claim ledger and reproducible benchmark artifacts, not by inheriting claims from the reference work.

**Prior-art citation**: the relevant prior-art reference is tracked privately and will be cited in the manuscript bibliography. It is intentionally kept out of the public package documentation per the project's brand-independence policy.

## Reproducible runs / experiment harness

Every benchmark and sweep run emits a `run_manifest.json` beside its results
JSON, capturing the seed, sweep parameters, dataset id, environment versions
(Python / NumPy / PyTorch), and git SHA.  This makes every run self-describing
and auditable from day one.

**One-command entrypoints:**

```bash
# Full adapter smoke benchmark — writes results/benchmark/synthetic_smoke.json
python scripts/ci/run_synthetic_benchmark.py

# Sparsity / detection-rate sweep across five fractions
python scripts/ci/run_sparsity_sweep.py

# Leave-one-context cross-validation over cancer-type contexts
python scripts/ci/run_leave_one_context.py
```

Each script writes its results JSON and a `run_manifest.json` in the same
output directory.  The manifest is emitted best-effort — any I/O failure is
warned and never disrupts the results write.  See
[`docs/EXPERIMENT_HARNESS.md`](./docs/EXPERIMENT_HARNESS.md) for the full
schema and design notes.

## Status

- **Phase 0** — Skeletons, audit baselines, git repos, data docs: **complete**
- **Phase 1** — Clean-room `flow/` primitives (Linear/GVP/VP paths, `FlowTransport`, `FlowSampler`, ODE/SDE integrators, CFG-ready): **complete**, zero original strings.
- **Phase 2** — High-level API (`LuminaImputer`), latent encoders (`lumina_st/latents/`), `LuminaTransformer`, Lightning training module, guided imputation, and the `lumina_st/benchmarks/` suite: **implemented**.

The full suite runs via `python -m pytest` (see CI for the current pass count). Methodological scope is locked by tests; biological/superiority claims remain gated by the claim ledger.

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
