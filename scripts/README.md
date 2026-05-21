# LuminaST Scripts Organization

This directory contains all executable entry points for LuminaST, clearly separated by purpose.

## Directory Structure

```
scripts/
├── README.md                 # this file
├── ci/                       # Code-level CI / unit tests (fast, no heavy data)
├── data_flow/                # Data pipeline & integration tests (synthetic data, loaders, processors)
└── e2e/                      # Real full-run End-to-End tests (training + inference on real or large data)
```

---

## 1. `ci/` — Code-level CI Workflow Tests

**Purpose**: Fast, deterministic tests suitable for GitHub Actions, pre-commit hooks, or `pytest` in CI.

- Must run in < 2 minutes on CPU with minimal dependencies.
- No real spatial data, no heavy model training.
- Examples: shape checks, loss finite checks, flow primitive correctness, config validation, small model forward passes.

**How to run**:
```bash
# from the repo root
python -m pytest tests/ -q                    # primary location for CI tests
# or
python -m pytest scripts/ci/ -q
```

**Current scripts**: (add small focused tests here as needed)

---

## 2. `data_flow/` — Data Flowing / Integration Tests

**Purpose**: Test the movement of data through the pipeline (AnnData loaders, processors, gene alignment, cancer registry, UOT coupling if applicable, synthetic data generators).

- Still mostly synthetic or small public data.
- Can take a few minutes.
- Useful for debugging data bugs before launching expensive E2E runs.

**How to run**:
```bash
conda run -n dl python -m pytest scripts/data_flow/ -q
# or individual scripts
```

**Current scripts**: (place new data-prep or loader tests here)

---

## 3. `e2e/` — Real Full-Run End-to-End Tests (Most Important for You)

**Purpose**: The actual research scripts that train models and produce **meaningful scientific output** on real (or large simulated) spatial transcriptomics data.

These are the ones you run manually on your workstation using the `dl` conda environment when you want real results for papers.

### Key Scripts

| Script                        | What it does                                      | When to use                          | Expected output |
|-------------------------------|---------------------------------------------------|--------------------------------------|-----------------|
| `enhance_real_st.py`          | Train scVI + Lumina flow on reference → enhance real ST slice | Your main daily research tool       | `.h5ad` with `.layers['imputed']`, `.obsm['latent_enhanced']` + printed metrics (Pearson, ARI, etc.) |
| `train_latent_flow.py`        | Train only the flow model (advanced / checkpointing) | When you want to pre-train the diffusion part separately | checkpoint |
| `run_enhancement.py`          | Inference-only using a pre-trained checkpoint    | Quick enhancement on new slices     | enhanced `.h5ad` |
| `verify_lumina_pipeline.py`   | Deep synthetic E2E verification (training + enhancement on fake data) | Quick sanity check that the whole stack still works | Pass/fail + loss curve |

### Recommended Way to Run E2E Scripts (Real Data)

```bash
cd lumina-st

# The correct and recommended way
conda run -n dl python scripts/e2e/enhance_real_st.py \
    --reference /path/to/your/reference_atlas.h5ad \
    --target    /path/to/your/real_xenium_or_visium.h5ad \
    --cancer    COAD \
    --output    ./results/my_enhanced.h5ad \
    --max_epochs 30
```

This is the script that should produce the "meaningful results" you validate biologically before writing papers.

---

## Quick Decision Guide

- Need to check if the code still compiles / basic math works after a change? → `tests/` or `scripts/ci/`
- Debugging why a real dataset fails to load / align / produce correct tensors? → `scripts/data_flow/`
- Want to actually enhance a real ST dataset and see improved clustering / imputation quality for a paper? → `scripts/e2e/enhance_real_st.py` (run in `dl` env)

Keep the `e2e/` folder focused on scripts that output real `.h5ad` files and printed biological/technical metrics.

---

**Last updated**: 2026-05-21 (after reorganization for clarity between CI, data-flow, and real research E2E runs)
