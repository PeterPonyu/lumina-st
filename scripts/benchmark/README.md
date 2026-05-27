# LuminaST Synthetic Benchmark Harness

Local, no-download configuration sweep that captures quality metrics and resource
usage across several refined model variants. Designed so the diff is reviewable
and the artifacts are reproducible without any external data.

## Run

```bash
conda run --no-capture-output -n dl python scripts/benchmark/run_synthetic_sweep.py
conda run --no-capture-output -n dl python scripts/benchmark/make_plots.py
```

The first command writes raw JSON under `results/benchmark/` (git-ignored).
The second command renders figures and a Markdown report under
`docs/benchmark/` (git-tracked).

## What is swept

| Name   | latent | hidden | depth | heads | epochs |
|--------|--------|--------|-------|-------|--------|
| tiny   | 16     | 32     | 2     | 2     | 4      |
| small  | 32     | 64     | 2     | 2     | 4      |
| wide   | 32     | 128    | 2     | 4     | 4      |
| deep   | 32     | 64     | 4     | 4     | 4      |

Every config sees the same synthetic reference + ST sample (same seed).

## What is captured per config

- `mean_pearson`, `mean_spearman` (per-gene quality)
- `ari_enhanced_vs_original`, `nmi_enhanced_vs_original` (Leiden agreement)
- Wall seconds for VAE pretrain, flow train, enhance
- Peak GPU MB (when CUDA active)
- Parameter count
- Flow training loss curve per epoch
- Enhanced AnnData (`.h5ad`) for downstream inspection

## Outputs at a glance

```
results/benchmark/                          (git-ignored)
  lumina_sweep_<TS>.json
  lumina_sweep_latest.json
  curves/<config>.json
  enhanced/<config>.h5ad

docs/benchmark/                             (git-tracked)
  BENCHMARK_REPORT.md
  summary.csv
  figures/metric_*.png
  figures/loss_curves.png
  figures/runtime.png
  figures/peak_gpu_mem.png
  figures/latent_umap.png
```

Real-data sweeps (when `data/baselines/st_impute_ref/` is populated) can be added
later by pointing `run_synthetic_sweep.py` at a different data loader; the
metric/resource capture machinery stays the same.
