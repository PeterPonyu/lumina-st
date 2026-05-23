# LuminaST Biology Figure Pack

What it does: produces the biology-focused outputs LuminaST is actually meant to
deliver — imputed spatial expression maps, enhanced-latent UMAPs, marker
discovery — for both synthetic and on-disk baseline data. No online downloads.

## Run

```bash
# both synthetic + real (real uses local baselines under data/baselines/st_impute_ref/)
conda run --no-capture-output -n dl python scripts/visualize/biology_figures.py --mode all

# synthetic only (uses the existing benchmark sweep output)
conda run --no-capture-output -n dl python scripts/visualize/biology_figures.py --mode synthetic

# real only, single cancer
conda run --no-capture-output -n dl python scripts/visualize/biology_figures.py --mode real --cancer COAD
```

Synthetic mode requires that `scripts/benchmark/run_synthetic_sweep.py` has
been run first; it consumes `results/benchmark/enhanced/<config>.h5ad`.
Real mode trains a small LuminaST (TinyVAE, 3 epochs) on each baseline slice
on the fly — no SCVI dependency, no Drive download.

## Figures emitted

Under `docs/biology/<mode>/<dataset>/figures/`:

- `spatial_marker_<gene>.png` — raw vs imputed expression scatter on the tissue
- `latent_umap.png` — UMAP of `obsm['latent_enhanced']`
- `pergene_pearson_box.png` — per-gene Pearson, HVG vs non-HVG
- `gene_gene_corrheatmap.png` — top-50 HVG correlation matrix, raw vs imputed
- `sparsity_histogram.png` — per-cell zero count, raw vs imputed
- `marker_volcano.png` — `rank_genes_groups` markers per Leiden cluster on imputed data
- `cancer_panel.png` (real only) — small-multiples panel across all available cancer slices

`docs/biology/BIOLOGY_REPORT.md` collects everything with embedded thumbnails.
