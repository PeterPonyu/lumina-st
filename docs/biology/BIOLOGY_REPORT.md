# LuminaST Biology Figure Pack

LuminaST is a latent flow-matching model for spatial transcriptomics enhancement. Given a tissue slice with sparse, noisy gene measurements, it produces:

- a **dense imputed expression matrix** (`layers['imputed']`) that fills in dropouts while preserving the slice's spatial structure;
- an **enhanced latent embedding** (`obsm['latent_enhanced']`) optimised for tissue-domain clustering and marker discovery;
- a **post-processed sparsity profile** matched to the expected per-gene detection rate for the cancer type.

This report exercises those outputs on two data sources: a synthetic reference + ST pair (fully reproducible from the sweep artifacts), and a panel of on-disk spatial transcriptomics cancer slices that LuminaST imputes in place. No online downloads are required.

## Synthetic

### wide

- source: `results/benchmark/enhanced/wide.h5ad`
- cells: 300, genes: 120, runtime: 10.3s, device: `cpu (precomputed)`

**Enhanced-latent UMAP** (colored by Leiden + true label when available)

![latent_umap](./synthetic/wide/figures/latent_umap.png)

**Per-gene Pearson, HVG vs non-HVG**

![pergene_pearson_box](./synthetic/wide/figures/pergene_pearson_box.png)

**Top-50 HVG gene-gene correlation matrix, raw vs imputed**

![gene_gene_corrheatmap](./synthetic/wide/figures/gene_gene_corrheatmap.png)

**Per-cell sparsity histogram, raw vs imputed**

![sparsity_histogram](./synthetic/wide/figures/sparsity_histogram.png)

**Marker discovery on imputed expression** (Wilcoxon per Leiden cluster)

![marker_volcano](./synthetic/wide/figures/marker_volcano.png)

**Comparative UMAPs** (raw PCA / `latent_observed` / `latent_enhanced`)

![comparative_umaps](./synthetic/wide/figures/comparative_umaps.png)

**Canonical-lineage-marker dot plot** on imputed expression per Leiden cluster

![lineage_dotplot](./synthetic/wide/figures/lineage_dotplot.png)

**PCC / SSIM / RMSE / JS line plots** vs n_HVG (post-hoc subset, no retraining)

![pcc_ssim_nhvg_sweep](./synthetic/wide/figures/pcc_ssim_nhvg_sweep.png)

**Sankey of Leiden → in-data label**

![sankey](./synthetic/wide/figures/leiden_to_label_sankey.png)

[interactive HTML](./synthetic/wide/figures/leiden_to_label_sankey.html)

## Real

### CESC

- source: `data/baselines/stpainter/processed_data/st_CESC_test.h5ad`
- cells: 4,000, genes: 10,000, runtime: 52.6s, device: `cuda`

**Spatial expression — raw vs LuminaST-imputed (top-variance genes)**

![spatial_marker_DSC3.png](./real/CESC/figures/spatial_marker_DSC3.png)
![spatial_marker_HSPB1.png](./real/CESC/figures/spatial_marker_HSPB1.png)

**Enhanced-latent UMAP** (colored by Leiden + true label when available)

![latent_umap](./real/CESC/figures/latent_umap.png)

**Per-gene Pearson, HVG vs non-HVG**

![pergene_pearson_box](./real/CESC/figures/pergene_pearson_box.png)

**Top-50 HVG gene-gene correlation matrix, raw vs imputed**

![gene_gene_corrheatmap](./real/CESC/figures/gene_gene_corrheatmap.png)

**Per-cell sparsity histogram, raw vs imputed**

![sparsity_histogram](./real/CESC/figures/sparsity_histogram.png)

**Marker discovery on imputed expression** (Wilcoxon per Leiden cluster)

![marker_volcano](./real/CESC/figures/marker_volcano.png)

**Comparative UMAPs** (raw PCA / `latent_observed` / `latent_enhanced`)

![comparative_umaps](./real/CESC/figures/comparative_umaps.png)

**Canonical-lineage-marker dot plot** on imputed expression per Leiden cluster

![lineage_dotplot](./real/CESC/figures/lineage_dotplot.png)

**PCC / SSIM / RMSE / JS line plots** vs n_HVG (post-hoc subset, no retraining)

![pcc_ssim_nhvg_sweep](./real/CESC/figures/pcc_ssim_nhvg_sweep.png)

**Per-patch raw vs imputed marker grid** (top-variance genes, quadrant split)

![spatial_marker_grid](./real/CESC/figures/spatial_marker_grid.png)

**Held-out HVG recovery** — single-modality imputation benchmark (NOT a proteomics surrogate)

![gene_holdout_recovery](./real/CESC/figures/gene_holdout_recovery.png)

**Sub-celltype dissection per lineage** (Jaccard self-check enforced)

![subcelltype_T_cell.png](./real/CESC/figures/subcelltype_T_cell.png)
![subcelltype_Myeloid.png](./real/CESC/figures/subcelltype_Myeloid.png)
![subcelltype_Epithelial.png](./real/CESC/figures/subcelltype_Epithelial.png)
- `T_cell`: gated=1000 cells, sub-clusters=4, J(gating, top-K DEGs)=0.00
- `Myeloid`: gated=1000 cells, sub-clusters=8, J(gating, top-K DEGs)=0.00
- `Epithelial`: gated=1000 cells, sub-clusters=6, J(gating, top-K DEGs)=0.00

**Sankey of Leiden → in-data label**

![sankey](./real/CESC/figures/leiden_to_label_sankey.png)

[interactive HTML](./real/CESC/figures/leiden_to_label_sankey.html)

**Pan-cancer panel — one column per on-disk baseline slice (real mode only)**

![cancer_panel](./real/cancer_panel.png)

Cancers rendered: CESC, COAD, LIHC, NSCLC, OV, PRAD

---

<!-- TODO(ref-parity): biology_figures generator removed pending real-data
results; regenerate the figure pipeline once computational results exist. -->
Figure-generation script pending regeneration after real-data results.
