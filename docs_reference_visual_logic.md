# LuminaST reference visualization logic (clean-room)

Source read: `../2026.02.11.704553v1.full.pdf`, rendered locally under `../results/reference_paper_visual_audit/2026.02.11.704553v1.full/`.

Reference panel layout conventions:

- Fig. 1 style: top-level pipeline schematic from sparse ST + atlas/reference information to imputed expression, latent embedding, and downstream analyses.
- Fig. 2 style: metric curves across evaluation settings, spatial patch marker maps, and scatter/correlation comparisons against external spatial/protein data.
- Fig. 3 style: clustering metrics plus cluster-to-annotation flow/concordance and spatial cluster maps.
- Fig. 4 style: UMAP/marker-dotplot/validation matrices connecting latent structure to biological interpretation.
- Fig. 5 style: sub-lineage dissection with UMAPs, marker dot plots, spatial distributions, and independent validation views.

Clean-room adaptation rules used here:

- Keep only the visualization grammar: method overview, metric trend, spatial marker recovery, cluster/annotation balance, lineage dot plot, validation matrix.
- Use LuminaST-specific labels and repo claim-gating terms.
- Do not copy source paper artwork, logos, captions, exact panel ordering, or unsupported claims.
- Demo values are labelled as planning/demo values until replaced with local benchmark JSON.

Implemented script:

```bash
python scripts/visualize/fig_reference_visual_story.py
```

Primary output:

- `results/figures/lumina_reference_visual_story.png`

## Composed main-claim figure

Implemented script:

```bash
python scripts/visualize/fig_composed_main_claim.py
```

Primary output:

- `results/figures/lumina_composed_main_claim.png`

This figure is intentionally different from the earlier `lumina_reference_visual_story.png`: the earlier file checks that individual panel types align with the reference visual conventions, while the composed main-claim figure arranges those panel types into a single argument chain:

1. sparse target ST and optional reference context;
2. schema/latent/imputation workflow;
3. quantitative recovery trend;
4. spatial marker restoration;
5. annotation concordance gate;
6. downstream biology validation gate;
7. explicit evidence-tier and claim-ledger limitation.

The figure remains `demo/planning` unless local benchmark JSON and real validation artifacts are present. It must not be used for paper-ready superiority or pan-cancer claims until those gates pass.
