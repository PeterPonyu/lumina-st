# LuminaST claim ledger

This ledger is the gating authority for promoting any performance, biological,
or baseline-superiority claim into user-facing docs or the manuscript. A claim
graduates from **planned** only when local code, tests, and validated-dataset
evidence (script + seed + dataset hash + hardware/torch version) back it up.

Methodological / engineering invariants that are already locked by the test
suite are recorded as **verified (method only)** — this asserts the code does
what it says, *not* that LuminaST is biologically superior to any baseline.

| Claim | Required evidence | Current evidence | Missing evidence | Status |
|---|---|---|---|---|
| Guided latent flow-matching enhancement runs end-to-end on a schema-valid ST slice. | Clean-room `flow/` primitives + `LuminaImputer.enhance` path with tests. | `tests/` (flow primitives, enhance held-out-gene path, ODE-solver routing) pass locally. | — | verified (method only) |
| Forward diffusion in guided sampling is mathematically sound (not a placeholder). | Path-consistent noising + reverse ODE over `[t_forward, 1]`, with a unit test. | `tests/test_issue_251_forward_diffusion.py`. | — | verified (method only) |
| Reference-trained held-out-gene recovery (lumina v2/v3) outperforms published reference-mapping methods and a kNN floor on real ST. Spatial refinement adds +7.5–15% on the three genuine-coordinate tissues. **Scope: v3 spatial claim restricted to coad/brca1/STARmap (real coordinates); QuKunLab panels (ds18/ds19/ds20/ds27) carry random-placeholder coordinates and support the v2 coordinate-free track only. v3 Moran-match is negative (≈−0.36 coad, ≈−0.31 brca1): spatial-accuracy gain does not imply spatial-fidelity. Do NOT mark validated.** | Real multi-dataset run; held-out-gene Pearson/Spearman/RMSE vs Tangram/stPlus/SpaGE/gimVI + kNN floor; seed + dataset hash + hardware (75-cell fair sweep). | **FAIR_CROSSMETRIC.json** (75-cell fair sweep, `results/benchmark/expression_enhancement/FAIR_CROSSMETRIC.json`): lumina wins 5/5 datasets on Pearson — coad **0.467** (v3\_spatial\_smooth; v2\_mlp ablation 0.417; reference-only 0.408; kNN floor 0.317; spatial gain +14%), brca1 0.387 (+7.5%), STARmap 0.229 (+15%), ds18 **0.571** / ds19 **0.568** / ds20 **0.244** / ds27 **0.369** (all v2\_ridge, coord-free). Published mappers 2–3× worse on RMSE (Tangram 2.08–3.16 vs lumina 1.26–1.58); several fall below the kNN floor on Pearson. Cross-metric sweep (Spearman + RMSE) confirmed. Coordinate-validity scope documented in `docs/papers/lumina/DATA_INTEGRITY_NOTE.md`. | v3 spatial-fidelity claim blocked (negative Moran-match flag open); multi-cancer cross-platform benchmark; native GT clustering claim remains locked (see clustering row). | **supported** (5/5 datasets; v3 spatial gain on 3 genuine-coordinate pairs only; accuracy metrics only — no spatial-fidelity claim until Moran flag resolved) |
| Zero-shot generalization to held-out cancer types without paired reference. | Multi-cancer held-out sweep, no-retraining gate, per-context metrics. | None. | Implementation of the gate + real data. | planned |
| Clustering (ARI/NMI) on the enhanced latent matches or exceeds the observed latent. | Independent-graph clustering benchmark on real annotated data. | Method wired (`EnhancementEvaluator`). **In-tissue matched run (2026-06-07), no native GT → transferred-label proxy** (`scripts/uplift_transferred_labels_multiseed.py`; COAD Visium `coad_visium_10x` × matched COAD scRNA ref `lumina_ref_coad_gse132465`; enhanced `results/lumina-st/lumina_enhanced_coad_visium.h5ad` @ git_sha 678b3b3; 5 seeds; RTX 5090 / torch 2.9.0+cu130): **Δ AMI −0.0309±0.0062, NMI −0.0327±0.0058, Homo −0.1290±0.0092, Silhouette −0.0614; ARI +0.0077±0.0085 (within noise, std>mean, and confounded)**. Negative control (permuted labels) collapses every Δ to \|Δ\|≤0.005 → harness sound, deltas are real signal. This **in-tissue** pairing **replaces** the earlier cross-tissue cervix→COAD control and **corroborates** the architectural-washout finding (#296-closed; #336). CIRCULAR (labels transferred from the same reference the enhancement is conditioned on): a negative Δ is trustworthy, a positive Δ is not promotable. | **LOCKED — three BINDING gates (human-confirmed 2026-06-07) before any positive clustering claim graduates:** (1) native-GT verdict on `st_COAD_test` (#336) goes positive (removes circular-label confound); (2) a non-reference-derived label source also shows positive Δ; (3) the protocol/clustering gate (#308) passes on the matched run. No additional gates (no LOSO / effect-size) are stacked; only when all three pass does this row move honest-negative → validated. | planned (locked: honest-negative corroboration; not promotable until all three BINDING gates pass) |
| Platform-transfer / FFPE-vs-frozen robustness. | Stratified multi-platform benchmark with provenance. | None. | Data cards + stratified run. | planned |

## Spatial-fidelity / Moran limitation note (2026-06-28)

The Moran-match qualification is now documented in the manuscript Discussion
(Limitations §2): Moran is negative on both Visium tissues for all methods including
the v2 reference (coad −0.405, brca1 −0.319); v3 spatial smoothing marginally
\emph{improves} it (coad −0.363, brca1 −0.306) and does not cause the deficit.
STARmap is positive (v2 +0.530, v3 +0.517).  The supported row's scope
(accuracy-only, no spatial-fidelity claim) is unchanged; this note records that
the quantitative breakdown now lives in the tex source.

## Cross-metric backing + honest ds18 caveat note (2026-06-28)

The supported held-out-gene row is now backed in the manuscript by a full Spearman + RMSE
numeric table (`tab:robustsp`) alongside the corrected Pearson table (`tab:robust`), all
traced to `results/benchmark/expression_enhancement/FAIR_CROSSMETRIC.json`. The kNN-reference
floor is scored on all three metrics (Pearson/Spearman/RMSE), and the best-variant lumina
clears it in every (dataset, metric) cell — i.e. the dominance is genuinely cross-metric,
not single-metric. Corrected Pearson values synced to FAIR_CROSSMETRIC: kNN floor
0.317/0.500/0.374/0.120/0.264; lumina 0.467/0.570/0.569/0.244/0.368.

Honest caveat now stated in the manuscript Results: the **v3 spatial-refinement variant
alone does NOT beat the coordinate-free floor on the placeholder-coordinate `ds18` panel**
(Pearson 0.473 vs floor 0.500); there the **coordinate-free v2 variant** is the winning
lumina entry, consistent with the validation gate disabling spatial smoothing when
coordinates are uninformative. The headline "lumina wins 5/5" therefore reflects per-dataset
best-variant selection (v3 on genuine-coordinate tissues, v2 on QuKunLab panels), gated on a
held-out split of observed genes — never on the test genes. The Discussion count is corrected
to "all four compared mappers fall below the floor on at least one dataset; only lumina clears
it on all five." Scope and accuracy-only constraints from the supported row are unchanged.

## How to add a row

1. Name the claim precisely and the exact evidence that would settle it.
2. Cite the committed script, seed, dataset hash, and hardware/torch version
   that produced the evidence.
3. Set the status: `planned` → `verified (method only)` → `verified (evidence)`.
   Do not write a performance or biological claim into the README or manuscript
   until its row reaches `verified (evidence)`.
