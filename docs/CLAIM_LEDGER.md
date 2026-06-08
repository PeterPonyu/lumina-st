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
| Pan-cancer enhancement improves gene recovery vs observed/baseline on real ST. | Real COAD/pan-cancer run; held-out-gene Pearson/SSIM vs baselines; seed + dataset hash + hardware. | Synthetic/local-small smoke only. | Real-data run + logged metrics. | planned |
| Zero-shot generalization to held-out cancer types without paired reference. | Multi-cancer held-out sweep, no-retraining gate, per-context metrics. | None. | Implementation of the gate + real data. | planned |
| Clustering (ARI/NMI) on the enhanced latent matches or exceeds the observed latent. | Independent-graph clustering benchmark on real annotated data. | Method wired (`EnhancementEvaluator`). **In-tissue matched run (2026-06-07), no native GT → transferred-label proxy** (`scripts/uplift_transferred_labels_multiseed.py`; COAD Visium `coad_visium_10x` × matched COAD scRNA ref `lumina_ref_coad_gse132465`; enhanced `results/lumina-st/lumina_enhanced_coad_visium.h5ad` @ git_sha 678b3b3; 5 seeds; RTX 5090 / torch 2.9.0+cu130): **Δ AMI −0.0309±0.0062, NMI −0.0327±0.0058, Homo −0.1290±0.0092, Silhouette −0.0614; ARI +0.0077±0.0085 (within noise, std>mean, and confounded)**. Negative control (permuted labels) collapses every Δ to \|Δ\|≤0.005 → harness sound, deltas are real signal. This **in-tissue** pairing **replaces** the earlier cross-tissue cervix→COAD control and **corroborates** the architectural-washout finding (#296-closed; #336). CIRCULAR (labels transferred from the same reference the enhancement is conditioned on): a negative Δ is trustworthy, a positive Δ is not promotable. | **LOCKED — three gates before any positive clustering claim graduates:** (1) native-GT verdict on `st_COAD_test` (#336) goes positive (removes circular-label confound); (2) a non-reference-derived label source also shows positive Δ; (3) the protocol/clustering gate (#308) passes on the matched run. Final promotion-gate sign-off is HUMAN-GATED. | planned (locked: honest-negative corroboration; not promotable) |
| Platform-transfer / FFPE-vs-frozen robustness. | Stratified multi-platform benchmark with provenance. | None. | Data cards + stratified run. | planned |

## How to add a row

1. Name the claim precisely and the exact evidence that would settle it.
2. Cite the committed script, seed, dataset hash, and hardware/torch version
   that produced the evidence.
3. Set the status: `planned` → `verified (method only)` → `verified (evidence)`.
   Do not write a performance or biological claim into the README or manuscript
   until its row reaches `verified (evidence)`.
