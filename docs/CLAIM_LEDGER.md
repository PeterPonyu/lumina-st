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
| Clustering (ARI/NMI) on the enhanced latent matches or exceeds the observed latent. | Independent-graph clustering benchmark on real annotated data. | Method wired (`EnhancementEvaluator`); no real-data result. | Real annotated run + logged metrics. | planned |
| Platform-transfer / FFPE-vs-frozen robustness. | Stratified multi-platform benchmark with provenance. | None. | Data cards + stratified run. | planned |

## How to add a row

1. Name the claim precisely and the exact evidence that would settle it.
2. Cite the committed script, seed, dataset hash, and hardware/torch version
   that produced the evidence.
3. Set the status: `planned` → `verified (method only)` → `verified (evidence)`.
   Do not write a performance or biological claim into the README or manuscript
   until its row reaches `verified (evidence)`.
