# META Contract Proposals (lumina-st L1 → META)

The canonical results schema (`src/lumina_st/results_contract.py`) is META-owned and
byte-locked. L1 sessions must not edit it or its `.sha256` sidecar. This file collects
proposals from the lumina-st L1 lane for **new first-class contract fields** that META may
consider adopting (and then propagating across all four repos via `check_sync.py`).

Until a proposal is accepted, the corresponding gate is implemented using the OPEN
`metrics: dict[str, float | None]` field and/or adapter-level metadata in the benchmark
layer (`benchmarks/contract.py`), which is *not* the byte-locked schema.

---

## Proposal 1 — first-class `task_type` field on a result row

- **Raised by:** lumina-st L1, for GitHub issues #307 / #309.
- **Status:** PROPOSED (not implemented in the contract). Interim implementation shipped in
  the benchmark layer — see "Interim implementation" below.

### Field

| name        | type  | required | allowed values                                     |
|-------------|-------|----------|----------------------------------------------------|
| `task_type` | `str` (enum) | yes | `gene_recovery`, `denoising`, `pathway_aggregate` |

### Rationale

Denoising and pathway/aggregate tasks are **not** held-out-gene recovery. Today a result
row records imputation metrics (Pearson/Spearman/RMSE over a held-out panel) with no
machine-checkable statement of *which task produced those numbers*. That makes it possible
for a denoising or pathway-aggregate result to be silently displayed in a figure or table as
if it were gene recovery — an apples-to-oranges comparison that would invalidate the headline
benchmark claim.

A required `task_type` column makes the boundary load-bearing at the **persisted** layer: any
consumer (figure scripts, cross-repo aggregation, leakage audits) can assert
`row.task_type == "gene_recovery"` before treating Pearson/RMSE as recovery quality, and the
byte-locked contract would guarantee the field is always present and constrained to the enum.

This also reinforces the #307 parity guarantee: a comparison table is only valid when every
row in it shares one `task_type` *and* one held-out protocol. With `task_type` as a real
column, that invariant is checkable without reaching into the open `metrics` dict.

### Why it is not implemented here

Adding a required column is a structural schema change and would alter the byte-lock SHA, which
is META's sole authority to change and propagate. Per the L1 boundary, lumina-st does not touch
`results_contract.py` or its sidecar.

### Interim implementation (already shipped on `feat/lumina-st-307-309-protocol-gates`)

Implemented entirely in the benchmark layer, with no change to the byte-locked schema:

- `benchmarks/contract.py`
  - `TaskType` enum (`gene_recovery` | `denoising` | `pathway_aggregate`).
  - `AdapterInput.task_type` (defaults to `gene_recovery`) and `BaseAdapter.task_type`
    class attribute (defaults to `gene_recovery`; a non-recovery adapter must override it).
  - `compute_imputation_metrics(..., task_type=...)` — the single gene-recovery scorer —
    raises `TaskBoundaryError` for any non-`gene_recovery` task (#309), and stamps the open
    metrics dict with `"task_type": "gene_recovery"` so a persisted row self-identifies.
  - `AdapterInput.assert_no_encoder_leakage()` — fail-closed `EncoderLeakageError` if a
    held-out gene is absent from `var_names` (cannot be masked), the named `observed_layer`
    is missing (adapter would read an unmasked layer), or the held-out columns are not
    actually zeroed in the layer the adapter conditions on (#307 encoder-leakage guard).
  - `AdapterInput.protocol_signature()` — canonical, order-insensitive
    `(sorted held-out genes, observed_layer, truth_layer, task_type)` used to assert parity.
- `benchmarks/runner.py` — `enforce_protocol_parity()` enforces all gates centrally before any
  adapter runs (single shared input → identical held-out split/scoring genes/masking),
  rejecting non-recovery adapters and empty held-out sets.
- `benchmarks/cross_validation.py` — `run_cv` rejects a factory that returns a non-recovery
  adapter and runs the leakage guard per fold; all folds share one panel by construction.

The interim `task_type` lives in the benchmark `metrics_json` dict (value is a string), which
is the benchmark module's own open dict — distinct from the byte-locked `results_contract`
`metrics: dict[str, float | None]`. If META adopts the first-class field, the gate can be
re-pointed at the contract column with no behavioral change.
