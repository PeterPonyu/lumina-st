# Figure & table generators

Schema-driven generators that consume the aggregated benchmark results JSON and
render publication figures / tables. They are **read-only consumers** of the
results contract: every value is taken verbatim from a method's `metrics` block,
nothing is recomputed.

The generators run today on synthetic / smoke results JSON (the generation code
+ schema contract is the deliverable); when the real benchmark metrics land they
auto-render with no code change.

## Shared bundle schema

Every generator expects the bundle emitted by
`lumina_st.benchmarks.runner.aggregate_results` / `write_results_json`:

```jsonc
{
  "schema_version": "1",
  "panels": {
    "<dataset>/<panel>": {
      "<method>": {
        "status": "ok" | "unavailable:<reason>" | "error:<reason>",
        "runtime_s": 0.12,
        "metrics": { /* open metric dict — see per-generator keys below */ },
        "provenance": { /* ... */ }
      }
    }
  }
}
```

`_figbase.py` centralises loading + schema validation (`load_results`, raising
`SchemaError` on malformed input), focal-method-first ordering (`order_methods`),
the "no data / synthetic-smoke" guard (`iter_ok_records` + `require_records`),
and the output helpers (`save_fig` headless-`Agg`-guarded, `save_table` writing
CSV **and** markdown without needing matplotlib). Only `status == "ok"` records
are ever consumed.

## Generators

Each is `python scripts/visualize/<name>.py --results <bundle.json> --out <path>`
with `main(argv)` returning an exit code. Figures need matplotlib; tables (CSV +
markdown) do not.

| Script | Issue(s) | Reads from `metrics` | Emits |
| --- | --- | --- | --- |
| `fig_method_comparison.py` | (template) | `mean_pearson`, `mean_spearman`, `mean_rmse`, `zero_pattern_jaccard`, `mean_ssim`, `per_gene_*` | PNG |
| `fig_recovery_boxplots.py` | #266 | `per_gene_pearson`, `per_gene_spearman`, `per_gene_rmse` | PNG + `*_rank_aggregation.csv/.md` |
| `table_heldout_stratified.py` | #270, #292 | `per_gene_pearson/spearman/rmse`, optional `per_gene_mean_expression`, `per_gene_detection_fraction` | CSV + markdown |
| `fig_runtime_memory.py` | #267, #288 | record `runtime_s`; `metrics.n_cells`, `n_genes_scored`, `peak_memory_mb`/`peak_gpu_mem_mb` | PNG + `*_table.csv/.md` |
| `fig_clustering_concordance.py` | #287 | `<m>_raw_vs_gt`, `<m>_enhanced_vs_gt`, `<m>_delta_over_raw` for m ∈ {ari, ami, nmi, homogeneity} | PNG + `*_table.csv/.md` |
| `fig_risk_coverage.py` | #291 | `risk_coverage` array **or** `empirical_coverage`/`nominal_coverage` (+ `mean_interval_width`) | PNG + `*_calibration.csv/.md` |

Stratification bins (`table_heldout_stratified.py`): when
`per_gene_mean_expression` / `per_gene_detection_fraction` annotations are
absent, every gene falls in the single `all` bin so the table degrades
gracefully on synthetic data and auto-stratifies once the annotations ship.

## Adding a subpanel

The remaining ~30 [figure]/[table] issues are mechanical follow-ons: copy a
generator, point its metric keys at the relevant `metrics` entries, reuse
`_figbase` for loading / ordering / output, and add a tiny synthetic fixture to
`tests/visualize/test_figure_generators.py`. Keep each generator a read-only
consumer with a `main(argv)` + `--out` interface.
