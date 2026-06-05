# Experiment Harness & Run Manifest

Tracking issues: **#179** (unified harness + run-manifest contract) and **#181**
(first concrete slice: emit a minimal manifest from the core paths).

This is reproducibility plumbing put in place **before** real-data training runs,
so every run is self-describing from day one. The first artifact is the
`RunManifest` (`lumina_st.experiment.run_manifest`): a flat, JSON-serializable
snapshot written next to a run's outputs/checkpoint.

## Why a dataclass (not pydantic)

`LuminaSTConfig` is pydantic because it validates/coerces user input. A manifest
is the opposite: a write-once provenance record that must serialize trivially and
never reject. It therefore mirrors the light contract modules (`repro.py`,
`results_contract.py`) as a plain `@dataclass`. It does **not** duplicate config
or seeding — it *snapshots* `LuminaSTConfig` (via `model_dump_for_checkpoint`) and
records the seed that `repro.set_seed` / `pl.seed_everything` already applied.

## Schema (`RunManifest`)

| Field | Type | Source / meaning |
| --- | --- | --- |
| `run_id` | `str` | Caller-chosen unique id (core paths use `"{kind}-{experiment_name}"`). |
| `timestamp` | `str` | ISO-8601 UTC. **Injected** by the caller; `create` reads the clock once at call time (never at import) so tests pin it. |
| `git_commit` | `str` | Short SHA via read-only `results_contract.git_sha` (resolves the package repo from `__file__`). `"unknown"` off-git; `"ambiguous-parent-checkout"` from the parent repo. |
| `seed` | `int \| None` | Applied seed; defaults to `config.seed`. |
| `seeds` | `dict[str,int]` | Named seeds (currently `{"global": seed}`); room for per-component seeds. |
| `python_version` | `str` | `platform.python_version()`. |
| `torch_version` | `str` | `torch.__version__` or `"absent"`. |
| `numpy_version` | `str` | `numpy.__version__` or `"absent"`. |
| `device` | `str` | `cpu` or `cuda:<name>` (best-effort; never raises). |
| `config` | `dict` | Snapshot of the `LuminaSTConfig` used (`model_dump_for_checkpoint`, JSON-safe). |
| `sweep_params` | `dict` | Sweep coordinate / ablation point; core paths also stash `run_kind`, `n_obs`, `n_vars`. |
| `logging_config` | `dict` | Logging backend config; defaults surface `use_wandb` / `wandb_project`. |
| `checkpoint_path` | `str \| None` | Where the run writes/resolves its checkpoint. |
| `resume_from` | `str \| None` | Checkpoint resumed from, if any. |
| `eval_cadence` | `dict` | How often eval runs (e.g. `early_stopping_patience`, `every_n_epochs`). |
| `dataset_id` | `str \| None` | Stable dataset identifier (defaults to `experiment_name`). `None` pre-data is fine. |
| `dataset_hash` | `str \| None` | Best-effort content hash (`dataset_hash_for(path)`); `None` when unavailable. |
| `manifest_version` | `str` | Schema version of the manifest contract (`1.0.0`). |

Methods: `RunManifest.create(...)` (captures the environment), `to_dict()`,
`to_json(path)` (atomic temp + `replace`), `from_json(path)` — round-trip exact.

## Where it is emitted (#181 concrete slice)

- **`LuminaImputer.enhance(..., run_dir=None, run_manifest_timestamp=None)`** —
  when `run_dir` is given, writes `run_dir/run_manifest.json` after producing the
  enhanced AnnData. Default `run_dir=None` emits nothing, preserving every
  existing call site and the return value.
- **`LuminaImputer.fit(..., run_dir=None, run_manifest_timestamp=None,
  resume_from=None)`** — writes `run_manifest.json` **before** `trainer.fit` so a
  crashed run is still self-describing. `run_dir` defaults to
  `config.output_dir`. `checkpoint_path` points at `output_dir/checkpoints`.

Emission is **best-effort**: any failure (e.g. read-only dir) is caught, warns,
and never changes results or raises. Manifest creation does no I/O on the model.

## How downstream consumers are intended to use it

- **Resume / checkpointing:** `checkpoint_path` + `resume_from` let a relaunch
  locate the prior checkpoint; `fit` already wires `ModelCheckpoint` into
  `output_dir/checkpoints`, which the manifest records.
- **Eval cadence:** `eval_cadence` is the single place a scheduler reads to decide
  validation frequency; today it carries `early_stopping_patience`.
- **Sweeps:** a sweep driver sets a distinct `run_id` + `sweep_params` per grid
  point and points each run at its own `run_dir`; the manifests are the
  per-trial provenance ledger.

## Deferred (out of scope for #181)

- A sweep/grid runner that iterates `sweep_params` and launches trials.
- A scheduler that *consumes* `eval_cadence` (the field is recorded, not yet read
  by training loops).
- Dataset hashing on the real-data path (`dataset_hash_for` exists; wiring it to
  the actual atlas/ST inputs lands with real data).
- Per-component seed streams beyond `seeds["global"]`.
- A central `ExperimentHarness` orchestrator object — the manifest is the
  contract; the orchestrator is future work.
