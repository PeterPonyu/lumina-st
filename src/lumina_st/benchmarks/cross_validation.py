"""Leave-one-context cross-validation for LuminaST imputation benchmarks.

Tests the pan-cancer / zero-shot generalization claim in the manuscript: for
each held-out context (cancer type or platform), the adapter must not have
seen any of that context's data at training time. The adapter-factory pattern
makes that contract explicit: the caller supplies a function that builds the
adapter for a given holdout, receiving the *other* contexts as training data
(or ignoring them, for stateless baselines).

The harness is intentionally separable from training:
- Stateless baselines (mean, knn) work out of the box — factory just returns
  the adapter; training data is unused.
- Trainable adapters (lumina, spaim) use the `train_contexts` dict the
  factory receives. If LuminaImputer.fit() exists, the factory calls it.

Bootstrap confidence intervals across folds provide an honest summary even
when only a handful of contexts (e.g., 6 cancer types) are available — the
manuscript can report mean ± 95% CI for pan-cancer Pearson, etc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import anndata as ad
import numpy as np

from .contract import AdapterInput, BaseAdapter
from .panels import MarkerPanel


AdapterFactory = Callable[[str, dict[str, ad.AnnData]], BaseAdapter]


@dataclass
class CVFold:
    """One leave-one-out fold."""

    test_context: str
    train_contexts: tuple[str, ...]
    result: dict[str, Any]  # AdapterResult.to_dict() of the per-fold scoring
    runtime_s: float = 0.0


def run_cv(
    contexts: dict[str, ad.AnnData],
    adapter_factory: AdapterFactory,
    panel: MarkerPanel,
    seed: int = 0,
) -> list[CVFold]:
    """Run leave-one-context CV across every context.

    Args:
        contexts: mapping {context_name → AnnData}. Typical use:
            {"COAD": adata_coad, "OV": adata_ov, ...}
        adapter_factory: callable(test_context, others) → BaseAdapter. For
            stateless baselines this just returns a fresh adapter; for
            stateful ones it fits on `others` first.
        panel: held-out gene panel used at evaluation time.
        seed: random seed.

    Returns:
        list of CVFold, one per context.
    """
    folds: list[CVFold] = []
    context_names = list(contexts.keys())
    for test_name in context_names:
        train_names = tuple(c for c in context_names if c != test_name)
        train_dict = {c: contexts[c] for c in train_names}

        t0 = time.perf_counter()
        adapter = adapter_factory(test_name, train_dict)

        inp = AdapterInput(
            input_h5ad=contexts[test_name],
            held_out_genes=list(panel.genes),
            seed=seed,
            cancer_type=test_name,
            extra={
                "context": test_name,
                "n_train_contexts": len(train_names),
                "panel": panel.name,
            },
        )
        result = adapter.run(inp)
        runtime = time.perf_counter() - t0

        folds.append(CVFold(
            test_context=test_name,
            train_contexts=train_names,
            result=result.to_dict(),
            runtime_s=runtime,
        ))
    return folds


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap-resample `values` and return point estimate + (1-α) CI.

    For a leave-one-context CV with K folds, resamples K folds with
    replacement n_boot times and reports the percentile CI of the statistic.
    """
    vals = np.asarray([v for v in values if not (isinstance(v, float) and np.isnan(v))],
                      dtype=np.float64)
    if vals.size == 0:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    n = vals.size
    for i in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        boots[i] = float(statistic(sample))
    return {
        "point": float(statistic(vals)),
        "lower": float(np.quantile(boots, alpha / 2)),
        "upper": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(n),
    }


def summarize_cv(
    folds: Sequence[CVFold],
    metric_keys: Sequence[str] = ("mean_pearson", "mean_spearman", "mean_rmse"),
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate fold-level metrics into bootstrap CIs.

    Returns:
        {
          "schema_version": "1",
          "n_folds": K,
          "fold_contexts": [...],
          "per_metric": {
            "mean_pearson": {"point": ..., "lower": ..., "upper": ..., "per_fold": [...]},
            ...
          },
          "per_fold_status": {"COAD": "ok", "OV": "unavailable:...", ...}
        }
    """
    out: dict[str, Any] = {
        "schema_version": "1",
        "n_folds": len(folds),
        "fold_contexts": [f.test_context for f in folds],
        "per_metric": {},
        "per_fold_status": {f.test_context: f.result.get("status", "unknown") for f in folds},
        "per_fold_runtime_s": {f.test_context: float(f.runtime_s) for f in folds},
    }
    for key in metric_keys:
        per_fold: list[float] = []
        for f in folds:
            mj = f.result.get("metrics_json", {}) or {}
            v = mj.get(key, float("nan"))
            per_fold.append(float(v))
        ci = bootstrap_ci(per_fold, statistic=np.mean, n_boot=n_boot, alpha=alpha, seed=seed)
        ci["per_fold"] = per_fold
        out["per_metric"][key] = ci
    return out


def aggregate_cv_results(
    folds: Sequence[CVFold],
    panel_name: str,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON-serializable aggregation of CV folds + bootstrap summary."""
    out: dict[str, Any] = {
        "schema_version": "1",
        "panel": panel_name,
        "n_folds": len(folds),
        "folds": [
            {
                "test_context": f.test_context,
                "train_contexts": list(f.train_contexts),
                "runtime_s": float(f.runtime_s),
                "result": f.result,
            }
            for f in folds
        ],
    }
    if summary is not None:
        out["summary"] = summary
    return out


def stateless_factory(adapter_class: type[BaseAdapter], **kwargs: Any) -> AdapterFactory:
    """Convenience factory for stateless adapters (mean, knn, ...).

    Example:
        factory = stateless_factory(KNNAdapter, k=10)
        folds = run_cv(contexts, factory, panel)
    """
    def _factory(test_context: str, train_contexts: dict[str, ad.AnnData]) -> BaseAdapter:
        del test_context, train_contexts  # stateless: no training data needed
        return adapter_class(**kwargs)
    return _factory
