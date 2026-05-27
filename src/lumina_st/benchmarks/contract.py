"""Shared adapter contract for ST imputation benchmarks.

The contract is defined to satisfy three properties:

1. Comparability — every adapter receives the same input dict and returns
   the same result schema, so the runner can compare them on identical inputs.
2. Audit-safety — held-out genes are masked at the raw-input layer before
   the adapter sees them, and the truth layer is never passed in. This is
   verified by a unit test (tests/benchmarks/test_contract.py).
3. Provenance — every result records the command, seed, hardware, dependency
   notes, and (when in a git checkout) the SHA — so a metrics row is
   reproducible.

This module has no PyTorch or scanpy import at the top level; it only depends
on numpy + anndata so it can be loaded cheaply by the runner before deciding
which adapter to instantiate.
"""

from __future__ import annotations

import platform
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import anndata as ad
import numpy as np


@dataclass
class AdapterInput:
    """The shape every adapter receives. See docs/LUMINAST_PRIORITY_ENHANCEMENTS.md."""

    input_h5ad: ad.AnnData
    held_out_genes: list[str] = field(default_factory=list)
    observed_layer: Optional[str] = None  # None means use .X
    truth_layer: Optional[str] = None  # None means truth is .X (synthetic / masked-gene mode)
    seed: int = 0
    cancer_type: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def masked_input(self) -> ad.AnnData:
        """Return a copy of input_h5ad with held_out_genes zeroed in the observed layer."""
        adata = self.input_h5ad.copy()
        if not self.held_out_genes:
            return adata

        var_names = list(adata.var_names)
        present = [g for g in self.held_out_genes if g in var_names]
        if len(present) != len(self.held_out_genes):
            missing = [g for g in self.held_out_genes if g not in var_names]
            import warnings

            warnings.warn(
                f"[AdapterInput] {len(missing)}/{len(self.held_out_genes)} held-out "
                f"genes not found in adata.var_names. First 5 missing: {missing[:5]}"
            )
        if not present:
            return adata

        idx = [var_names.index(g) for g in present]

        if self.observed_layer is None or self.observed_layer not in adata.layers:
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32).copy()
            X[:, idx] = 0.0
            adata.X = X
        else:
            L = adata.layers[self.observed_layer]
            if hasattr(L, "toarray"):
                L = L.toarray()
            L = np.asarray(L, dtype=np.float32).copy()
            L[:, idx] = 0.0
            adata.layers[self.observed_layer] = L
        return adata

    def truth_matrix(self) -> np.ndarray:
        """Truth matrix for scoring; never passed to the adapter."""
        if self.truth_layer is not None and self.truth_layer in self.input_h5ad.layers:
            T = self.input_h5ad.layers[self.truth_layer]
        else:
            T = self.input_h5ad.X
        if hasattr(T, "toarray"):
            T = T.toarray()
        return np.asarray(T, dtype=np.float32)


@dataclass
class Provenance:
    method: str
    command: str = ""
    git_sha: Optional[str] = None
    seed: int = 0
    hostname: str = field(default_factory=socket.gethostname)
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    device: str = "cpu"
    dependency_notes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls, method: str, seed: int, device: str = "cpu", **extra: str) -> "Provenance":
        sha: Optional[str] = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0:
                sha = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        deps: dict[str, str] = {}
        for pkg in ("numpy", "anndata", "torch", "scanpy"):
            try:
                mod = __import__(pkg)
                deps[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                deps[pkg] = "not-installed"
        deps.update(extra)
        return cls(
            method=method,
            git_sha=sha,
            seed=seed,
            device=device,
            dependency_notes=deps,
        )


@dataclass
class AdapterResult:
    method: str
    imputed_h5ad: Optional[ad.AnnData]  # None when status != "ok"
    metrics_json: dict[str, Any]
    provenance: Provenance
    status: str = "ok"  # "ok" | "unavailable:<reason>" | "error:<reason>"
    runtime_s: float = 0.0
    peak_memory_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("imputed_h5ad")  # not JSON-serializable
        return d


class BaseAdapter(ABC):
    """Abstract adapter base. Subclasses implement `_impute` and optionally `_check_available`."""

    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs

    def is_available(self) -> tuple[bool, str]:
        """Return (available, reason). Override to record dependency unavailability."""
        try:
            return self._check_available()
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def _check_available(self) -> tuple[bool, str]:
        return True, ""

    @abstractmethod
    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        """Run the imputation. Returns adata with .layers['imputed']."""

    def run(self, inp: AdapterInput) -> AdapterResult:
        available, reason = self.is_available()
        if not available:
            return AdapterResult(
                method=self.name,
                imputed_h5ad=None,
                metrics_json={},
                provenance=Provenance.capture(self.name, inp.seed),
                status=f"unavailable:{reason}",
            )

        # Audit boundary: zero held-out genes before the adapter sees them.
        masked = inp.masked_input()

        np.random.seed(inp.seed)
        t0 = time.perf_counter()
        try:
            imputed = self._impute(masked, inp)
            runtime = time.perf_counter() - t0
            truth = inp.truth_matrix()
            metrics = compute_imputation_metrics(
                truth=truth,
                imputed=imputed,
                held_out_genes=inp.held_out_genes,
            )
        except Exception as exc:
            return AdapterResult(
                method=self.name,
                imputed_h5ad=None,
                metrics_json={},
                provenance=Provenance.capture(self.name, inp.seed),
                status=f"error:{type(exc).__name__}: {exc}",
                runtime_s=time.perf_counter() - t0,
            )

        return AdapterResult(
            method=self.name,
            imputed_h5ad=imputed,
            metrics_json=metrics,
            provenance=Provenance.capture(self.name, inp.seed),
            runtime_s=runtime,
        )


def compute_imputation_metrics(
    truth: np.ndarray,
    imputed: ad.AnnData,
    held_out_genes: list[str],
) -> dict[str, Any]:
    """Per-gene and overall scoring under the held-out-gene benchmark contract.

    Pearson and Spearman are computed per gene (column-wise); RMSE is computed
    on z-score-normalized values; sparsity match is the per-cell Jaccard of
    zero patterns.

    All metrics are computed only on `held_out_genes` when provided; otherwise
    on every gene.
    """
    if "imputed" in imputed.layers:
        X_hat = imputed.layers["imputed"]
    else:
        X_hat = imputed.X
    if hasattr(X_hat, "toarray"):
        X_hat = X_hat.toarray()
    X_hat = np.asarray(X_hat, dtype=np.float32)

    if X_hat.shape != truth.shape:
        raise ValueError(
            f"Imputed shape {X_hat.shape} does not match truth shape {truth.shape}"
        )

    var_names = list(imputed.var_names)
    if held_out_genes:
        cols = [var_names.index(g) for g in held_out_genes if g in var_names]
        if not cols:
            raise ValueError(
                f"None of the {len(held_out_genes)} requested held_out_genes are present "
                f"in imputed.var_names; requested={held_out_genes[:5]}..., "
                f"available={var_names[:5]}..."
            )
    else:
        cols = list(range(truth.shape[1]))

    per_gene_pearson: dict[str, float] = {}
    per_gene_spearman: dict[str, float] = {}
    per_gene_rmse: dict[str, float] = {}

    for j in cols:
        t = truth[:, j].astype(np.float64)
        h = X_hat[:, j].astype(np.float64)

        # Pearson (handle zero variance)
        if t.std() > 1e-9 and h.std() > 1e-9:
            per_gene_pearson[var_names[j]] = float(np.corrcoef(t, h)[0, 1])
        else:
            per_gene_pearson[var_names[j]] = float("nan")

        # Spearman via rank correlation
        try:
            from scipy.stats import spearmanr  # type: ignore

            rho, _ = spearmanr(t, h)
            per_gene_spearman[var_names[j]] = float(rho) if not np.isnan(rho) else float("nan")
        except ImportError:
            tr = _rank(t)
            hr = _rank(h)
            if tr.std() > 1e-9 and hr.std() > 1e-9:
                per_gene_spearman[var_names[j]] = float(np.corrcoef(tr, hr)[0, 1])
            else:
                per_gene_spearman[var_names[j]] = float("nan")

        # RMSE on z-normalized values (cross-platform-invariant convention)
        tn = _zscore(t)
        hn = _zscore(h)
        per_gene_rmse[var_names[j]] = float(np.sqrt(np.mean((tn - hn) ** 2)))

    pearson_vals = [v for v in per_gene_pearson.values() if not np.isnan(v)]
    spearman_vals = [v for v in per_gene_spearman.values() if not np.isnan(v)]
    rmse_vals = list(per_gene_rmse.values())

    sparsity_truth = float(np.mean(truth == 0))
    sparsity_imputed = float(np.mean(X_hat == 0))
    zero_match = float(np.mean((truth == 0) == (X_hat == 0)))

    return {
        "per_gene_pearson": per_gene_pearson,
        "per_gene_spearman": per_gene_spearman,
        "per_gene_rmse": per_gene_rmse,
        "mean_pearson": float(np.mean(pearson_vals)) if pearson_vals else float("nan"),
        "mean_spearman": float(np.mean(spearman_vals)) if spearman_vals else float("nan"),
        "mean_rmse": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
        "sparsity_truth": sparsity_truth,
        "sparsity_imputed": sparsity_imputed,
        "zero_pattern_jaccard": zero_match,
        "n_genes_scored": len(cols),
        "n_cells": int(truth.shape[0]),
    }


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    if sd < 1e-9:
        return x - x.mean()
    return (x - x.mean()) / sd


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x))
    return ranks
