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
from enum import Enum
from typing import Any, Optional

import anndata as ad
import numpy as np


class TaskType(str, Enum):
    """The measurement an adapter/result represents (issue #309).

    Task boundaries are *load-bearing*: a denoising or pathway/aggregate result
    is NOT a held-out-gene-recovery measurement and must never be silently
    scored or reported as one. The gene-recovery scorer
    (`compute_imputation_metrics`) and the runner reject any task that is not
    ``GENE_RECOVERY``, so a mislabeled result fails closed instead of producing
    a misleading Pearson/Spearman row.
    """

    GENE_RECOVERY = "gene_recovery"
    DENOISING = "denoising"
    PATHWAY_AGGREGATE = "pathway_aggregate"


class TaskBoundaryError(ValueError):
    """Raised when a non-gene-recovery task is routed to gene-recovery scoring (#309)."""


class EncoderLeakageError(ValueError):
    """Raised when a held-out gene could structurally leak into the encoder input (#307).

    Fail-closed: if the held-out genes cannot be proven absent from every input
    the adapter conditions on (i.e., not actually zeroed in the layer the
    adapter reads), we refuse to run rather than emit a leakage-contaminated
    recovery number.
    """


class ProtocolParityError(ValueError):
    """Raised when adapters in one comparison do not share one held-out protocol (#307)."""


@dataclass
class AdapterInput:
    """The shape every adapter receives (the shared adapter contract)."""

    input_h5ad: ad.AnnData
    held_out_genes: list[str] = field(default_factory=list)
    observed_layer: Optional[str] = None  # None means use .X
    truth_layer: Optional[str] = None  # None means truth is .X (synthetic / masked-gene mode)
    seed: int = 0
    cancer_type: Optional[str] = None
    # Which measurement this input represents. GENE_RECOVERY is the only task
    # the held-out scorer / runner will score; anything else fails closed at the
    # task boundary (issue #309).
    task_type: TaskType = TaskType.GENE_RECOVERY
    extra: dict[str, Any] = field(default_factory=dict)

    def protocol_signature(self) -> tuple:
        """Canonical, order-insensitive description of the held-out protocol.

        Two inputs sharing this signature are scored under an IDENTICAL protocol:
        the same held-out split (sorted gene set), the same observed/truth layer
        wiring (= same masking), and the same task type. The runner uses this to
        enforce cross-adapter parity centrally rather than per-adapter (#307).
        """
        return (
            tuple(sorted(self.held_out_genes)),
            self.observed_layer,
            self.truth_layer,
            self.task_type.value,
        )

    def assert_no_encoder_leakage(self) -> None:
        """Fail-closed guard that held-out genes cannot leak into the encoder (#307).

        The adapter conditions only on the layer returned by ``masked_input()``
        (``.X`` or ``observed_layer``). Recovery is only honest if every held-out
        gene is (a) present in ``var_names`` so it *can* be masked, and (b)
        actually zeroed in that observed layer. If either is structurally
        violated, raise ``EncoderLeakageError`` instead of running.
        """
        if not self.held_out_genes:
            return

        var_names = list(self.input_h5ad.var_names)
        missing = [g for g in self.held_out_genes if g not in var_names]
        if missing:
            raise EncoderLeakageError(
                "Held-out gene(s) not in adata.var_names, so they cannot be "
                f"masked before the encoder sees the input: {missing[:5]}"
                f"{' ...' if len(missing) > 5 else ''}. A gene the masker cannot "
                "reach is a structural leakage path — refusing to run."
            )

        masked = self.masked_input()
        if self.observed_layer is not None and self.observed_layer not in masked.layers:
            raise EncoderLeakageError(
                f"observed_layer {self.observed_layer!r} is absent from the masked "
                "input, so the adapter would fall back to an UNMASKED layer — "
                "held-out genes could leak into the encoder. Refusing to run."
            )

        if self.observed_layer is not None:
            obs = masked.layers[self.observed_layer]
        else:
            obs = masked.X
        if hasattr(obs, "toarray"):
            obs = obs.toarray()
        obs = np.asarray(obs)
        idx = [var_names.index(g) for g in self.held_out_genes]
        leaked = obs[:, idx]
        if not np.all(leaked == 0.0):
            max_abs = float(np.abs(leaked).max())
            raise EncoderLeakageError(
                "Held-out gene columns are non-zero in the layer the adapter "
                f"conditions on (max abs value {max_abs:g}). The encoder could "
                "read held-out expression — refusing to run (#307)."
            )

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
    # The measurement this adapter produces. Every concrete adapter under
    # adapters/ is a held-out-gene imputer, so the default is GENE_RECOVERY. A
    # denoising / pathway-aggregate adapter MUST override this so its result can
    # never be silently scored as gene recovery (issue #309).
    task_type: TaskType = TaskType.GENE_RECOVERY

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

        # Task-boundary gate (#309): the adapter's declared task must match the
        # input's task, and only GENE_RECOVERY reaches the gene-recovery scorer.
        # These are protocol violations, not runtime failures, so they are
        # raised (fail-closed) rather than captured as an "error:" status.
        if inp.task_type != self.task_type:
            raise TaskBoundaryError(
                f"Adapter {self.name!r} declares task {self.task_type.value!r} but was "
                f"handed an input tagged {inp.task_type.value!r}. A result can never be "
                "scored under a task it was not produced for (#309)."
            )

        # Encoder-leakage gate (#307): prove held-out genes are zeroed in the
        # layer the adapter conditions on before it ever sees the input.
        inp.assert_no_encoder_leakage()

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
                task_type=inp.task_type,
            )
        except (TaskBoundaryError, EncoderLeakageError):
            raise
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
    task_type: TaskType = TaskType.GENE_RECOVERY,
) -> dict[str, Any]:
    """Per-gene and overall scoring under the held-out-gene benchmark contract.

    Pearson and Spearman are computed per gene (column-wise); RMSE is computed
    on raw values (so it reflects absolute magnitude error rather than being a
    rescaling of Pearson); sparsity match is the per-cell Jaccard of zero
    patterns.

    All metrics are computed only on `held_out_genes` when provided; otherwise
    on every gene.

    This function is the gene-recovery scorer and is the single chokepoint where
    the task boundary is enforced (#309): it refuses to score anything that is
    not ``TaskType.GENE_RECOVERY`` so a denoising / pathway-aggregate result can
    never be silently reported as gene recovery.
    """
    if task_type != TaskType.GENE_RECOVERY:
        raise TaskBoundaryError(
            f"compute_imputation_metrics is the gene-recovery scorer but was asked to "
            f"score task {task_type.value!r}. Denoising / pathway-aggregate tasks are "
            "NOT gene recovery and must use their own scorer (#309)."
        )
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

        # RMSE on RAW values. Z-scoring both vectors before the RMSE made it
        # an exact affine-invariant function of Pearson r — sqrt(2*(1 - r)) —
        # so mean_rmse carried no information beyond mean_pearson (issue #133).
        # Raw RMSE captures absolute magnitude / scale error that Pearson, by
        # construction, is blind to.
        per_gene_rmse[var_names[j]] = float(np.sqrt(np.mean((t - h) ** 2)))

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
        # Constant / zero-variance genes have undefined correlation and are
        # dropped from mean_pearson / mean_spearman. Expose the true denominator
        # of each correlation mean so it cannot be silently confused with
        # n_genes_scored, which counts every attempted held-out gene (issue #140).
        "n_genes_pearson_scored": len(pearson_vals),
        "n_genes_spearman_scored": len(spearman_vals),
        "n_cells": int(truth.shape[0]),
        # Tag the measurement so a downstream reader can never mistake this row
        # for a non-gene-recovery task (#309). This is a metadata string in the
        # benchmark's open metrics dict, NOT a field on the byte-locked
        # results_contract schema.
        "task_type": TaskType.GENE_RECOVERY.value,
    }


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x))
    return ranks
