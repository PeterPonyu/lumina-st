"""First-party conformal-uncertainty calibration for ST imputation.

Provides per-gene split-conformal prediction intervals on top of any
BaseAdapter. The math is standard split-conformal (Vovk; Romano et al.); the
implementation is independent — no TISSUE dependency required, so the
manuscript can frame this as a native LuminaST feature that neither reference
paper offers.

Why split-conformal here:
- Distribution-free: works for any imputer that produces point estimates.
- Cell-level: each test cell gets per-gene (lower, upper) bounds, useful for
  downstream filtering.
- Calibrated coverage: at level α, the empirical fraction of held-out truth
  values inside [lower, upper] is approximately (1 - α), with a finite-sample
  correction.

Splitting strategy:
- Cells are partitioned into a calibration half and a test half. The base
  adapter sees the full AnnData (held-out genes zeroed) so it can use spatial
  context from both halves; we only split the *scoring* into cal/test.
- Per-gene residual quantile q_j is the (1 - α)(1 + 1/n_cal) quantile of
  |truth_cal - imputed_cal| restricted to gene j.
- The test prediction interval for cell i, gene j is
  [imputed[i,j] - q_j, imputed[i,j] + q_j].
"""

from __future__ import annotations

import time
from typing import Any

import anndata as ad
import numpy as np

from .contract import (
    AdapterInput,
    AdapterResult,
    BaseAdapter,
    Provenance,
    compute_imputation_metrics,
)


class ConformalCalibrator(BaseAdapter):
    """Wrap any BaseAdapter with per-gene split-conformal prediction intervals.

    Args:
        base: any BaseAdapter (mean / knn / lumina / spaim / ...).
        alpha: miscoverage level. α=0.1 ⇒ 90% intervals.
        cal_fraction: fraction of cells used for calibration (default 0.5).
    """

    name = "conformal"

    def __init__(
        self,
        base: BaseAdapter,
        alpha: float = 0.1,
        cal_fraction: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if not (0.0 < cal_fraction < 1.0):
            raise ValueError(f"cal_fraction must be in (0, 1); got {cal_fraction}")
        self.base = base
        self.alpha = alpha
        self.cal_fraction = cal_fraction
        self.name = f"conformal[{base.name}]"

    def _check_available(self) -> tuple[bool, str]:
        return self.base.is_available()

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        # The base adapter sees the same masked input the calibrator does.
        # We feed it `masked` directly (bypassing its own masked_input step,
        # since that has already happened).
        base_result = self.base._impute(masked, inp)
        if "imputed" not in base_result.layers:
            raise RuntimeError(
                f"base adapter {self.base.name} produced no 'imputed' layer"
            )
        return base_result

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

        masked = inp.masked_input()
        truth = inp.truth_matrix()

        rng = np.random.default_rng(inp.seed)
        n_cells = masked.n_obs
        n_cal = max(1, int(round(self.cal_fraction * n_cells)))
        # Random split, reproducible per seed.
        perm = rng.permutation(n_cells)
        cal_idx = perm[:n_cal]
        test_idx = perm[n_cal:]
        if test_idx.size == 0:
            return AdapterResult(
                method=self.name,
                imputed_h5ad=None,
                metrics_json={},
                provenance=Provenance.capture(self.name, inp.seed),
                status="error:too-few-cells-for-cal-test-split",
            )

        t0 = time.perf_counter()
        try:
            imputed_full = self._impute(masked, inp)
        except Exception as exc:
            return AdapterResult(
                method=self.name,
                imputed_h5ad=None,
                metrics_json={},
                provenance=Provenance.capture(self.name, inp.seed),
                status=f"error:{type(exc).__name__}: {exc}",
                runtime_s=time.perf_counter() - t0,
            )
        runtime = time.perf_counter() - t0

        X_hat = imputed_full.layers["imputed"]
        if hasattr(X_hat, "toarray"):
            X_hat = X_hat.toarray()
        X_hat = np.asarray(X_hat, dtype=np.float32)

        var_names = list(imputed_full.var_names)
        if inp.held_out_genes:
            held_cols = [var_names.index(g) for g in inp.held_out_genes if g in var_names]
        else:
            held_cols = list(range(X_hat.shape[1]))

        # Per-gene calibration: |truth - imputed| over cal cells, restricted
        # to the held-out gene set. Quantile is (1-α)(1 + 1/n_cal).
        n_cal_actual = len(cal_idx)
        q_level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / max(n_cal_actual, 1)))
        per_gene_halfwidth = np.zeros(X_hat.shape[1], dtype=np.float32)
        for j in held_cols:
            residuals = np.abs(truth[cal_idx, j] - X_hat[cal_idx, j])
            per_gene_halfwidth[j] = float(np.quantile(residuals, q_level))

        # Build lower / upper / width layers on the full AnnData (cal + test).
        lower = X_hat - per_gene_halfwidth[None, :]
        upper = X_hat + per_gene_halfwidth[None, :]
        width = upper - lower

        out = imputed_full.copy()
        out.layers["imputed_lower"] = lower
        out.layers["imputed_upper"] = upper
        out.layers["imputed_width"] = width

        # Score the base imputation on the held-out gene panel (test cells only,
        # so coverage and metrics aren't double-counting the cal set).
        truth_test = truth[test_idx]
        imputed_test_ad = ad.AnnData(X=X_hat[test_idx])
        imputed_test_ad.var_names = imputed_full.var_names
        imputed_test_ad.layers["imputed"] = X_hat[test_idx]
        metrics = compute_imputation_metrics(
            truth=truth_test,
            imputed=imputed_test_ad,
            held_out_genes=inp.held_out_genes,
        )

        # Empirical coverage on test cells over held-out genes.
        if held_cols:
            lower_test = lower[test_idx][:, held_cols]
            upper_test = upper[test_idx][:, held_cols]
            truth_test_held = truth_test[:, held_cols]
            covered = (truth_test_held >= lower_test) & (truth_test_held <= upper_test)
            metrics["empirical_coverage"] = float(np.mean(covered))
            metrics["mean_interval_width"] = float(np.mean(width[test_idx][:, held_cols]))
            metrics["alpha"] = self.alpha
            metrics["nominal_coverage"] = 1.0 - self.alpha
            metrics["n_calibration_cells"] = n_cal_actual
            metrics["n_test_cells"] = int(test_idx.size)

        return AdapterResult(
            method=self.name,
            imputed_h5ad=out,
            metrics_json=metrics,
            provenance=Provenance.capture(self.name, inp.seed),
            runtime_s=runtime,
        )
