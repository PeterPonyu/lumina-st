"""TISSUE adapter — conformal-uncertainty wrapper around any imputer.

Reference:
    Sun, E. D. et al. "TISSUE: uncertainty-calibrated prediction of single-cell
    spatial transcriptomics improves downstream analyses." Nat. Methods (2024).
    https://github.com/sunericd/TISSUE

TISSUE is unique among the competitors because it produces *calibrated*
prediction intervals via split-conformal inference. The adapter contract
exposes the per-gene interval width as an extra metric, in addition to the
usual point-estimate metrics, so the runner can report empirical coverage.
"""

from __future__ import annotations

from typing import Any, Optional

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter

_INSTALL_HINT = (
    "tissue-not-installed: pip install tissue-sc (see "
    "https://github.com/sunericd/TISSUE for the current install path)"
)


class TISSUEAdapter(BaseAdapter):
    """Adapter for the TISSUE conformal-uncertainty wrapper.

    Required adapter options:
        base_imputer : callable
            A callable taking (masked: AnnData, inp: AdapterInput) -> AnnData
            that produces the *point estimate* TISSUE will calibrate. Without
            this, TISSUE has nothing to wrap.
        reference_adata : ad.AnnData
            scRNA-seq reference; passed through to base_imputer if it needs one.
        alpha : float
            Miscoverage rate for conformal intervals (default 0.1 → 90% intervals).
    """

    name = "tissue"

    def __init__(
        self,
        base_imputer=None,
        reference_adata: Optional[ad.AnnData] = None,
        alpha: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_imputer = base_imputer
        self.reference_adata = reference_adata
        self.alpha = alpha

    def _check_available(self) -> tuple[bool, str]:
        import importlib

        for candidate in ("tissue", "TISSUE", "tissue_sc"):
            try:
                importlib.import_module(candidate)
                self._module_name = candidate
                break
            except ImportError:
                continue
        else:
            return False, _INSTALL_HINT

        if self.base_imputer is None:
            return False, (
                "tissue-requires-base-imputer: pass base_imputer=callable "
                "(takes masked AnnData + AdapterInput, returns AnnData with "
                ".layers['imputed']) — TISSUE only calibrates an existing model"
            )
        return True, ""

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        import importlib

        tissue = importlib.import_module(getattr(self, "_module_name", "tissue"))

        # Step 1: get point estimate from base imputer
        point = self.base_imputer(masked, inp)
        if "imputed" not in point.layers:
            raise RuntimeError(
                "base_imputer must populate .layers['imputed'] before TISSUE wraps it"
            )

        # Step 2: ask TISSUE to calibrate. The exact API varies across releases;
        # we try the canonical entry points in order.
        calibrator = None
        for attr in ("ConformalCalibrator", "TISSUE", "calibrate"):
            calibrator = getattr(tissue, attr, None)
            if calibrator is not None:
                break
        if calibrator is None:
            raise RuntimeError(
                "TISSUE module does not expose ConformalCalibrator / TISSUE / calibrate"
            )

        if callable(calibrator) and not isinstance(calibrator, type):
            # `calibrate(...)` function form
            intervals = calibrator(point=point, reference=self.reference_adata, alpha=self.alpha)
        else:
            # class form
            cal = calibrator(alpha=self.alpha)
            cal.fit(point=point, reference=self.reference_adata)
            intervals = cal.predict(point)

        out = point.copy()
        # Store calibrated lower/upper bounds and width
        if hasattr(intervals, "lower") and hasattr(intervals, "upper"):
            out.layers["imputed_lower"] = np.asarray(intervals.lower, dtype=np.float32)
            out.layers["imputed_upper"] = np.asarray(intervals.upper, dtype=np.float32)
            out.layers["imputed_width"] = np.asarray(
                intervals.upper - intervals.lower, dtype=np.float32
            )
        return out
