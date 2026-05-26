"""stMCDI adapter — masked conditional diffusion + GNN for ST imputation.

Reference:
    Anonymous (lllxxyyy-lxy GitHub). "stMCDI: Masked Conditional Diffusion
    Model with Graph Neural Network for Spatial Transcriptomics Data
    Imputation."
    https://github.com/lllxxyyy-lxy/stMCDI

Closest diffusion competitor to LuminaST architecturally; useful for showing
flow-matching vs DDPM-style imputation.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter

_INSTALL_HINT = (
    "stmcdi-not-installed: clone https://github.com/lllxxyyy-lxy/stMCDI and "
    "install requirements; adapter expects an `stmcdi` importable module"
)


class STMCDIAdapter(BaseAdapter):
    name = "stmcdi"

    def __init__(self, n_steps: int = 1000, device: str = "cpu", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_steps = n_steps
        self.device = device

    def _check_available(self) -> tuple[bool, str]:
        import importlib

        for candidate in ("stmcdi", "stMCDI", "st_mcdi"):
            try:
                importlib.import_module(candidate)
                self._module_name = candidate
                return True, ""
            except ImportError:
                continue
        return False, _INSTALL_HINT

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        import importlib

        stmcdi = importlib.import_module(getattr(self, "_module_name", "stmcdi"))

        model_cls = None
        for cls_name in ("stMCDI", "STMCDI", "Model"):
            model_cls = getattr(stmcdi, cls_name, None)
            if model_cls is not None:
                break
        if model_cls is None:
            raise RuntimeError(
                "stmcdi module does not expose stMCDI / STMCDI / Model class"
            )

        model = model_cls(n_steps=self.n_steps, device=self.device)
        model.fit(masked)
        predicted = model.impute(masked)

        if isinstance(predicted, ad.AnnData):
            if "imputed" not in predicted.layers:
                predicted.layers["imputed"] = np.asarray(predicted.X, dtype=np.float32)
            return predicted

        pred_arr = np.asarray(predicted, dtype=np.float32)
        if pred_arr.shape != masked.X.shape:
            raise RuntimeError(
                f"stMCDI produced shape {pred_arr.shape}; expected {masked.X.shape}"
            )
        out = masked.copy()
        out.layers["imputed"] = pred_arr
        return out
