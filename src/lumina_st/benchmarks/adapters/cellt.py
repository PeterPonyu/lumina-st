"""CellT adapter — spatial-tokens transformer for ST imputation.

Reference:
    "Single-cells are Spatial Tokens: Transformers for Spatially Resolved
    Transcriptomics Data Imputation."
    https://github.com/wehos/CellT

Pure attention baseline; useful for showing flow-matching vs feed-forward
attention behavior on the same data.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter

_INSTALL_HINT = (
    "cellt-not-installed: clone https://github.com/wehos/CellT and install "
    "requirements; adapter expects a `cellt` importable module"
)


class CellTAdapter(BaseAdapter):
    name = "cellt"

    def __init__(self, device: str = "cpu", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.device = device

    def _check_available(self) -> tuple[bool, str]:
        import importlib

        for candidate in ("cellt", "CellT", "cell_t"):
            try:
                importlib.import_module(candidate)
                self._module_name = candidate
                return True, ""
            except ImportError:
                continue
        return False, _INSTALL_HINT

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        import importlib

        cellt = importlib.import_module(getattr(self, "_module_name", "cellt"))

        model_cls = None
        for cls_name in ("CellT", "CellTransformer", "Model"):
            model_cls = getattr(cellt, cls_name, None)
            if model_cls is not None:
                break
        if model_cls is None:
            raise RuntimeError(
                "cellt module does not expose CellT / CellTransformer / Model class"
            )

        model = model_cls(device=self.device)
        model.fit(masked)
        predicted = model.predict(masked)

        if isinstance(predicted, ad.AnnData):
            if "imputed" not in predicted.layers:
                predicted.layers["imputed"] = np.asarray(predicted.X, dtype=np.float32)
            return predicted

        pred_arr = np.asarray(predicted, dtype=np.float32)
        if pred_arr.shape != masked.X.shape:
            raise RuntimeError(
                f"CellT produced shape {pred_arr.shape}; expected {masked.X.shape}"
            )
        out = masked.copy()
        out.layers["imputed"] = pred_arr
        return out
