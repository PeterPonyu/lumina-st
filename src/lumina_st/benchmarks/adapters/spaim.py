"""SpaIM head-to-head adapter.

Wraps the external SpaIM package (Song lab, Nat Comms 2025) into the
BaseAdapter contract. When SpaIM is not installed on this machine, the
adapter records `status='unavailable:spaim-not-installed'` with the install
hint, so the runner can include the row in the comparison table.

Reference:
    Song et al. "SpaIM: single-cell spatial transcriptomics imputation via
    style transfer." Nat. Commun. 16 (2025).
    https://github.com/QSong-github/SpaIM
"""

from __future__ import annotations

from typing import Any, Optional

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter

_INSTALL_HINT = (
    "spaim-not-installed: pip install git+https://github.com/QSong-github/SpaIM "
    "(adapter expects the public SpaIM API; consult upstream README for the "
    "current install path)"
)


class SpaIMAdapter(BaseAdapter):
    """Adapter for the SpaIM style-transfer ST imputer.

    Required adapter option:
        reference_adata : ad.AnnData
            scRNA-seq reference covering the gene panel; SpaIM uses style
            transfer between reference and ST.
    """

    name = "spaim"

    def __init__(
        self,
        reference_adata: Optional[ad.AnnData] = None,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.reference_adata = reference_adata
        self.device = device

    def _check_available(self) -> tuple[bool, str]:
        try:
            import importlib

            for candidate in ("SpaIM", "spaim"):
                try:
                    importlib.import_module(candidate)
                    self._module_name = candidate
                    break
                except ImportError:
                    continue
            else:
                return False, _INSTALL_HINT
        except Exception as exc:
            return False, f"spaim-check-error: {type(exc).__name__}: {exc}"

        if self.reference_adata is None:
            return False, (
                "spaim-requires-reference: pass reference_adata= (scRNA-seq "
                "AnnData covering the ST gene panel) to SpaIMAdapter"
            )

        return True, ""

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        import importlib

        module_name = getattr(self, "_module_name", "SpaIM")
        spaim_module = importlib.import_module(module_name)

        # SpaIM upstream exposes (per its public README) a model class plus
        # `.fit()` and `.predict()` methods that take AnnData objects. The
        # exact class name has varied across releases; we try the canonical
        # names in order and surface the first that matches.
        model_class = None
        for cls_name in ("SpaIM", "SpaIMModel", "Spaim"):
            cls = getattr(spaim_module, cls_name, None)
            if cls is not None:
                model_class = cls
                break
        if model_class is None:
            raise RuntimeError(
                f"Module {module_name!r} does not expose a known SpaIM class; "
                f"checked: SpaIM, SpaIMModel, Spaim"
            )

        if self.reference_adata is None:
            raise RuntimeError("SpaIMAdapter requires reference_adata=")

        model = model_class(device=self.device)
        model.fit(reference=self.reference_adata, spatial=masked)
        predicted = model.predict(masked)

        # Normalize SpaIM's output (which may be an AnnData or an ndarray) to
        # an AnnData with .layers['imputed'].
        if isinstance(predicted, ad.AnnData):
            if "imputed" not in predicted.layers:
                predicted.layers["imputed"] = np.asarray(predicted.X, dtype=np.float32)
            return predicted

        # Otherwise predicted is array-like (n_cells, n_genes)
        pred_arr = np.asarray(predicted, dtype=np.float32)
        if pred_arr.shape != masked.X.shape:
            raise RuntimeError(
                f"SpaIM produced shape {pred_arr.shape}; expected {masked.X.shape}"
            )
        out = masked.copy()
        out.layers["imputed"] = pred_arr
        return out
