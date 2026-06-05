"""stDiff head-to-head adapter.

Wraps the external stDiff package (Xu et al., Briefings in Bioinformatics 2024)
into the BaseAdapter contract. When stDiff is not installed on this machine,
the adapter records `status='unavailable:stdiff-not-installed'` with an install
hint, so the runner can include the row in the comparison table.

stDiff uses a diffusion model for imputation of spatial transcriptomics data,
conditioning the reverse diffusion process on observed genes.

Reference:
    Xu, Z. et al. "stDiff: a diffusion model for imputing spatial
    transcriptomics through single-cell transcriptomics."
    Briefings in Bioinformatics 25(3), bbae171 (2024).
    https://doi.org/10.1093/bib/bbae171
    https://github.com/hannshu/stDiff
"""

from __future__ import annotations

from typing import Any, Optional

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter

_INSTALL_HINT = (
    "stdiff-not-installed: clone https://github.com/hannshu/stDiff and "
    "install requirements; adapter expects an `stDiff` or `stdiff` importable module"
)


class StDiffAdapter(BaseAdapter):
    """Adapter for the stDiff diffusion-based ST imputer.

    Optional adapter options:
        reference_adata : ad.AnnData
            scRNA-seq reference covering the gene panel (required by stDiff for
            conditioning the diffusion process).
        n_steps : int
            Number of diffusion steps (default 1000).
        device : str
            'cpu' or 'cuda' (default 'cpu').
    """

    name = "stdiff"

    def __init__(
        self,
        reference_adata: Optional[ad.AnnData] = None,
        n_steps: int = 1000,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.reference_adata = reference_adata
        self.n_steps = n_steps
        self.device = device

    def _check_available(self) -> tuple[bool, str]:
        try:
            import importlib

            for candidate in ("stDiff", "stdiff", "st_diff"):
                try:
                    importlib.import_module(candidate)
                    self._module_name = candidate
                    return True, ""
                except ImportError:
                    continue
            return False, _INSTALL_HINT
        except Exception as exc:
            return False, f"stdiff-check-error: {type(exc).__name__}: {exc}"

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        import importlib

        stDiff = importlib.import_module(getattr(self, "_module_name", "stDiff"))

        # stDiff upstream exposes a model class with fit()/impute() methods
        # that accept AnnData objects. Try canonical class names in order.
        model_cls = None
        for cls_name in ("stDiff", "StDiff", "Model"):
            cls = getattr(stDiff, cls_name, None)
            if cls is not None:
                model_cls = cls
                break
        if model_cls is None:
            raise RuntimeError(
                f"Module {getattr(self, '_module_name', 'stDiff')!r} does not expose "
                f"a known stDiff class; checked: stDiff, StDiff, Model"
            )

        model_kwargs: dict[str, Any] = {"n_steps": self.n_steps, "device": self.device}

        if self.reference_adata is not None:
            model = model_cls(reference=self.reference_adata, **model_kwargs)
        else:
            model = model_cls(**model_kwargs)

        model.fit(masked)
        predicted = model.impute(masked)

        if isinstance(predicted, ad.AnnData):
            if "imputed" not in predicted.layers:
                predicted.layers["imputed"] = np.asarray(predicted.X, dtype=np.float32)
            return predicted

        pred_arr = np.asarray(predicted, dtype=np.float32)
        if hasattr(masked.X, "shape"):
            expected_shape = masked.X.shape
        else:
            expected_shape = np.asarray(masked.X).shape
        if pred_arr.shape != expected_shape:
            raise RuntimeError(
                f"stDiff produced shape {pred_arr.shape}; expected {expected_shape}"
            )
        out = masked.copy()
        out.layers["imputed"] = pred_arr
        return out
