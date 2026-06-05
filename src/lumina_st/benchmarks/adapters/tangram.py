"""Tangram head-to-head adapter.

Wraps the external Tangram package (Biancalani et al., Nat Methods 2021) into
the BaseAdapter contract. When Tangram is not installed on this machine, the
adapter records `status='unavailable:tangram-not-installed'` with an install
hint, so the runner can include the row in the comparison table.

Tangram maps single-cell RNA-seq data onto spatial coordinates using optimal
transport, then transfers unmeasured gene expression from the reference.

Reference:
    Biancalani, T. et al. "Deep learning and alignment of spatially resolved
    single-cell transcriptomes with Tangram." Nat Methods 18, 1352-1362 (2021).
    https://doi.org/10.1038/s41592-021-01264-7
    https://github.com/broadinstitute/Tangram
"""

from __future__ import annotations

from typing import Any, Optional

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter

_INSTALL_HINT = (
    "tangram-not-installed: pip install tangram-sc "
    "(adapter expects the public Tangram API; see https://github.com/broadinstitute/Tangram)"
)


class TangramAdapter(BaseAdapter):
    """Adapter for the Tangram spatial transcriptomics alignment method.

    Required adapter option:
        reference_adata : ad.AnnData
            scRNA-seq reference covering the gene panel; Tangram maps reference
            cells onto spatial locations, then transfers unmeasured genes.

    Optional adapter options:
        num_epochs : int
            Number of optimisation epochs (default 1000).
        device : str
            'cpu' or 'cuda' (default 'cpu').
        mode : str
            Tangram mapping mode: 'cells', 'clusters', or 'constrained'
            (default 'cells').
    """

    name = "tangram"

    def __init__(
        self,
        reference_adata: Optional[ad.AnnData] = None,
        num_epochs: int = 1000,
        device: str = "cpu",
        mode: str = "cells",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.reference_adata = reference_adata
        self.num_epochs = num_epochs
        self.device = device
        self.mode = mode

    def _check_available(self) -> tuple[bool, str]:
        try:
            import importlib

            for candidate in ("tangram", "tangram_sc"):
                try:
                    importlib.import_module(candidate)
                    self._module_name = candidate
                    break
                except ImportError:
                    continue
            else:
                return False, _INSTALL_HINT
        except Exception as exc:
            return False, f"tangram-check-error: {type(exc).__name__}: {exc}"

        if self.reference_adata is None:
            return False, (
                "tangram-requires-reference: pass reference_adata= (scRNA-seq "
                "AnnData covering the ST gene panel) to TangramAdapter"
            )

        return True, ""

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        import importlib

        tg = importlib.import_module(getattr(self, "_module_name", "tangram"))

        if self.reference_adata is None:
            raise RuntimeError("TangramAdapter requires reference_adata=")

        # Tangram public API: pp_adatas → map_cells_to_space → project_genes
        # Requires shared marker genes between reference and target.
        var_names_spatial = list(masked.var_names)
        var_names_ref = list(self.reference_adata.var_names)
        markers = [g for g in var_names_spatial if g in var_names_ref]
        if not markers:
            raise RuntimeError(
                "TangramAdapter: no overlapping genes between reference and spatial data"
            )

        ref = self.reference_adata.copy()
        sp = masked.copy()

        tg.pp_adatas(ref, sp, genes=markers)

        ad_map = tg.map_cells_to_space(
            ref,
            sp,
            mode=self.mode,
            num_epochs=self.num_epochs,
            device=self.device,
            verbose=False,
        )

        # Project all genes from the reference onto spatial locations.
        ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=ref)

        # Align result back to original spatial var order.
        X_hat = np.zeros(masked.X.shape, dtype=np.float32)
        for i, g in enumerate(var_names_spatial):
            if g in ad_ge.var_names:
                j = list(ad_ge.var_names).index(g)
                X_hat[:, i] = np.asarray(ad_ge.X[:, j], dtype=np.float32).ravel()

        out = masked.copy()
        out.layers["imputed"] = X_hat
        return out
