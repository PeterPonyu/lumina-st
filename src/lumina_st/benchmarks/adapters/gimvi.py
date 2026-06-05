"""gimVI head-to-head adapter.

Wraps the external scvi-tools gimVI model (Lopez et al., NeurIPS 2019) into
the BaseAdapter contract. When scvi-tools is not installed on this machine,
the adapter records `status='unavailable:gimvi-not-installed'` with an install
hint, so the runner can include the row in the comparison table.

gimVI (Generalised Imputation Model for Variational Inference) jointly models
scRNA-seq and ST data in a shared latent space and imputes unmeasured spatial
genes.

Reference:
    Lopez, R. et al. "A joint model of unpaired data from scRNA-seq and
    spatial transcriptomics for imputing missing gene expression
    measurements." NeurIPS 2019 Workshop on Learning Meaningful
    Representations of Life.
    https://doi.org/10.48550/arXiv.1905.02269
    https://github.com/scverse/scvi-tools  (GIMVI class)
"""

from __future__ import annotations

from typing import Any, Optional

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter

_INSTALL_HINT = (
    "gimvi-not-installed: pip install scvi-tools "
    "(adapter expects scvi.model.GIMVI; see https://docs.scvi-tools.org)"
)


class GimVIAdapter(BaseAdapter):
    """Adapter for the gimVI joint scRNA-seq + ST imputer.

    Required adapter option:
        reference_adata : ad.AnnData
            scRNA-seq reference covering the gene panel; gimVI jointly models
            reference and spatial data.

    Optional adapter options:
        n_latent : int
            Dimensionality of the shared latent space (default 10).
        n_epochs : int
            Number of training epochs (default 200).
        use_gpu : bool
            Whether to request GPU training via scvi-tools (default False).
    """

    name = "gimvi"

    def __init__(
        self,
        reference_adata: Optional[ad.AnnData] = None,
        n_latent: int = 10,
        n_epochs: int = 200,
        use_gpu: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.reference_adata = reference_adata
        self.n_latent = n_latent
        self.n_epochs = n_epochs
        self.use_gpu = use_gpu

    def _check_available(self) -> tuple[bool, str]:
        try:
            import importlib

            for candidate in ("scvi",):
                try:
                    mod = importlib.import_module(candidate)
                    # Verify GIMVI is accessible within scvi-tools.
                    model_mod = importlib.import_module("scvi.model")
                    if not hasattr(model_mod, "GIMVI"):
                        return False, (
                            "gimvi-not-installed: scvi-tools is installed but "
                            "scvi.model.GIMVI is missing; upgrade to scvi-tools>=0.16"
                        )
                    self._module_name = candidate
                    break
                except ImportError:
                    continue
            else:
                return False, _INSTALL_HINT
        except Exception as exc:
            return False, f"gimvi-check-error: {type(exc).__name__}: {exc}"

        if self.reference_adata is None:
            return False, (
                "gimvi-requires-reference: pass reference_adata= (scRNA-seq "
                "AnnData covering the ST gene panel) to GimVIAdapter"
            )

        return True, ""

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        from scvi.model import GIMVI

        if self.reference_adata is None:
            raise RuntimeError("GimVIAdapter requires reference_adata=")

        ref = self.reference_adata.copy()
        sp = masked.copy()

        # gimVI requires the spatial and reference to share gene names.
        # We use the intersection as the modelled gene set.
        var_names_spatial = list(sp.var_names)
        var_names_ref = list(ref.var_names)
        shared = [g for g in var_names_spatial if g in var_names_ref]
        if not shared:
            raise RuntimeError(
                "GimVIAdapter: no overlapping genes between reference and spatial data"
            )

        # Subset both to shared genes only.
        ref_sub = ref[:, shared].copy()
        sp_sub = sp[:, shared].copy()

        GIMVI.setup_anndata(ref_sub)
        GIMVI.setup_anndata(sp_sub)

        model = GIMVI(ref_sub, sp_sub, n_latent=self.n_latent)
        model.train(n_epochs=self.n_epochs, use_gpu=self.use_gpu)

        # Impute: predict the spatial gene expression from the trained model.
        # GIMVI.get_imputed_values returns (n_cells, n_genes) for the spatial
        # dataset.
        imputed_vals = model.get_imputed_values(normalized=True)
        if hasattr(imputed_vals, "values"):
            imputed_vals = imputed_vals.values
        imputed_vals = np.asarray(imputed_vals, dtype=np.float32)

        # Build the full output matrix (genes not in shared stay as-observed).
        X_full = np.asarray(masked.X, dtype=np.float32).copy()
        if hasattr(masked.X, "toarray"):
            X_full = masked.X.toarray().astype(np.float32)

        for j_local, g in enumerate(shared):
            j_global = var_names_spatial.index(g)
            X_full[:, j_global] = imputed_vals[:, j_local]

        out = masked.copy()
        out.layers["imputed"] = X_full
        return out
