"""NovoSpaRc head-to-head adapter.

Wraps the external NovoSpaRc package (Nitzan et al., Nature 2019) into the
BaseAdapter contract. When NovoSpaRc is not installed on this machine, the
adapter records `status='unavailable:novosparc-not-installed'` with an install
hint, so the runner can include the row in the comparison table.

NovoSpaRc reconstructs spatial gene expression from scRNA-seq by optimal
transport de-novo reconstruction, optionally guided by a small set of landmark
genes with known spatial patterns.

Reference:
    Nitzan, M. et al. "Gene expression cartography." Nature 576, 132-137 (2019).
    https://doi.org/10.1038/s41586-019-1773-3
    https://github.com/rajewsky-lab/novosparc
"""

from __future__ import annotations

from typing import Any, Optional

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter

_INSTALL_HINT = (
    "novosparc-not-installed: pip install novosparc "
    "(adapter expects the public NovoSpaRc API; see https://github.com/rajewsky-lab/novosparc)"
)


class NovoSpaRcAdapter(BaseAdapter):
    """Adapter for the NovoSpaRc optimal-transport ST reconstruction method.

    Optional adapter options:
        num_locations : int
            Number of virtual spatial grid locations to reconstruct onto. When
            None (default), the number of spatial spots in the target adata is
            used directly.
        alpha_linear : float
            Trade-off between expression- and geometry-based transport cost
            (default 0.5). 0.0 = expression-only; 1.0 = geometry-only.
        epsilon : float
            Entropic regularisation for the Sinkhorn solver (default 5e-3).
    """

    name = "novosparc"

    def __init__(
        self,
        num_locations: Optional[int] = None,
        alpha_linear: float = 0.5,
        epsilon: float = 5e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_locations = num_locations
        self.alpha_linear = alpha_linear
        self.epsilon = epsilon

    def _check_available(self) -> tuple[bool, str]:
        try:
            import importlib

            for candidate in ("novosparc",):
                try:
                    importlib.import_module(candidate)
                    self._module_name = candidate
                    return True, ""
                except ImportError:
                    continue
            return False, _INSTALL_HINT
        except Exception as exc:
            return False, f"novosparc-check-error: {type(exc).__name__}: {exc}"

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        import importlib

        novosparc = importlib.import_module(getattr(self, "_module_name", "novosparc"))

        X = masked.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        n_cells, n_genes = X.shape
        n_locs = self.num_locations if self.num_locations is not None else n_cells

        # Build a Tissue object from the expression matrix.
        # NovoSpaRc public API: novosparc.cm.Tissue(dataset, locations)
        # `dataset` is (n_cells x n_genes) ndarray; `locations` is (n_locs x 2).
        # When we have spatial coordinates, use them; otherwise build a grid.
        if "spatial" in masked.obsm:
            locations = np.asarray(masked.obsm["spatial"], dtype=np.float64)
            if locations.shape[0] != n_cells:
                locations = locations[:n_cells]
        else:
            grid_side = int(np.ceil(np.sqrt(n_locs)))
            gx, gy = np.meshgrid(np.arange(grid_side), np.arange(grid_side))
            locations = np.column_stack([gx.ravel(), gy.ravel()])[:n_locs].astype(
                np.float64
            )

        # Build the tissue reconstruction object.
        tissue = novosparc.cm.Tissue(dataset=X, locations=locations)

        # Set up the transport plan using expression cost + optional geometry.
        tissue.setup_reconstruction(
            num_neighbors_s=5,
            num_neighbors_t=5,
            alpha_linear=self.alpha_linear,
        )

        # Recover the mapping (Sinkhorn-regularised OT plan).
        tissue.reconstruct(epsilon=self.epsilon)

        # Project gene expression onto spatial locations.
        # NovoSpaRc: sdge = (n_locs, n_genes)
        sdge = tissue.sdge  # np array (n_genes, n_locs) — transposed in some versions
        if sdge.shape == (n_genes, n_locs):
            sdge = sdge.T  # -> (n_locs, n_genes)

        imputed = np.asarray(sdge, dtype=np.float32)
        if imputed.shape[0] != n_cells:
            # Resize to match n_cells if grid was padded.
            imputed = imputed[:n_cells]

        out = masked.copy()
        out.layers["imputed"] = imputed
        return out
