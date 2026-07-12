"""PASTA head-to-head adapter.

Wraps the external PASTA package (Li et al., Nat Commun 2025) into the
BaseAdapter contract. PASTA is a Tangram-family method: it maps single-cell
RNA-seq onto spatial data via `pp_adatas` + `mapping`, then the learned
cell-by-spot mapping matrix is used to project the reference's full gene
panel onto spatial locations -- same shape of adaptation as this repo's
existing tangram.py adapter.

Reference:
    Li, R. et al. "PASTA: pathway-oriented spatial transcriptomics analysis."
    Nat Commun (2025). https://github.com/rx-li/PASTA
"""

from __future__ import annotations

from typing import Any, Optional

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter

_INSTALL_HINT = (
    "pasta-not-installed: clone https://github.com/rx-li/PASTA and add its "
    "`pasta/` package dir to sys.path (adapter expects pp_adatas + mapping)"
)


class PastaAdapter(BaseAdapter):
    """Adapter for the PASTA spatial mapping method.

    Required adapter option:
        reference_adata : ad.AnnData
            scRNA-seq reference covering the gene panel; PASTA maps reference
            cells onto spatial locations, then the mapping matrix transfers
            unmeasured genes.

    Optional adapter options:
        repo_path : str
            Path to a cloned PASTA checkout (its `pasta/` dir is added to
            sys.path so `import mapper` resolves).
        num_epochs : int (default 500, PASTA's own default)
        device : str ('cpu' or 'cuda', default 'cpu')
        celltype_key : str | None
            obs column with real cell/domain-type labels for PASTA's
            celltype-coherence term. If the dataset has no such column
            (e.g. lumina's coad card, which only carries array coordinates),
            falls back to a single constant label for every spot -- an
            honest neutral case, not a fabricated fine-grained label.
    """

    name = "pasta"

    def __init__(
        self,
        reference_adata: Optional[ad.AnnData] = None,
        repo_path: Optional[str] = None,
        num_epochs: int = 500,
        device: str = "cpu",
        celltype_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.reference_adata = reference_adata
        self.repo_path = repo_path
        self.num_epochs = num_epochs
        self.device = device
        self.celltype_key = celltype_key

    def _check_available(self) -> tuple[bool, str]:
        import sys

        if self.repo_path:
            sys.path.insert(0, self.repo_path)
        try:
            import importlib

            importlib.import_module("mapper")
        except ImportError:
            return False, _INSTALL_HINT
        except Exception as exc:
            return False, f"pasta-check-error: {type(exc).__name__}: {exc}"

        if self.reference_adata is None:
            return False, (
                "pasta-requires-reference: pass reference_adata= (scRNA-seq "
                "AnnData covering the ST gene panel) to PastaAdapter"
            )
        return True, ""

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        import sys

        if self.repo_path and self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        import mapper as pasta_mapper

        if self.reference_adata is None:
            raise RuntimeError("PastaAdapter requires reference_adata=")

        var_names_spatial = list(masked.var_names)
        var_names_ref = list(self.reference_adata.var_names)
        markers = [g for g in var_names_spatial if g in var_names_ref]
        if not markers:
            raise RuntimeError(
                "PastaAdapter: no overlapping genes between reference and spatial data"
            )

        ref = self.reference_adata.copy()
        sp = masked.copy()
        # gene_to_lowercase=False: pp_adatas defaults to lowercasing every var_name
        # in-place on both ref and sp. Left at its default, ref.var_names silently
        # stop matching the original-case gene symbols we look up below (X_hat came
        # back all-zero the first time -- not a crash, just a case-fold mismatch).
        pasta_mapper.pp_adatas(ref, sp, genes=markers, gene_to_lowercase=False)

        # PASTA's own mapping() does `sp_celltypes == i` then `.values` on the
        # resulting mask -- it assumes a pandas Series (ndarray has no .values),
        # so this must stay a Series, not be converted to a bare array.
        if self.celltype_key and self.celltype_key in sp.obs.columns:
            sp_celltypes = sp.obs[self.celltype_key].astype(str)
        else:
            # No real cell/domain-type column on this card -- a single constant
            # label keeps PASTA's celltype-coherence term well-defined without
            # inventing fine-grained categories that aren't actually there.
            import pandas as pd
            sp_celltypes = pd.Series(["all"] * sp.n_obs, index=sp.obs.index)

        # Likewise sp_coords: mapping() does `sp_coords[i.values]` then `.iloc[j]`
        # on the result and `.sum(axis=1)` -- it needs a DataFrame, not an ndarray.
        import pandas as pd
        sp_coords = pd.DataFrame(np.asarray(sp.obsm["spatial"], dtype=np.float64),
                                  index=sp.obs.index, columns=["x", "y"])

        # PASTA's pathway_genes arg is NOT optional in its own mapping() despite the
        # signature default suggesting otherwise (it unconditionally iterates it).
        # We have no real pathway/gene-set annotation plumbed into this benchmark,
        # so pass the full training-gene overlap as one undifferentiated "pathway"
        # -- an honest neutral default, not a fabricated pathway boundary.
        pathway_genes = list(ref.uns.get("training_genes", markers))
        ad_map = pasta_mapper.mapping(
            ref,
            sp,
            pathway_genes=pathway_genes,
            sp_coords=sp_coords,
            sp_celltypes=sp_celltypes,
            num_epochs=self.num_epochs,
            device=self.device,
            random_state=int(inp.seed),
            verbose=False,
        )

        # PASTA returns a cell-by-spot probability AnnData (mirrors Tangram's
        # ad_map); project the reference's full gene panel onto spatial
        # locations the same way Tangram's project_genes does internally:
        # spot_expr[spot, gene] = sum_cell mapping[cell, spot] * ref_expr[cell, gene].
        M = np.asarray(ad_map.X, dtype=np.float64)          # [n_cells x n_spots]
        ref_X = ref.X.toarray() if hasattr(ref.X, "toarray") else np.asarray(ref.X)
        col_sum = M.sum(axis=0, keepdims=True)
        col_sum = np.where(col_sum < 1e-12, 1.0, col_sum)
        M_norm = M / col_sum                                  # normalize per spot
        projected = (M_norm.T @ ref_X).astype(np.float32)      # [n_spots x n_ref_genes]
        ref_var_names = list(ref.var_names)

        X_hat = np.zeros(masked.X.shape, dtype=np.float32)
        for i, g in enumerate(var_names_spatial):
            if g in ref_var_names:
                X_hat[:, i] = projected[:, ref_var_names.index(g)]

        out = masked.copy()
        out.layers["imputed"] = X_hat
        return out
