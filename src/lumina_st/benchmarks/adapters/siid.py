"""SIID head-to-head adapter.

Wraps the external SIID package (Zheng, Sarkar & Raphael, Genome Research
2025) into the BaseAdapter contract. SIID is a joint-NMF method designed to
map between TWO spatial modalities (their own naming: `vi` = a full-gene-panel
platform, `xe` = a targeted/limited-panel platform) via a spatial registration
transform `R` and a neighbour-correspondence matrix `gamma`, then reconstructs
the `vi`-side full gene panel onto `xe`-side locations.

lumina's benchmark is single-platform (one dataset, some genes masked), not a
cross-platform pair. This adapter repurposes SIID for that setting the same
way a same-platform ablation of a cross-platform method is normally done:
  - `vi` (the "has everything" view) = `inp.input_h5ad`, the unmasked reference
    AnnData already used as ground truth elsewhere in this repo's own
    adapters (see knn.py's `inp.input_h5ad.X` read) -- audit-safe, not the
    thing being scored.
  - `xe` (the "only sees observed genes" view) = `masked`, the current
    held-out-masked AnnData. SIID's own `prepare_data` already restricts the
    `xe` view to the train-gene columns, so held-out genes are structurally
    absent from what SIID's encoder conditions on, matching the contract's
    audit-safety requirement independent of the runner's own masking.
  - `R = identity` (3x3): no cross-platform registration is needed since both
    views share the same coordinate system by construction.

Reference:
    Zheng, Y., Sarkar, A. & Raphael, B.J. "SIID: joint imputation and
    deconvolution for spatial transcriptomics." Genome Research (2025).
    https://github.com/raphael-group/SIID
"""

from __future__ import annotations

from typing import Any, Optional

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter

_INSTALL_HINT = (
    "siid-not-installed: clone https://github.com/raphael-group/SIID and add "
    "its `src/` dir to sys.path (adapter expects lib_helper.impute_genes)"
)


class SiidAdapter(BaseAdapter):
    """Adapter for the SIID joint-NMF imputation method.

    Optional adapter options:
        repo_path : str
            Path to a cloned SIID checkout (its `src/` dir is added to
            sys.path so `import lib_helper` resolves).
        hdim : int
            Number of shared low-dimensional NMF factors (default 20).
        epochs : int
            Training epochs (SIID's own default is 5000; pass a small value
            for a smoke-scale run -- this is a correctness check, not a
            production accuracy run).
        device : str ('cpu' or 'cuda:0', default 'cpu')
    """

    name = "siid"

    def __init__(
        self,
        repo_path: Optional[str] = None,
        hdim: int = 20,
        epochs: int = 5000,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.repo_path = repo_path
        self.hdim = hdim
        self.epochs = epochs
        self.device = device

    def _check_available(self) -> tuple[bool, str]:
        import sys

        if self.repo_path and self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        try:
            import importlib

            importlib.import_module("lib_helper")
        except ImportError:
            return False, _INSTALL_HINT
        except Exception as exc:
            return False, f"siid-check-error: {type(exc).__name__}: {exc}"
        return True, ""

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        import sys

        if self.repo_path and self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        import lib_helper

        if not inp.held_out_genes:
            out = masked.copy()
            X = out.X.toarray() if hasattr(out.X, "toarray") else np.asarray(out.X).copy()
            out.layers["imputed"] = X
            return out

        vi = inp.input_h5ad.copy()   # full-panel reference (truth for train+test genes)
        xe = masked.copy()           # masked view (only observed/train genes usable)
        R = np.eye(3)

        full_pred = lib_helper.impute_genes(
            vi, xe, list(inp.held_out_genes), self.hdim,
            R=R, seed=int(inp.seed), epochs=int(self.epochs), device=self.device,
        )

        var_names = list(masked.var_names)
        X = masked.X.toarray() if hasattr(masked.X, "toarray") else np.asarray(masked.X).copy()
        pred_var_names = list(full_pred.var_names)
        pred_X = full_pred.X.toarray() if hasattr(full_pred.X, "toarray") else np.asarray(full_pred.X)
        for g in inp.held_out_genes:
            if g in var_names and g in pred_var_names:
                X[:, var_names.index(g)] = pred_X[:, pred_var_names.index(g)]

        out = masked.copy()
        out.layers["imputed"] = X.astype(np.float32)
        return out
