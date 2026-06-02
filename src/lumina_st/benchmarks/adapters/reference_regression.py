"""Reference-regression baseline: a ridge linear map from the shared (observed)
gene panel to the held-out genes, fit on the reference atlas and applied to the
spatial target.

Let ``S`` be the shared-gene expression matrix (cells x |shared|) and ``H`` the
held-out matrix (cells x |held|). We fit ridge weights

    W = (Sc^T Sc + alpha * I)^-1 Sc^T Hc

on the *reference* atlas (``inp.input_h5ad``), where ``Sc``/``Hc`` are
column-centred, then predict the target's held-out genes from its observed
panel:

    H_hat = (S_target - mean(S_ref)) @ W + mean(H_ref).

This is the "learn a linear map from the shared genes" lower bound in the style
of reference methods such as SpaGE / Tangram, implemented clean-room in pure
numpy — no external package is imported. The closed-form ridge solution is
deterministic; the runner's seed only affects its global RNG, not this fit.

Predictions are distinct per held-out gene by construction (each held-out gene
owns a column of ``W``), satisfying the per-gene contract (issue #137). The
target's observed panel is the only signal used, so the audit boundary
(held-out columns zeroed before the adapter runs) is respected.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np

from ..contract import AdapterInput, BaseAdapter


class ReferenceRegressionAdapter(BaseAdapter):
    name = "reference_regression"

    def __init__(self, alpha: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        X = masked.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32).copy()

        out = masked.copy()
        if not inp.held_out_genes:
            out.layers["imputed"] = X
            return out

        var_names = list(masked.var_names)
        held_cols = [var_names.index(g) for g in inp.held_out_genes if g in var_names]
        if not held_cols:
            out.layers["imputed"] = X
            return out

        observed_mask = np.ones(X.shape[1], dtype=bool)
        observed_mask[held_cols] = False

        ref_X = inp.input_h5ad.X
        if hasattr(ref_X, "toarray"):
            ref_X = ref_X.toarray()
        ref_X = np.asarray(ref_X, dtype=np.float64)
        ref_var_names = list(inp.input_h5ad.var_names)

        # Map target-space columns to reference-space columns by gene name.
        target_to_ref = {
            i: ref_var_names.index(var_names[i])
            for i in range(len(var_names))
            if var_names[i] in ref_var_names
        }

        shared_target_cols = [
            i for i in range(X.shape[1]) if observed_mask[i] and i in target_to_ref
        ]
        held_present = [
            (g, c) for g, c in zip(inp.held_out_genes, held_cols) if g in ref_var_names
        ]

        if not shared_target_cols or not held_present:
            # No shared signal or no predictable held-out gene -> reference
            # per-gene mean fallback (still distinct per gene).
            for g, c in held_present:
                X[:, c] = float(ref_X[:, ref_var_names.index(g)].mean())
            out.layers["imputed"] = X
            return out

        ref_shared_idx = [target_to_ref[i] for i in shared_target_cols]
        ref_held_idx = [ref_var_names.index(g) for g, _ in held_present]

        S = ref_X[:, ref_shared_idx]  # (n_ref, n_shared)
        H = ref_X[:, ref_held_idx]  # (n_ref, n_held)
        s_mean = S.mean(axis=0, keepdims=True)
        h_mean = H.mean(axis=0, keepdims=True)
        Sc = S - s_mean
        Hc = H - h_mean

        n_shared = Sc.shape[1]
        A = Sc.T @ Sc + self.alpha * np.eye(n_shared)
        W = np.linalg.solve(A, Sc.T @ Hc)  # (n_shared, n_held)

        # Apply the map to the target's observed panel (audit-safe).
        S_tgt = X[:, shared_target_cols].astype(np.float64) - s_mean
        H_hat = S_tgt @ W + h_mean  # (n_cells, n_held)

        for j, (_g, c) in enumerate(held_present):
            X[:, c] = H_hat[:, j].astype(np.float32)

        out.layers["imputed"] = X
        return out
