"""LuminaST self-adapter: wraps LuminaImputer.enhance() in the BaseAdapter contract.

Honors the same input/output schema as every external method, so the runner
can compare them on identical inputs.
"""

from __future__ import annotations

import anndata as ad

from ..contract import AdapterInput, BaseAdapter


class LuminaAdapter(BaseAdapter):
    name = "lumina"

    def __init__(self, imputer=None, config=None, **kwargs):
        super().__init__(**kwargs)
        self.imputer = imputer
        self.config = config

    def _check_available(self) -> tuple[bool, str]:
        if self.imputer is None and self.config is None:
            return False, "lumina-adapter requires either imputer= or config= (none provided)"
        return True, ""

    def _impute(self, masked: ad.AnnData, inp: AdapterInput) -> ad.AnnData:
        imputer = self.imputer
        if imputer is None:
            from ...core.lumina_imputer import LuminaImputer

            imputer = LuminaImputer.from_config(self.config)

        # Held-out genes were already zeroed by AdapterInput.masked_input(); we
        # pass them through to LuminaImputer.enhance() so its internal book-
        # keeping logs the masked set, but the input is already zeroed.
        enhanced = imputer.enhance(
            masked,
            cancer_type=inp.cancer_type,
            held_out_genes=inp.held_out_genes,
        )

        if "imputed" not in enhanced.layers:
            raise RuntimeError(
                "LuminaImputer.enhance produced no 'imputed' layer "
                "(likely latent-only output). Configure VAE decode for benchmark use."
            )
        return enhanced
