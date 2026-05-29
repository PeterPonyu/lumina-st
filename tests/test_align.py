"""Regression test for lumina-st #104 + #111.

``AnnDataSchemaValidator.align_genes`` used to return
``list(set(target_genes).intersection(ref_genes))``. Python ``set``
iteration order over strings depends on ``PYTHONHASHSEED``, so the
returned overlap list (and any AnnData subset / column reorder built
from it) varied run-to-run for identical inputs.

This pinned a reproducibility hazard: a pipeline whose model input is
an ordered gene vector would silently see a different gene ordering
across processes.

The fix sorts the intersection alphabetically (``sorted(set_a & set_b)``)
so the order is deterministic and independent of either input's
``var_names`` ordering.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from lumina_st.data.validation import AnnDataSchemaValidator


def _make_adata(gene_names: list[str]) -> ad.AnnData:
    n_obs = 2
    X = np.ones((n_obs, len(gene_names)), dtype=np.float32)
    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=X, var=var)
    adata.obsm["spatial"] = np.zeros((n_obs, 2), dtype=np.float32)
    return adata


def test_align_genes_deterministic() -> None:
    """Two calls with the same input must produce the identical ordered list.

    Before the fix the order came from ``set`` iteration and could change
    across processes. After the fix the order is the sorted intersection,
    so repeated calls — and calls from differently-ordered inputs — give
    the same list.
    """

    target = _make_adata(["GENE_C", "GENE_A", "GENE_D", "GENE_B"])
    ref = _make_adata(["GENE_B", "GENE_D", "GENE_A", "GENE_C", "GENE_E"])

    first = AnnDataSchemaValidator.align_genes(target, ref, min_overlap_ratio=0.5)
    second = AnnDataSchemaValidator.align_genes(target, ref, min_overlap_ratio=0.5)

    assert first == second, (first, second)

    # Order must be deterministic AND independent of either input's
    # var_names ordering. Sorted intersection is the documented contract.
    assert first == sorted(first), first
    assert first == ["GENE_A", "GENE_B", "GENE_C", "GENE_D"], first


def test_align_genes_order_independent_of_input_order() -> None:
    """Swapping target var_names ordering must NOT change the output order.

    This catches an implementation that preserves one input's ordering by
    accident, which would still be deterministic per-process but would
    flip the ordering whenever a caller re-shuffled their AnnData.
    """

    target_a = _make_adata(["GENE_C", "GENE_A", "GENE_B"])
    target_b = _make_adata(["GENE_B", "GENE_C", "GENE_A"])
    ref = _make_adata(["GENE_A", "GENE_B", "GENE_C"])

    out_a = AnnDataSchemaValidator.align_genes(target_a, ref)
    out_b = AnnDataSchemaValidator.align_genes(target_b, ref)

    assert out_a == out_b == ["GENE_A", "GENE_B", "GENE_C"], (out_a, out_b)
