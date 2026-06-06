"""Tests for the reference/target gene-panel loader (lumina-st #314/#296).

Covers:
  * ``align_to_shared_panel`` — robust to either direction of panel containment,
    which is what lets the new whole-transcriptome COAD scRNA reference
    (GSE132465, ~33k genes) pair against a ~5k-gene Xenium target without the
    old "reference not fully covered by target" hard-fail.
  * ``looks_log_normalized`` + the ``ReferenceAtlasDataset`` raw-count guard —
    rejects a log1p-normalized reference (#314) while accepting integer counts
    that merely happen to be stored as ``float``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.data.datasets import (
    ReferenceAtlasDataset,
    align_to_shared_panel,
    looks_log_normalized,
)


def _counts_adata(n_obs: int, genes: list[str], seed: int = 0) -> AnnData:
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(n_obs, len(genes))).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = genes
    adata.obs["cancer_type"] = ["COAD"] * n_obs
    return adata


# --------------------------------------------------------------------------- #
# align_to_shared_panel
# --------------------------------------------------------------------------- #
def test_target_subset_of_reference_subsets_the_reference() -> None:
    """New case: 33k-style reference superset vs a small Xenium-style panel.

    The old loader hard-raised here ("reference not fully covered by target").
    The robust loader subsets the reference DOWN to the shared panel.
    """
    ref = _counts_adata(5, [f"G{i}" for i in range(8)])  # G0..G7
    target = _counts_adata(4, ["G2", "G4", "G6"])  # subset of ref

    ref_a, tgt_a, stats = align_to_shared_panel(ref, target)

    assert list(ref_a.var_names) == ["G2", "G4", "G6"]  # reference order preserved
    assert list(tgt_a.var_names) == ["G2", "G4", "G6"]
    assert ref_a.n_vars == tgt_a.n_vars == 3
    assert stats == {"n_shared": 3, "n_reference_dropped": 5, "n_target_dropped": 0}


def test_reference_subset_of_target_subsets_the_target() -> None:
    """Legacy case: reference ⊆ target. Reference unchanged, target reordered."""
    ref = _counts_adata(5, ["G1", "G3", "G5"])
    target = _counts_adata(4, [f"G{i}" for i in range(8)])  # superset

    ref_a, tgt_a, stats = align_to_shared_panel(ref, target)

    assert list(ref_a.var_names) == ["G1", "G3", "G5"]
    assert list(tgt_a.var_names) == ["G1", "G3", "G5"]  # reordered to ref order
    assert stats == {"n_shared": 3, "n_reference_dropped": 0, "n_target_dropped": 5}


def test_partial_overlap_keeps_only_intersection() -> None:
    ref = _counts_adata(3, ["A", "B", "C", "D"])
    target = _counts_adata(3, ["C", "D", "E", "F"])

    ref_a, tgt_a, stats = align_to_shared_panel(ref, target)

    assert list(ref_a.var_names) == ["C", "D"]
    assert list(tgt_a.var_names) == ["C", "D"]
    assert stats["n_shared"] == 2


def test_no_overlap_raises() -> None:
    ref = _counts_adata(3, ["A", "B"])
    target = _counts_adata(3, ["X", "Y"])
    with pytest.raises(ValueError, match="share only 0 gene"):
        align_to_shared_panel(ref, target)


def test_min_shared_genes_floor() -> None:
    ref = _counts_adata(3, ["A", "B", "C"])
    target = _counts_adata(3, ["C", "Z"])
    with pytest.raises(ValueError, match="share only 1 gene"):
        align_to_shared_panel(ref, target, min_shared_genes=2)


def test_alignment_preserves_expression_values() -> None:
    ref = _counts_adata(5, [f"G{i}" for i in range(6)], seed=1)
    target = _counts_adata(4, ["G3", "G1"], seed=2)
    before = {g: target[:, g].X.ravel().copy() for g in ["G1", "G3"]}

    _, tgt_a, _ = align_to_shared_panel(ref, target)

    # Reordered to reference order (G1 before G3), values intact.
    assert list(tgt_a.var_names) == ["G1", "G3"]
    np.testing.assert_array_equal(tgt_a[:, "G1"].X.ravel(), before["G1"])
    np.testing.assert_array_equal(tgt_a[:, "G3"].X.ravel(), before["G3"])


# --------------------------------------------------------------------------- #
# looks_log_normalized + ReferenceAtlasDataset raw-count guard (#314)
# --------------------------------------------------------------------------- #
def test_log_normalized_matrix_is_flagged() -> None:
    rng = np.random.default_rng(0)
    counts = rng.poisson(5.0, size=(10, 6)).astype(np.float32)
    lognorm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    # Scale into the small-but-non-integer regime characteristic of log1p input.
    lognorm = lognorm / lognorm.max() * 3.0
    assert looks_log_normalized(lognorm) is True


def test_integer_counts_stored_as_float_not_flagged() -> None:
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, size=(10, 6)).astype(np.float32)  # max small, integer
    assert looks_log_normalized(counts) is False


def test_large_integer_counts_not_flagged() -> None:
    rng = np.random.default_rng(0)
    counts = rng.poisson(50.0, size=(10, 6)).astype(np.float32)
    assert looks_log_normalized(counts) is False


def test_sparse_counts_not_flagged() -> None:
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(10, 6)).astype(np.float32)
    assert looks_log_normalized(sp.csr_matrix(counts)) is False


def test_reference_dataset_rejects_normalized_reference() -> None:
    rng = np.random.default_rng(0)
    counts = rng.poisson(5.0, size=(10, 6)).astype(np.float32)
    lognorm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    lognorm = lognorm / lognorm.max() * 3.0
    adata = AnnData(X=lognorm)
    adata.var_names = [f"G{i}" for i in range(6)]
    adata.obs["cancer_type"] = ["COAD"] * 10

    cfg = LuminaSTConfig(vae_batch_key="cancer_type")
    with pytest.raises(ValueError, match="#314"):
        ReferenceAtlasDataset(adata, cfg)

    # Opt-out path still constructs.
    ds = ReferenceAtlasDataset(adata, cfg, validate_counts=False)
    assert len(ds) == 10


def test_reference_dataset_accepts_raw_counts() -> None:
    adata = _counts_adata(8, [f"G{i}" for i in range(6)])
    cfg = LuminaSTConfig(vae_batch_key="cancer_type")
    ds = ReferenceAtlasDataset(adata, cfg)
    x, y = ds[0]
    assert x.shape == (6,)
