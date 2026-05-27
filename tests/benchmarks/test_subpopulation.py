"""Tests for lumina_st.benchmarks.subpopulation (Round 12 W002)."""

from __future__ import annotations

import numpy as np
import pytest

from lumina_st.benchmarks.subpopulation import (
    per_subtype_degs,
    subcluster_within_label,
    validate_subtype_markers,
)


class TestSubclusterWithinLabel:
    def test_target_cells_get_subcluster_ids(self) -> None:
        rng = np.random.default_rng(0)
        # 60 T cells forming 3 well-separated sub-clusters + 40 non-T cells.
        t_a = rng.normal(loc=[0, 0], scale=0.2, size=(20, 2))
        t_b = rng.normal(loc=[5, 0], scale=0.2, size=(20, 2))
        t_c = rng.normal(loc=[0, 5], scale=0.2, size=(20, 2))
        non_t = rng.normal(loc=[10, 10], scale=0.2, size=(40, 2))
        embedding = np.vstack([t_a, t_b, t_c, non_t])
        labels = ["T"] * 60 + ["Other"] * 40
        sub = subcluster_within_label(embedding, labels, "T", n_subclusters=3,
                                      random_state=0)
        assert (sub[60:] == -1).all()  # non-T cells get -1
        assert set(sub[:60].tolist()) == {0, 1, 2}
        # Each of the 3 ground-truth groups should map to a single sub-cluster id.
        for start in (0, 20, 40):
            ids_in_group = set(sub[start:start + 20].tolist())
            assert len(ids_in_group) == 1

    def test_too_few_cells_defaults_to_single_cluster(self) -> None:
        embedding = np.array([[0.0, 0.0], [1.0, 1.0]])
        labels = ["X", "X"]
        sub = subcluster_within_label(embedding, labels, "X", n_subclusters=5)
        assert (sub == 0).all()

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError):
            subcluster_within_label(np.zeros((10, 2)), ["A"] * 10, "A", method="zzz")


class TestPerSubtypeDegs:
    def test_distinct_markers_surface_in_top_k(self) -> None:
        rng = np.random.default_rng(7)
        # 30 cells, 8 genes; first 10 cells over-express gene 0,
        # next 10 over-express gene 4, last 10 over-express gene 7.
        X = rng.normal(size=(30, 8))
        X[:10, 0] += 6.0
        X[10:20, 4] += 6.0
        X[20:, 7] += 6.0
        labels = [0] * 10 + [1] * 10 + [2] * 10
        var_names = [f"G{i}" for i in range(8)]
        deg = per_subtype_degs(X, var_names, labels, n_top=3)
        # Each sub-label's top DEG should be its over-expressed gene.
        assert deg[0][0]["gene"] == "G0"
        assert deg[1][0]["gene"] == "G4"
        assert deg[2][0]["gene"] == "G7"

    def test_unassigned_cells_excluded(self) -> None:
        X = np.random.default_rng(0).normal(size=(20, 5))
        labels = [-1] * 10 + [0] * 10
        deg = per_subtype_degs(X, [f"G{i}" for i in range(5)], labels, n_top=2)
        assert 0 in deg
        assert -1 not in deg

    def test_empty_when_too_few_cells_per_group(self) -> None:
        X = np.random.default_rng(0).normal(size=(4, 3))
        labels = [0, 0, 1, 1]   # only 2 in each — below the threshold of 3
        deg = per_subtype_degs(X, ["A", "B", "C"], labels, n_top=2)
        assert deg[0] == []
        assert deg[1] == []


class TestValidateSubtypeMarkers:
    def test_hit_rate_matches_top_k(self) -> None:
        deg = {
            "Treg": [{"gene": "FOXP3", "auc": 0.95}, {"gene": "IL2RA", "auc": 0.9}],
            "CD4Tn": [{"gene": "CCR7", "auc": 0.93}, {"gene": "SELL", "auc": 0.91}],
        }
        out = validate_subtype_markers(
            deg,
            expected_markers={"Treg": ["FOXP3"], "CD4Tn": ["CCR7", "MISSING"]},
            top_k=3,
        )
        assert out["Treg"]["hit_rate"] == pytest.approx(1.0)
        assert out["Treg"]["found_in_top_k"] == ["FOXP3"]
        assert out["CD4Tn"]["hit_rate"] == pytest.approx(0.5)
        assert "MISSING" in out["CD4Tn"]["missing"]
        assert out["CD4Tn"]["top_rank"]["CCR7"] == 1
        assert out["CD4Tn"]["top_rank"]["MISSING"] is None
