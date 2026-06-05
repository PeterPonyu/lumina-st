"""Tests for foundation-model provenance guard (issue #310).

Covers:
- Foundation backend marked reference_reported serializes with flag + source.
- Guard raises when a foundation backend is recorded as a claimed re-run.
- Non-foundation methods (mean/knn/lumina) are unaffected (flag stays False).
- Round-trip / dataclass serialization still works after field additions.
"""

from __future__ import annotations

import pytest

from lumina_st.benchmarks.contract import AdapterResult, Provenance
from lumina_st.benchmarks.foundation_provenance import (
    FOUNDATION_BACKENDS,
    FoundationRerunError,
    assert_not_claimed_rerun,
    is_reference_reported,
    mark_reference_reported,
    require_reference_reported,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provenance(method: str = "mean") -> Provenance:
    return Provenance(method=method, seed=0)


def _make_result(method: str = "mean") -> AdapterResult:
    return AdapterResult(
        method=method,
        imputed_h5ad=None,
        metrics_json={},
        provenance=_make_provenance(method),
        status="ok",
    )


# ---------------------------------------------------------------------------
# FOUNDATION_BACKENDS canonical set
# ---------------------------------------------------------------------------


def test_foundation_backends_contains_expected():
    assert "scgpt" in FOUNDATION_BACKENDS
    assert "nicheformer" in FOUNDATION_BACKENDS
    assert "geneformer" in FOUNDATION_BACKENDS
    assert "uce" in FOUNDATION_BACKENDS


def test_foundation_backends_does_not_contain_local_methods():
    for m in ("mean", "knn", "lumina", "tissue", "novosparc"):
        assert m not in FOUNDATION_BACKENDS


def test_foundation_backends_is_frozenset():
    assert isinstance(FOUNDATION_BACKENDS, frozenset)


# ---------------------------------------------------------------------------
# mark_reference_reported + is_reference_reported on AdapterResult
# ---------------------------------------------------------------------------


def test_mark_reference_reported_sets_flag_and_source_on_result():
    result = _make_result("scgpt")
    returned = mark_reference_reported(result, source="scGPT paper Table 2")

    assert returned is result, "mark_reference_reported must return the same object"
    assert result.reference_reported is True
    assert result.reference_source == "scGPT paper Table 2"


def test_is_reference_reported_true_after_mark():
    result = _make_result("nicheformer")
    assert not is_reference_reported(result)
    mark_reference_reported(result, source="Nicheformer supplementary S3")
    assert is_reference_reported(result)


def test_marked_result_serializes_with_flag_and_source():
    """to_dict() round-trip must include reference_reported and reference_source."""
    result = _make_result("geneformer")
    mark_reference_reported(result, source="Geneformer Nature 2023 Table 1")

    d = result.to_dict()

    assert d["reference_reported"] is True
    assert d["reference_source"] == "Geneformer Nature 2023 Table 1"


# ---------------------------------------------------------------------------
# mark_reference_reported + is_reference_reported on Provenance
# ---------------------------------------------------------------------------


def test_mark_reference_reported_on_provenance():
    prov = _make_provenance("uce")
    mark_reference_reported(prov, source="UCE preprint Table S2")

    assert prov.reference_reported is True
    assert prov.reference_source == "UCE preprint Table S2"
    assert is_reference_reported(prov)


# ---------------------------------------------------------------------------
# require_reference_reported / assert_not_claimed_rerun guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(FOUNDATION_BACKENDS))
def test_require_reference_reported_raises_for_each_foundation_backend(backend):
    with pytest.raises(FoundationRerunError, match=backend):
        require_reference_reported(backend)


@pytest.mark.parametrize("backend", sorted(FOUNDATION_BACKENDS))
def test_assert_not_claimed_rerun_raises_for_each_foundation_backend(backend):
    with pytest.raises(FoundationRerunError):
        assert_not_claimed_rerun(backend)


def test_require_reference_reported_case_insensitive():
    """The guard is case-insensitive so 'scGPT' and 'SCGPT' are caught."""
    with pytest.raises(FoundationRerunError):
        require_reference_reported("scGPT")
    with pytest.raises(FoundationRerunError):
        require_reference_reported("GENEFORMER")


@pytest.mark.parametrize("method", ["mean", "knn", "lumina", "tissue", "novosparc"])
def test_require_reference_reported_does_not_raise_for_non_foundation(method):
    """Non-foundation methods pass through the guard silently."""
    require_reference_reported(method)  # must not raise


@pytest.mark.parametrize("method", ["mean", "knn", "lumina"])
def test_assert_not_claimed_rerun_does_not_raise_for_non_foundation(method):
    assert_not_claimed_rerun(method)  # must not raise


# ---------------------------------------------------------------------------
# Backward compatibility: default fields are False / None
# ---------------------------------------------------------------------------


def test_new_result_has_reference_reported_false_by_default():
    result = _make_result("mean")
    assert result.reference_reported is False
    assert result.reference_source is None


def test_new_provenance_has_reference_reported_false_by_default():
    prov = _make_provenance("knn")
    assert prov.reference_reported is False
    assert prov.reference_source is None


def test_to_dict_includes_reference_fields_with_defaults():
    result = _make_result("lumina")
    d = result.to_dict()

    assert "reference_reported" in d
    assert "reference_source" in d
    assert d["reference_reported"] is False
    assert d["reference_source"] is None


# ---------------------------------------------------------------------------
# is_reference_reported is safe on objects without the attribute
# ---------------------------------------------------------------------------


def test_is_reference_reported_safe_on_plain_object():
    class Bare:
        pass

    assert is_reference_reported(Bare()) is False


# ---------------------------------------------------------------------------
# Full round-trip: foundation result stamped and serialized
# ---------------------------------------------------------------------------


def test_full_round_trip_foundation_result():
    result = _make_result("scgpt")

    # Before stamping: guard raises
    with pytest.raises(FoundationRerunError):
        require_reference_reported(result.method)

    # Stamp it
    mark_reference_reported(result, source="scGPT paper Table 2")

    # is_reference_reported now true
    assert is_reference_reported(result)

    # Serializes correctly
    d = result.to_dict()
    assert d["reference_reported"] is True
    assert d["reference_source"] == "scGPT paper Table 2"
    assert d["method"] == "scgpt"
    assert d["status"] == "ok"
