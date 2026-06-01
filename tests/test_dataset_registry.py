"""Offline contract tests for the LuminaST dataset registry (no network)."""

import pytest

from lumina_st.data import (
    DATASET_REGISTRY,
    all_ids,
    get,
    issues_covered,
    unverified_specs,
    verified_specs,
)
from lumina_st.data.dataset_registry import LoaderKind, UrlStatus


def test_registry_non_empty_and_ids_unique():
    assert len(DATASET_REGISTRY) >= 16
    ids = all_ids()
    assert len(ids) == len(set(ids))
    assert all(spec.id == key for key, spec in DATASET_REGISTRY.items())


def test_every_open_data_issue_is_covered():
    # The [data] ingestion issues this PR consolidates (incl. 05-29 Prime cohort #184).
    expected = {54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 184}
    assert expected.issubset(set(issues_covered()))


def test_verified_and_unverified_partition():
    assert len(verified_specs()) + len(unverified_specs()) == len(DATASET_REGISTRY)
    # Xenium Prime cohort + the 377-gene skin hotlink are the UNVERIFIED ones.
    unverified_ids = {s.id for s in unverified_specs()}
    assert {"xenium_skin", "xenium_prime_breast", "xenium_prime_skin_melanoma",
            "xenium_prime_ovarian"} == unverified_ids


def test_specs_have_required_provenance_fields():
    for spec in DATASET_REGISTRY.values():
        assert spec.name and spec.platform and spec.tissue
        assert spec.cancer_type and spec.accession and spec.url
        assert spec.citation_key
        assert spec.raw_count_policy
        assert spec.contract_mapping
        assert spec.issues  # at least one tracking issue
        assert isinstance(spec.url_status, UrlStatus)
        assert isinstance(spec.loader, LoaderKind)


def test_reference_atlas_is_flagged_non_spatial():
    # sc_mouse_cortex is the VAE reference atlas (no .obsm['spatial']).
    ref = get("sc_mouse_cortex")
    assert ref.is_reference is True
    assert "ReferenceAtlasDataset" in ref.contract_mapping
    # All spatial targets map onto SpatialTranscriptomicsDataset.
    for spec in DATASET_REGISTRY.values():
        if not spec.is_reference:
            assert "SpatialTranscriptomicsDataset" in spec.contract_mapping


def test_get_unknown_id_raises():
    with pytest.raises(KeyError):
        get("does_not_exist")
