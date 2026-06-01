"""Regression tests for the static marker-gene panels.

Issue #107 — the VASCULAR_3D panel used non-HGNC symbols. CD20, CK5 and
CD31 are aliases / common names, not HGNC-approved symbols, so they fail
to match var_names from HGNC-annotated references and silently drop from
the held-out benchmark. Verified against the HGNC REST API:

    CD20 -> MS4A1   (CD20 is an alias)
    CK5  -> KRT5    (CK5 / CK-5 is an alias)
    CD31 -> PECAM1  (CD31 is an alias)
    CD3D -> CD3D    (already an approved symbol)
"""

from __future__ import annotations

from lumina_st.benchmarks.panels import REGISTERED_PANELS, VASCULAR_3D

# Common-name / alias symbols that are NOT HGNC-approved. These must never
# appear in any panel definition.
KNOWN_NON_HGNC_ALIASES = {
    "CD20",  # -> MS4A1
    "CK5",   # -> KRT5
    "CD31",  # -> PECAM1
    "CK-5",  # -> KRT5
}


def test_vascular_3d_all_hgnc() -> None:
    """Every VASCULAR_3D symbol is HGNC-approved (issue #107)."""
    assert VASCULAR_3D.genes == ("CD3D", "MS4A1", "KRT5", "PECAM1")

    offending = set(VASCULAR_3D.genes) & KNOWN_NON_HGNC_ALIASES
    assert not offending, (
        f"VASCULAR_3D contains non-HGNC alias symbols {sorted(offending)}; "
        "use the HGNC-approved symbols instead"
    )


def test_no_panel_uses_known_aliases() -> None:
    """No registered panel may use a known non-HGNC alias symbol."""
    for name, panel in REGISTERED_PANELS.items():
        offending = set(panel.genes) & KNOWN_NON_HGNC_ALIASES
        assert not offending, (
            f"panel {name!r} contains non-HGNC alias symbols {sorted(offending)}"
        )
