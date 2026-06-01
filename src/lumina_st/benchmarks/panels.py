"""Static marker-gene panels for held-out benchmark evaluation.

These panels are defined here, before any imputation runs, so the runner can
verify the held-out set has not been derived from clustering or DEG analysis
of the imputed matrix. That is the non-circular-validation discipline called
out in `docs/LUMINAST_PRIORITY_ENHANCEMENTS.md`.

The canonical TME panel mirrors the marker set the two reference papers
validate against (T-cell, macrophage, epithelial, endothelial, fibroblast
markers); it is *not* copied from those papers — these are standard
immune-stroma markers used across the field.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarkerPanel:
    name: str
    genes: tuple[str, ...]
    description: str


TME_IMMUNE_STROMAL = MarkerPanel(
    name="tme-immune-stromal",
    genes=("CD4", "CD8A", "CD68", "EPCAM", "ENG", "POSTN"),
    description=(
        "Standard tumor-microenvironment marker set covering T cells (CD4, CD8A), "
        "macrophages (CD68), epithelial cells (EPCAM), endothelial cells (ENG), and "
        "fibroblasts (POSTN). Used for non-circular held-out gene benchmarks."
    ),
)


VASCULAR_3D = MarkerPanel(
    name="vascular-3d",
    # HGNC-approved symbols only (issue #107). Prior values CD20, CK5 and
    # CD31 were aliases/common names, not approved symbols, so they failed to
    # match var_names from HGNC-annotated references:
    #   CD20 -> MS4A1, CK5 -> KRT5, CD31 -> PECAM1. CD3D is already approved.
    genes=("CD3D", "MS4A1", "KRT5", "PECAM1"),
    description=(
        "T/B/epithelial/endothelial markers (CD3D, MS4A1/CD20, KRT5/CK5, "
        "PECAM1/CD31) used for 3D vascular-network continuity validation in "
        "the Aether3D track. All symbols are HGNC-approved."
    ),
)


NEURONAL_BRAIN = MarkerPanel(
    name="neuronal-brain",
    genes=("Slc17a7", "Gad1", "Gad2", "Gfap"),
    description=(
        "Excitatory neurons (Slc17a7), inhibitory neurons (Gad1/Gad2), and "
        "astrocytes (Gfap). Used for brain-atlas held-out validation."
    ),
)


REGISTERED_PANELS: dict[str, MarkerPanel] = {
    p.name: p
    for p in (TME_IMMUNE_STROMAL, VASCULAR_3D, NEURONAL_BRAIN)
}


def get_panel(name: str) -> MarkerPanel:
    if name not in REGISTERED_PANELS:
        raise KeyError(
            f"Unknown marker panel {name!r}. Available: {sorted(REGISTERED_PANELS)}"
        )
    return REGISTERED_PANELS[name]
