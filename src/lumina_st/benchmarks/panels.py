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
    genes=("CD3D", "CD20", "CK5", "CD31"),
    description=(
        "T/B/epithelial/endothelial markers used for 3D vascular-network "
        "continuity validation in the Aether3D track."
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
