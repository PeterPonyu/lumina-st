"""Concrete adapters for ST imputation benchmarks."""

from .cellt import CellTAdapter
from .gimvi import GimVIAdapter
from .knn import KNNAdapter
from .lumina import LuminaAdapter
from .mean import MeanAdapter
from .novosparc import NovoSpaRcAdapter
from .reference_regression import ReferenceRegressionAdapter
from .spaim import SpaIMAdapter
from .spatial_neighbor import SpatialNeighborAvgAdapter
from .stdiff import StDiffAdapter
from .stmcdi import STMCDIAdapter
from .tangram import TangramAdapter
from .tissue import TISSUEAdapter

__all__ = [
    "MeanAdapter",
    "KNNAdapter",
    "SpatialNeighborAvgAdapter",
    "ReferenceRegressionAdapter",
    "LuminaAdapter",
    "SpaIMAdapter",
    "TISSUEAdapter",
    "STMCDIAdapter",
    "CellTAdapter",
    "TangramAdapter",
    "GimVIAdapter",
    "StDiffAdapter",
    "NovoSpaRcAdapter",
]
