"""Concrete adapters for ST imputation benchmarks."""

from .cellt import CellTAdapter
from .knn import KNNAdapter
from .lumina import LuminaAdapter
from .mean import MeanAdapter
from .reference_regression import ReferenceRegressionAdapter
from .spaim import SpaIMAdapter
from .spatial_neighbor import SpatialNeighborAvgAdapter
from .stmcdi import STMCDIAdapter
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
]
