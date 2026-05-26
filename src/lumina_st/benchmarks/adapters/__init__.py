"""Concrete adapters for ST imputation benchmarks."""

from .cellt import CellTAdapter
from .knn import KNNAdapter
from .lumina import LuminaAdapter
from .mean import MeanAdapter
from .spaim import SpaIMAdapter
from .stmcdi import STMCDIAdapter
from .tissue import TISSUEAdapter

__all__ = [
    "MeanAdapter",
    "KNNAdapter",
    "LuminaAdapter",
    "SpaIMAdapter",
    "TISSUEAdapter",
    "STMCDIAdapter",
    "CellTAdapter",
]
