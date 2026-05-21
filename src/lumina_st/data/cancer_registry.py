"""
Configurable cancer / condition registry for LuminaST.

Replaces the hard-coded TUMOR_TO_IDX dictionary that was scattered across
the original stPainter code. Now fully data-driven via YAML/JSON so that
new tumor types can be added without touching source code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class CancerRegistry:
    """Bi-directional mapping between cancer names and integer indices."""

    def __init__(self, name_to_idx: Dict[str, int]):
        self.name_to_idx = {k.upper(): v for k, v in name_to_idx.items()}
        self.idx_to_name = {v: k for k, v in self.name_to_idx.items()}
        self.num_classes = len(self.name_to_idx)

    @classmethod
    def from_file(cls, path: Path) -> "CancerRegistry":
        path = Path(path)
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(path.read_text())
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
        else:
            raise ValueError(f"Unsupported registry format: {path.suffix}")

        if "name_to_idx" in data:
            return cls(data["name_to_idx"])
        # allow flat dict as well
        return cls(data)

    @classmethod
    def default_pan_cancer(cls) -> "CancerRegistry":
        """A sensible default covering the original 6 main types + room to grow."""
        default = {
            "COAD": 0,
            "OV": 1,
            "LIHC": 2,
            "PRAD": 3,
            "NSCLC": 4,
            "CESC": 5,
            "UNKNOWN": 99,
        }
        return cls(default)

    def __getitem__(self, name: str) -> int:
        name = name.upper()
        return self.name_to_idx.get(name, self.name_to_idx.get("UNKNOWN", 99))

    def get_name(self, idx: int) -> str:
        return self.idx_to_name.get(idx, "UNKNOWN")

    def __len__(self) -> int:
        return self.num_classes

    def __repr__(self) -> str:
        return f"CancerRegistry({len(self)} types)"
