"""Experiment harness primitives for LuminaST (issues #179 / #181).

This subpackage defines the reproducible-run contract that training and
inference paths emit so every run is self-describing. The first concrete
artifact is :class:`RunManifest` — a serializable snapshot of seeds, config,
environment, checkpoint/eval intent, and sweep parameters.
"""

from __future__ import annotations

from .run import ExperimentRun, run_experiment
from .run_manifest import RunManifest

__all__ = ["ExperimentRun", "RunManifest", "run_experiment"]
