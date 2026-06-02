"""Centralized reproducibility control for LuminaST (issue #180).

LuminaST seeds RNGs in several ad-hoc places (``pl.seed_everything`` inside
``LuminaImputer.fit``/``enhance``, ``torch.Generator`` for DataLoaders, per-test
seeding). This module provides a single, documented, package-level entry point
that pins **every** random source the package can reach so multi-seed
experiments, ablation grids, and real-data runs are reproducible by
construction. It mirrors the contract of ``factorgraph_st.repro.set_seed``.

Usage::

    from lumina_st import set_seed
    set_seed(42)            # Python, NumPy, torch, torch.cuda all seeded
    state = set_seed(42)    # returns the applied seed for logging/manifests
"""

from __future__ import annotations

import os
import random as _py_random

import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: int, *, deterministic_torch: bool = False) -> int:
    """Seed Python ``random``, NumPy, and PyTorch (CPU + CUDA) from one place.

    Args:
        seed: Integer seed applied to every reachable RNG. Also exported to
            ``PYTHONHASHSEED`` for child processes spawned after this call.
        deterministic_torch: When True, also flips cuDNN into deterministic
            mode (``cudnn.deterministic=True``, ``cudnn.benchmark=False``).
            Off by default because it can slow training; turn it on for
            bit-exact reproducibility runs.

    Returns:
        The integer seed that was applied, so callers can record it in a run
        manifest or log line.

    Notes:
        ``torch`` is a hard dependency of LuminaST (unlike factorgraph-st where
        it is optional), so it is always seeded here. CUDA is seeded only when
        a device is available.
    """
    s = int(seed)
    os.environ["PYTHONHASHSEED"] = str(s)
    _py_random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if deterministic_torch:
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn is not None:
            cudnn.deterministic = True
            cudnn.benchmark = False
    return s
