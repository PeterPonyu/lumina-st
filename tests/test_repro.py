"""Regression tests for the central reproducibility contract (issue #180)."""

from __future__ import annotations

import random

import numpy as np
import torch

import lumina_st
from lumina_st import set_seed


def test_set_seed_is_exported_at_package_level():
    assert hasattr(lumina_st, "set_seed")
    assert lumina_st.set_seed is set_seed


def test_set_seed_returns_applied_seed():
    assert set_seed(123) == 123


def test_set_seed_makes_python_numpy_torch_deterministic():
    set_seed(7)
    py_a = [random.random() for _ in range(5)]
    np_a = np.random.rand(5)
    torch_a = torch.rand(5)

    set_seed(7)
    py_b = [random.random() for _ in range(5)]
    np_b = np.random.rand(5)
    torch_b = torch.rand(5)

    assert py_a == py_b
    np.testing.assert_array_equal(np_a, np_b)
    assert torch.equal(torch_a, torch_b)


def test_distinct_seeds_diverge():
    set_seed(1)
    a = torch.rand(8)
    set_seed(2)
    b = torch.rand(8)
    assert not torch.equal(a, b)


def test_set_seed_sets_pythonhashseed(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    set_seed(2024)
    import os

    assert os.environ["PYTHONHASHSEED"] == "2024"
