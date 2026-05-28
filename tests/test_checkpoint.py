"""Regression test for lumina-st #118.

``LuminaImputer.from_checkpoint`` previously called
``module.load_state_dict(..., strict=False)``. When a checkpoint's keys did
not match the current model definition (renamed layer, refactored module,
mismatched arch), the missing parameters were silently left at random init
— producing a "loaded" model whose published metrics actually came from
random weights.

The fix routes checkpoint loading through ``_strict_load_state_dict`` which
raises a ``RuntimeError`` enumerating both the missing and the unexpected
keys so the mismatch is impossible to miss.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from lumina_st.core.lumina_imputer import _strict_load_state_dict


def test_renamed_key_raises_with_key_list() -> None:
    """A renamed key must trigger ``RuntimeError`` and the message must list
    both the missing model-side key and the unexpected checkpoint-side key.
    """

    module = nn.Linear(3, 4)
    real_state = module.state_dict()

    # Simulate a checkpoint where ``weight`` was renamed to ``weight_renamed``.
    bad_state = {
        "weight_renamed": real_state["weight"].clone(),
        "bias": real_state["bias"].clone(),
    }

    with pytest.raises(RuntimeError) as excinfo:
        _strict_load_state_dict(module, bad_state)

    msg = str(excinfo.value)
    # The unexpected (checkpoint-only) key must appear in the error.
    assert "weight_renamed" in msg, msg
    # The missing (model-only) key must appear in the error.
    assert "weight" in msg, msg
    # The message must make it clear loading was refused.
    assert "strict=False" in msg or "random init" in msg or "Refusing" in msg, msg


def test_matching_keys_load_silently() -> None:
    """A state_dict whose keys exactly match must load without raising."""

    module = nn.Linear(3, 4)
    state = module.state_dict()

    # Mutate the values so we can verify the load actually happened.
    new_state = {k: torch.zeros_like(v) for k, v in state.items()}

    _strict_load_state_dict(module, new_state)

    for name, param in module.state_dict().items():
        assert torch.equal(param, torch.zeros_like(param)), name
