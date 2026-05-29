"""Regression tests for lumina-st checkpoint persistence.

#118 — ``LuminaImputer.from_checkpoint`` previously called
``module.load_state_dict(..., strict=False)``, silently leaving missing
parameters at random init. Now routed through ``_strict_load_state_dict``
which raises ``RuntimeError`` with the missing/unexpected key list.

#147 — ``scripts/e2e/train_latent_flow.py`` previously persisted only
``module.transformer.state_dict()``. Inference samples exclusively from
``self.ema_model`` (``lumina_flow_module.py:134/148/185``), so the EMA
weights used at enhancement time were not in the checkpoint at all —
reloads silently regenerated EMA from random init. The fix saves the
full ``module.state_dict()``, which contains both ``transformer.*`` and
``ema_model.*`` prefixes that ``from_checkpoint`` already understands.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import _strict_load_state_dict, save_flow_checkpoint
from lumina_st.models.lumina_transformer import LuminaTransformer
from lumina_st.modules.lumina_flow_module import LuminaFlowModule


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


# ---------------------------------------------------------------------------
# #147 — EMA persistence
# ---------------------------------------------------------------------------


def _build_minimal_module() -> LuminaFlowModule:
    cfg = LuminaSTConfig(
        latent_dim=8,
        hidden_size=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        cancer_types=["A", "B"],
    )
    transformer = LuminaTransformer(
        latent_dim=cfg.latent_dim,
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        num_classes=len(cfg.cancer_types),
        class_dropout_prob=cfg.class_dropout_prob,
    )
    return LuminaFlowModule(cfg, transformer)


def test_ema_state_persisted_and_restored(tmp_path) -> None:
    """The checkpoint format used by ``train_latent_flow.py`` must round-trip
    the EMA branch param-by-param.

    Pre-fix: ``torch.save({"state_dict": module.transformer.state_dict()})``
    dropped every ``ema_model.*`` key, so reloading regenerated EMA from
    random init even though guided sampling reads exclusively from EMA.

    Post-fix: ``torch.save({"state_dict": module.state_dict()})`` carries
    both ``transformer.*`` and ``ema_model.*``, and ``_strict_load_state_dict``
    refuses any silent mismatch.
    """

    torch.manual_seed(0)
    original = _build_minimal_module()

    # Simulate the post-training EMA having drifted off the transformer weights
    # (the whole point of EMA — and the part the pre-fix save threw away).
    with torch.no_grad():
        for p in original.ema_model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    # Snapshot EMA params before save so we can compare bit-for-bit on reload.
    ema_before = {n: p.detach().clone() for n, p in original.ema_model.state_dict().items()}

    # Persist using the SAME schema train_latent_flow.py emits after the fix.
    ckpt_path = tmp_path / "lumina_flow.ckpt"
    torch.save({"state_dict": original.state_dict()}, ckpt_path)

    # Saved checkpoint must literally contain ema_model.* keys — otherwise the
    # fix regressed back to transformer-only persistence.
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    saved_state = saved["state_dict"]
    assert any(k.startswith("ema_model.") for k in saved_state), (
        "Checkpoint missing ema_model.* keys — EMA weights would not survive reload."
    )

    # Reload into a freshly-initialized module and verify EMA equality.
    torch.manual_seed(1)  # different seed: any leaked random init would diverge
    restored = _build_minimal_module()
    _strict_load_state_dict(restored, saved_state)

    for name, param in restored.ema_model.state_dict().items():
        ref = ema_before[name]
        assert torch.equal(param, ref), (
            f"EMA parameter '{name}' did not round-trip through save/reload."
        )


def test_legacy_transformer_only_save_fails_loudly() -> None:
    """The pre-fix save shape — ``module.transformer.state_dict()`` — must
    now be rejected by ``_strict_load_state_dict``.

    Before #118 + #147, loading this shape silently left both
    ``transformer.*`` and ``ema_model.*`` keys at random init. The combined
    fix turns that into a loud ``RuntimeError`` so the user notices the
    broken artifact instead of publishing metrics off random weights.
    """

    module = _build_minimal_module()
    bad_state = module.transformer.state_dict()  # missing 'transformer.' / 'ema_model.' prefixes

    with pytest.raises(RuntimeError):
        _strict_load_state_dict(module, bad_state)
