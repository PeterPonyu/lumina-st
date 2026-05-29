"""Regression tests for issue #128 — guidance-scale validation.

Before the fix, ``cfg_scale`` flowed unchecked into the CFG blend
(``v_uncond + cfg_scale * (v_cond - v_uncond)``). NaN/inf silently produced
NaN latents and negative/absurd scales pushed wildly out of distribution with
no error. These tests pin that invalid guidance scales now raise a clear error
at both public entry points.
"""

import pytest
import torch

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.flow import create_flow_transport, FlowSampler
from lumina_st.flow.utils import validate_guidance_scale
from lumina_st.models.lumina_transformer import LuminaTransformer
from lumina_st.modules.lumina_flow_module import LuminaFlowModule


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0, 1e9])
def test_validate_guidance_scale_rejects_bad(bad):
    with pytest.raises((ValueError, TypeError)):
        validate_guidance_scale(bad)


@pytest.mark.parametrize("good", [0.0, 1.0, 3.0, 7.5])
def test_validate_guidance_scale_accepts_sane(good):
    assert validate_guidance_scale(good) == good


def _tiny_module():
    cfg = LuminaSTConfig(latent_dim=16)
    transformer = LuminaTransformer(
        latent_dim=16, patch_size=1, hidden_size=16, depth=1,
        num_heads=2, mlp_ratio=2.0, num_classes=4, class_dropout_prob=0.1,
    )
    return LuminaFlowModule(cfg, transformer)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -2.0])
def test_enhance_latent_rejects_bad_cfg_scale(bad):
    module = _tiny_module()
    z = torch.randn(2, 16)
    y = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError):
        module.enhance_latent(z, y, cfg_scale=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -3.0])
def test_sample_ode_rejects_bad_cfg_scale(bad):
    transport = create_flow_transport("linear", "velocity")
    sampler = FlowSampler(transport)

    class DummyModel(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            return x

    model = DummyModel()
    with pytest.raises(ValueError):
        sampler.sample_ode(model, shape=(2, 8), cfg_scale=bad)
