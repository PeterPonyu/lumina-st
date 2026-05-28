
import anndata as ad
import numpy as np
import pytest
import torch

from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer
from lumina_st.data.cancer_registry import CancerRegistry
from lumina_st.flow import GVPPath, LinearPath, VPPath, create_flow_transport
from lumina_st.models.attention_mlp import Attention
from lumina_st.models.embeddings import LabelEmbedder


def test_default_registry_unknown_is_in_embedding_range() -> None:
    registry = CancerRegistry.default_pan_cancer()

    assert registry["UNKNOWN"] == len(registry) - 1
    assert registry["not-a-real-cancer"] == registry["UNKNOWN"]

    embedder = LabelEmbedder(num_classes=len(registry), hidden_size=8, dropout_prob=0.1)
    out = embedder(torch.full((3,), registry["UNKNOWN"]), train=False)
    assert out.shape == (3, 8)


def test_registry_len_covers_non_contiguous_custom_indices() -> None:
    registry = CancerRegistry({"COAD": 0, "UNKNOWN": 99})

    assert len(registry) == 100
    assert registry["missing"] == 99


def test_enhance_unknown_cancer_runs_without_index_error() -> None:
    rng = np.random.default_rng(123)
    adata = ad.AnnData(X=rng.normal(size=(4, 6)).astype(np.float32))
    cfg = LuminaSTConfig(
        cancer_types=["COAD"],
        latent_dim=6,
        hidden_size=16,
        depth=1,
        num_heads=2,
        num_sampling_steps=2,
        guidance_scale=1.0,
        apply_sparsity=False,
        seed=11,
    )
    imputer = LuminaImputer.from_config(cfg)

    enhanced = imputer.enhance(adata, cancer_type="COAD")

    assert "imputed" in enhanced.layers
    assert enhanced.layers["imputed"].shape == adata.X.shape


def test_enhance_latent_class_dropout_zero_disables_cfg_not_index_error() -> None:
    cfg = LuminaSTConfig(
        latent_dim=4,
        hidden_size=8,
        depth=1,
        num_heads=2,
        class_dropout_prob=0.0,
        guidance_scale=3.0,
        num_sampling_steps=2,
        apply_sparsity=False,
    )
    module = LuminaImputer.from_config(cfg).module
    z = torch.randn(2, 4)
    y = torch.zeros(2, dtype=torch.long)

    out = module.enhance_latent(z, y, seed=7)

    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_enhance_is_deterministic_with_config_seed() -> None:
    rng = np.random.default_rng(456)
    adata = ad.AnnData(X=rng.normal(size=(3, 5)).astype(np.float32))
    cfg = LuminaSTConfig(
        cancer_types=["COAD"],
        latent_dim=5,
        hidden_size=16,
        depth=1,
        num_heads=2,
        num_sampling_steps=3,
        guidance_scale=1.0,
        apply_sparsity=False,
        seed=1234,
    )
    imputer = LuminaImputer.from_config(cfg)

    a = imputer.enhance(adata, cancer_type="COAD").layers["imputed"]
    b = imputer.enhance(adata, cancer_type="COAD").layers["imputed"]

    assert np.array_equal(a, b)


def test_attention_sdpa_matches_manual_path_in_eval() -> None:
    torch.manual_seed(2)
    attn = Attention(dim=12, num_heads=3, attn_drop=0.0, proj_drop=0.0).eval()
    x = torch.randn(2, 5, 12)

    with torch.no_grad():
        sdpa_out = attn(x)
        b, n, c = x.shape
        qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, c // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) * attn.scale
        manual = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, c)
        manual = attn.proj_drop(attn.proj(manual))

    assert torch.allclose(sdpa_out, manual, atol=1e-5)


@pytest.mark.parametrize("path", [LinearPath(), GVPPath(), VPPath()])
def test_path_derivatives_match_finite_difference(path) -> None:
    t = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    h = 1e-5
    _, da = path.alpha(t)
    _, ds = path.sigma(t)
    a_plus, _ = path.alpha(t + h)
    a_minus, _ = path.alpha(t - h)
    s_plus, _ = path.sigma(t + h)
    s_minus, _ = path.sigma(t - h)

    assert torch.allclose(da, (a_plus - a_minus) / (2 * h), atol=1e-4, rtol=1e-4)
    assert torch.allclose(ds, (s_plus - s_minus) / (2 * h), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("path", [GVPPath(), VPPath()])
def test_variance_preserving_paths_have_unit_variance(path) -> None:
    t = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    a, _ = path.alpha(t)
    s, _ = path.sigma(t)
    assert torch.allclose(a**2 + s**2, torch.ones_like(t), atol=1e-6)


def test_linear_path_noise_and_score_round_trip_to_velocity() -> None:
    path = LinearPath()
    t = torch.tensor([0.3, 0.7])
    x0 = torch.randn(2, 4)
    x1 = torch.randn(2, 4)
    _, xt, velocity = path.plan(t, x0, x1)

    noise = path.velocity_to_noise(velocity, xt, t)
    score = path.velocity_to_score(velocity, xt, t)

    assert torch.allclose(noise, x0, atol=1e-5)
    assert torch.allclose(path.noise_to_velocity(noise, xt, t), velocity, atol=1e-5)
    assert torch.allclose(path.score_to_velocity(score, xt, t), velocity, atol=1e-5)


def test_transport_converts_noise_prediction_to_velocity_for_sampling() -> None:
    transport = create_flow_transport("linear", "noise")
    t = torch.tensor([0.25, 0.75])
    x0 = torch.randn(2, 3)
    x1 = torch.randn(2, 3)
    _, xt, velocity = transport.path.plan(t, x0, x1)

    class PerfectNoise(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            return x0

    drift = transport.get_drift(PerfectNoise())
    assert torch.allclose(drift(xt, t), velocity, atol=1e-5)
