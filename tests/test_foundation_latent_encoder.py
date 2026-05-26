from __future__ import annotations

import pytest
import torch

from lumina_st.latents.foundation import (
    FoundationLatentEncoder,
    inspect_foundation_backend,
    rank_foundation_options,
)


class DummyFoundation(torch.nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(5, out_dim)

    def forward(self, x, y=None):
        return {"cell_emb": self.linear(x)}


def test_foundation_encoder_projects_and_decodes():
    encoder = FoundationLatentEncoder(
        encoder=DummyFoundation(out_dim=7),
        input_dim=5,
        embedding_dim=7,
        latent_dim=3,
    )
    x = torch.randn(4, 5)
    y = torch.zeros(4, dtype=torch.long)
    z, info = encoder.encode_to_latent(x, y)
    recon = encoder.decode_from_latent(z, y, info["library"])
    assert z.shape == (4, 3)
    assert info["foundation_embedding"].shape == (4, 7)
    assert recon.shape == x.shape


def test_foundation_encoder_rejects_wrong_embedding_dim():
    encoder = FoundationLatentEncoder(
        encoder=DummyFoundation(out_dim=6),
        input_dim=5,
        embedding_dim=7,
        latent_dim=3,
    )
    with pytest.raises(ValueError, match="embedding dim mismatch"):
        encoder.encode_to_latent(torch.randn(2, 5), None)


def test_backend_inspection_reports_optional_dependency_without_crashing():
    info = inspect_foundation_backend("nicheformer")
    assert info.name == "nicheformer"
    assert info.reference.startswith("https://")
    assert isinstance(info.available, bool)


def test_rank_foundation_options_prefers_spatial_models():
    assert rank_foundation_options()[0] == "nicheformer"
    assert "scgpt" in rank_foundation_options()
