"""
Proper pytest tests for LuminaST (and shared flow).

Run with:
    pytest tests/ -q --tb=line
"""

import numpy as np
import anndata as ad
import pytorch_lightning as pl
import torch

from lumina_st.flow import create_flow_transport, LinearPath
from lumina_st.config.lumina_config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer
from lumina_st.latents.tiny_vae import TinyVAE
from lumina_st.models.lumina_transformer import LuminaTransformer


def test_linear_path_velocity():
    path = LinearPath()
    t = torch.tensor([0.3, 0.7])
    x0 = torch.randn(2, 8)
    x1 = torch.randn(2, 8)
    _, _, ut = path.plan(t, x0, x1)
    assert torch.allclose(ut[0], x1[0] - x0[0], atol=1e-5)


def test_flow_transport_loss_finite():
    transport = create_flow_transport("linear", "velocity")

    class DummyModel(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            return x  # velocity prediction = identity for test

    model = DummyModel()
    x1 = torch.randn(4, 16)
    losses = transport.training_losses(model, x1, {"y": torch.zeros(4, dtype=torch.long)})
    assert torch.isfinite(losses["loss"])


def test_lumina_transformer_shapes():
    model = LuminaTransformer(
        latent_dim=32, patch_size=1, hidden_size=32, depth=2,
        num_heads=2, mlp_ratio=4.0, num_classes=5, class_dropout_prob=0.1
    )
    x = torch.randn(3, 32)
    t = torch.rand(3)
    y = torch.randint(0, 5, (3,))
    out = model(x, t, y)
    assert out.shape == (3, 32)


def test_tiny_vae_roundtrip():
    vae = TinyVAE(48, 16)
    x = torch.randn(5, 48)
    z, _ = vae.encode_to_latent(x)
    recon = vae.decode_from_latent(z)
    assert recon.shape == x.shape


def test_vectorized_sparsity_correctness():
    # Setup test matrix N x G
    N, G = 50, 100
    np.random.seed(123)
    torch.manual_seed(123)
    
    mat_np = np.random.uniform(0, 10, (N, G)).astype(np.float32)
    vals_np = np.random.uniform(0, 1, G).astype(np.float32)
    
    # Baseline CPU loop implementation
    res_np = mat_np.copy()
    for i in range(G):
        ratio = vals_np[i]
        thresh = np.percentile(res_np[:, i], ratio * 100)
        res_np[res_np[:, i] < thresh, i] = 0.0

    # Vectorized PyTorch implementation
    mat_torch = torch.tensor(mat_np)
    vals_tensor = torch.tensor(vals_np)
    
    sorted_data, _ = torch.sort(mat_torch, dim=0)
    indices = vals_tensor * (N - 1)
    indices_low = torch.floor(indices).long().clamp(0, N - 1)
    indices_high = torch.ceil(indices).long().clamp(0, N - 1)
    weight = indices - indices_low.float()
    
    val_low = torch.gather(sorted_data, 0, indices_low.unsqueeze(0))
    val_high = torch.gather(sorted_data, 0, indices_high.unsqueeze(0))
    thresh_tensor = val_low + weight.unsqueeze(0) * (val_high - val_low)
    
    mask = mat_torch < thresh_tensor
    active_cols = (vals_tensor > 0.0).unsqueeze(0)
    mask = mask & active_cols
    mat_torch[mask] = 0.0
    
    res_pt = mat_torch.cpu().numpy()
    
    # Assert exact match
    assert np.allclose(res_np, res_pt, atol=1e-6)


def test_lumina_imputer_fit_runs_one_training_batch(tmp_path):
    rng = np.random.default_rng(7)
    # Raw integer counts — ReferenceAtlasDataset enforces the raw-count input
    # contract (#314); rng.normal() would produce negative, non-integer values
    # that (correctly) trip the log1p-normalized guard.
    adata = ad.AnnData(
        X=rng.poisson(3.0, size=(8, 8)).astype(np.float32),
        obs={"cancer_type": ["COAD"] * 8},
    )
    cfg = LuminaSTConfig(
        latent_dim=8,
        hidden_size=16,
        depth=1,
        num_heads=2,
        vae_batch_key="cancer_type",
        batch_size=4,
        num_workers=0,
        max_epochs=1,
        output_dir=tmp_path,
    )
    imputer = LuminaImputer.from_config(cfg)
    trainer = pl.Trainer(
        fast_dev_run=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    returned = imputer.fit(trainer=trainer, reference_adata=adata)

    assert returned is trainer
    assert trainer.global_step == 1
