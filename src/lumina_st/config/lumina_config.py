"""
LuminaST Configuration — Pydantic v2 models.

This replaces the dozens of argparse groups in the legacy argparse-based configuration
with a single, validated, serializable, and reproducible configuration object.

All training, data, and inference hyperparameters live here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, List, Dict, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict


class LuminaSTConfig(BaseModel):
    """Master configuration for LuminaST enhancement / imputation runs."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # ------------------------------------------------------------------
    # Experiment identity
    # ------------------------------------------------------------------
    experiment_name: str = "lumina_default"
    seed: int = 42

    # ------------------------------------------------------------------
    # Data paths & cancer handling
    # ------------------------------------------------------------------
    data_root: Path = Field(default=Path("data/processed"))
    cancer_types: List[str] = Field(default_factory=lambda: ["COAD"])
    # Registry file (YAML/JSON) that maps cancer names -> integer indices
    cancer_registry_file: Optional[Path] = None

    # Gene space
    n_genes: int = 10000  # after HVG selection in processor
    gene_sparsity_ratio_file: Optional[Path] = None

    # ------------------------------------------------------------------
    # Latent VAE (encoder) settings
    # ------------------------------------------------------------------
    latent_dim: int = 50
    vae_checkpoint: Optional[Path] = None
    freeze_vae_encoder: bool = True
    vae_batch_key: str = "batch"  # or "Tumor Type" in original data
    latent_encoder_backend: Literal["tiny_vae", "scvi", "scgpt", "nicheformer", "geneformer", "uce"] = "tiny_vae"
    foundation_embedding_dim: Optional[int] = None
    foundation_checkpoint: Optional[Path] = None

    # ------------------------------------------------------------------
    # Flow / Diffusion model architecture (the LuminaTransformer)
    # ------------------------------------------------------------------
    patch_size: int = 1  # for latent vectors, 1 is common
    hidden_size: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1  # for CFG

    # ------------------------------------------------------------------
    # Flow matching training
    # ------------------------------------------------------------------
    path_type: Literal["linear", "gvp", "vp"] = "linear"
    prediction: Literal["velocity", "noise", "score"] = "velocity"
    loss_weight: Optional[Literal["velocity", "likelihood"]] = None

    lr: float = 2e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.9999

    train_eps: float = 0.0
    sample_eps: float = 0.0

    batch_size: int = 128
    num_workers: int = 4
    max_epochs: int = 100
    gradient_clip_val: float = 1.0

    # ------------------------------------------------------------------
    # Sampling / Imputation (inference)
    # ------------------------------------------------------------------
    guidance_scale: float = 3.0
    t_forward: float = 0.9  # how far to noise the observed genes before denoising
    sampling_method: str = "dopri5"
    num_sampling_steps: int = 50
    atol: float = 1e-5
    rtol: float = 1e-5

    # Sparsity post-processing (important for realistic gene counts).
    # Default is OFF (#142): the per-cell 95th-percentile fallback was
    # zeroing the bottom ~95% of every cell's imputation BEFORE held-out
    # gene Pearson was scored, so default benchmark numbers measured the
    # sparsifier, not the model. Callers who need sparsity matching must
    # opt in explicitly and document which sparsity mode produced each
    # reported metric.
    apply_sparsity: bool = False
    sparsity_percentile: float = 0.95

    # ------------------------------------------------------------------
    # Logging & output
    # ------------------------------------------------------------------
    output_dir: Path = Field(default=Path("results"))
    use_wandb: bool = False
    wandb_project: str = "lumina-st"

    # ------------------------------------------------------------------
    # Validation & reproducibility
    # ------------------------------------------------------------------
    @field_validator("cancer_types")
    @classmethod
    def validate_cancers(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one cancer_type must be provided")
        return [c.upper() for c in v]

    def get_cancer_index(self, name: str) -> int:
        """Return the integer index of ``name`` within ``cancer_types``.

        Names are matched case-insensitively — the ``cancer_types`` field
        validator upper-cases every entry, so any case maps to the same
        index. The full ``CancerRegistry`` in ``data/cancer_registry.py`` is
        the runtime-mutable version of this mapping; this method gives
        callers that already hold a ``LuminaSTConfig`` a self-contained
        lookup using the same convention.

        Raises:
            KeyError: if ``name`` is not present in ``self.cancer_types``.
        """
        key = name.upper()
        try:
            return self.cancer_types.index(key)
        except ValueError as exc:
            raise KeyError(
                f"Unknown cancer type {name!r}; configured cancer_types="
                f"{self.cancer_types}"
            ) from exc

    def model_dump_for_checkpoint(self) -> Dict[str, Any]:
        """Safe subset to store next to checkpoints."""
        d = self.model_dump()
        # Convert Paths to strings for JSON
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d
