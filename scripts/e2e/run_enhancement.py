#!/usr/bin/env python3
"""
Inference script: enhance a spatial transcriptomics slice with a trained LuminaST model.

Example:
    python scripts/run_enhancement.py \
        --checkpoint results/lumina_coad/lumina_flow.ckpt \
        --input data/processed/st_COAD_test.h5ad \
        --output results/enhanced_COAD.h5ad \
        --cancer COAD
"""

import argparse
from pathlib import Path

import scanpy as sc
import torch

from lumina_st.config import LuminaSTConfig
from lumina_st.core.lumina_imputer import LuminaImputer


def main(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = LuminaSTConfig(**ckpt.get("config", {}))

    # Rebuild model
    imputer = LuminaImputer.from_config(cfg)

    # Load weights
    imputer.module.transformer.load_state_dict(ckpt["state_dict"])
    imputer.module.ema_model.load_state_dict(ckpt["state_dict"])

    # Enhance
    adata = sc.read_h5ad(args.input)
    enhanced = imputer.enhance(adata, cancer_type=args.cancer)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    enhanced.write(args.output)
    print(f"Enhanced AnnData written to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cancer", default=None)
    args = parser.parse_args()
    main(args)
