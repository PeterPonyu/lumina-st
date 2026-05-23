from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lumina_st.config.lumina_config import LuminaSTConfig


def train_encoder(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a LuminaST encoder/flow run.")
    parser.add_argument("--reference", help="Reference atlas .h5ad path.")
    parser.add_argument("--output-dir", default="results/lumina-train", help="Output directory.")
    parser.add_argument("--cancer-type", default="COAD", help="Cancer type label for the run.")
    parser.add_argument("--max-epochs", type=int, default=LuminaSTConfig().max_epochs)
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments without training.")
    args = parser.parse_args(argv)

    cfg = LuminaSTConfig(cancer_types=[args.cancer_type], max_epochs=args.max_epochs)
    output_dir = Path(args.output_dir)

    if args.dry_run:
        print("LuminaST training dry run")
        print(f"  reference: {args.reference or '(synthetic/e2e fallback)'}")
        print(f"  output_dir: {output_dir}")
        print(f"  cancer_types: {', '.join(cfg.cancer_types)}")
        print(f"  max_epochs: {cfg.max_epochs}")
        return 0

    parser.error(
        "training data execution is provided by scripts/e2e/enhance_real_st.py today; "
        "rerun this command with --dry-run to validate package entry-point wiring"
    )
    return 2


def run_enhancement(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LuminaST enhancement.")
    parser.add_argument("--target", help="Target spatial transcriptomics .h5ad path.")
    parser.add_argument("--output", default="lumina_enhanced.h5ad", help="Output .h5ad path.")
    parser.add_argument("--cancer-type", default="COAD", help="Cancer type label for the target slice.")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments without enhancement.")
    args = parser.parse_args(argv)

    cfg = LuminaSTConfig(cancer_types=[args.cancer_type])

    if args.dry_run:
        print("LuminaST enhancement dry run")
        print(f"  target: {args.target or '(synthetic/e2e fallback)'}")
        print(f"  output: {Path(args.output)}")
        print(f"  cancer_types: {', '.join(cfg.cancer_types)}")
        return 0

    parser.error(
        "enhancement execution is provided by scripts/e2e/enhance_real_st.py today; "
        "rerun this command with --dry-run to validate package entry-point wiring"
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LuminaST command-line tools.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train_parser = subcommands.add_parser("train-encoder", help="Validate/train the encoder flow entry point.")
    train_parser.set_defaults(handler=train_encoder)

    enhance_parser = subcommands.add_parser("run-enhance", help="Validate/run the enhancement entry point.")
    enhance_parser.set_defaults(handler=run_enhancement)

    args, remaining = parser.parse_known_args(argv)
    return args.handler(remaining)


__all__ = ["main", "run_enhancement", "train_encoder"]
