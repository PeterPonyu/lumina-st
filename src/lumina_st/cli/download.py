"""
CLI tool for downloading pre-trained VAE and Diffusion models for LuminaST.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

# No public LuminaST checkpoints are published yet. Keep this registry EMPTY so the CLI
# cannot fetch non-existent artifacts (issue #71). Populate only when a real host exists.
PUBLISHED_CHECKPOINTS: dict[str, dict[str, str]] = {}

NO_PUBLIC_CHECKPOINTS_MSG = (
    "No public LuminaST checkpoints are published. The downloader is disabled until a real "
    "artifact host exists; train locally or pass an explicit local checkpoint path."
)


def download_file(url: str, dest_path: str, dry_run: bool = False, timeout: float = 30.0) -> None:
    """Download a checkpoint without fabricating invalid fallback bytes.

    Dry-runs report the target only. Network failures raise so downstream
    ``torch.load`` cannot silently ingest a fake checkpoint.
    """
    print(f"Downloading {url} -> {dest_path}")
    if dry_run:
        print("[Dry-run] Would download file; no checkpoint placeholder written.")
        return

    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{dest_path}.tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(tmp_path, "wb") as out_file:
                out_file.write(response.read())
        os.replace(tmp_path, dest_path)
        print("Download complete.")
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="LuminaST Atlas Checkpoint Downloader")
    parser.add_argument("--cancer-type", type=str, required=True, help="Cancer type to download (e.g., CESC, COAD, NSCLC)")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without actual network requests")
    args = parser.parse_args()

    cancer = args.cancer_type.upper()
    if cancer not in PUBLISHED_CHECKPOINTS:
        parser.exit(2, f"{NO_PUBLIC_CHECKPOINTS_MSG} Requested cancer_type={cancer!r}.\n")

    diff_url = PUBLISHED_CHECKPOINTS[cancer]["diffusion"]
    vae_url = PUBLISHED_CHECKPOINTS[cancer]["vae"]

    os.makedirs(args.output_dir, exist_ok=True)
    diff_dest = os.path.join(args.output_dir, f"lumina_{cancer.lower()}_50.ckpt")
    vae_dest = os.path.join(args.output_dir, f"vae_{cancer.lower()}_50.ckpt")

    print(f"Retrieving checkpoints for {cancer}...")
    download_file(diff_url, diff_dest, dry_run=args.dry_run)
    download_file(vae_url, vae_dest, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Dry-run complete; no files written to {args.output_dir}/")
    else:
        print(f"Successfully registered {cancer} checkpoints in {args.output_dir}/")


if __name__ == "__main__":
    main()
