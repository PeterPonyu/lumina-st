#!/usr/bin/env python3
"""
Unified, offline-safe dataset fetch entrypoint for LuminaST.

Dispatches per-dataset fetches from the machine-readable registry in
``src/lumina_st/data/dataset_registry.py``. Mirrors the conventions of the
checkpoint downloader (``src/lumina_st/cli/download.py``): dry-run by default,
explicit ``--download`` gate, and **no fabricated URLs / no placeholder bytes**.

Policy
------
* ``--list`` / ``--help`` are fully offline (exit 0); no network is touched.
* A real download only happens for a dataset whose registry URL is *verified*
  AND when the operator passes ``--download``.
* Datasets flagged ``UrlStatus.UNVERIFIED`` are **never** auto-downloaded — the
  fetcher prints ``URL UNVERIFIED - see issue #N`` and exits non-zero, pointing
  the operator at the dataset page / Download tab.
* squidpy/scanpy/HuggingFace/Census loaders are described, not executed here;
  they are one-liners run inside the ``dl`` conda env (see DATASETS.md).

Raw downloads land under ``~/Desktop/ST_research/data_cache/raw/<id>/`` (the repo
``.gitignore`` ignores all of ``data/``); only docs + the registry are tracked.

Examples
--------
    python scripts/data/fetch_datasets.py --list
    python scripts/data/fetch_datasets.py --dataset her2st --dry-run
    python scripts/data/fetch_datasets.py --dataset her2st --download   # real fetch
    python scripts/data/fetch_datasets.py --dataset xenium_prime_breast # refuses (UNVERIFIED)
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

# Allow running as a plain script (python scripts/data/fetch_datasets.py) by
# adding the package src/ to sys.path when lumina_st is not installed.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lumina_st.data.dataset_registry import (  # noqa: E402
    DATASET_REGISTRY,
    DatasetSpec,
    LoaderKind,
    all_ids,
)

DEFAULT_CACHE_ROOT = Path.home() / "Desktop" / "ST_research" / "data_cache" / "raw"

# Loader kinds that are materialised by a Python one-liner in the `dl` env rather
# than a direct HTTP GET. We describe these instead of downloading.
_LOADER_HINTS: dict[LoaderKind, str] = {
    LoaderKind.SQUIDPY: "run in the `dl` env: import squidpy as sq; sq.datasets.<loader>()",
    LoaderKind.SCANPY: "run in the `dl` env: import scanpy as sc; sc.datasets.visium_sge(...)",
    LoaderKind.HUGGINGFACE: "huggingface_hub.hf_hub_download(repo_id='MahmoodLab/hest', ...)",
    LoaderKind.CENSUS: "pip install cellxgene-census; cellxgene_census.get_anndata(...)",
    LoaderKind.PAGE_BUNDLE: "resolve the Output Bundle from the dataset page's Download tab",
}


def list_datasets() -> None:
    """Print every registered dataset (offline)."""
    header = f"{'id':<28} {'tier':<4} {'status':<10} {'platform':<28} {'issues'}"
    print(header)
    print("-" * len(header))
    for spec in DATASET_REGISTRY.values():
        issues = ",".join(f"#{n}" for n in spec.issues)
        print(
            f"{spec.id:<28} {spec.tier:<4} {spec.url_status.value:<10} "
            f"{spec.platform[:28]:<28} {issues}"
        )
    print(
        f"\n{len(DATASET_REGISTRY)} datasets "
        f"({sum(1 for s in DATASET_REGISTRY.values() if s.verified)} verified, "
        f"{sum(1 for s in DATASET_REGISTRY.values() if not s.verified)} UNVERIFIED). "
        "UNVERIFIED datasets are never auto-downloaded."
    )


def describe(spec: DatasetSpec) -> None:
    """Print the full card for one dataset (offline)."""
    print(f"== {spec.name} ({spec.id}) ==")
    print(f"  tracking issues : {', '.join('#' + str(n) for n in spec.issues)}")
    print(f"  platform        : {spec.platform}")
    print(f"  tissue          : {spec.tissue}")
    print(f"  cancer_type     : {spec.cancer_type}")
    print(f"  accession       : {spec.accession}")
    print(f"  url ({spec.url_status.value}): {spec.url}")
    print(f"  loader          : {spec.loader.value}")
    print(f"  raw-count artifact: {spec.download_artifact}")
    print(f"  raw-count policy: {spec.raw_count_policy}")
    print(f"  contract mapping: {spec.contract_mapping}")
    if spec.notes:
        print(f"  notes           : {spec.notes}")


def _download_url(url: str, dest_path: Path, timeout: float = 30.0) -> None:
    """Download ``url`` to ``dest_path`` without fabricating fallback bytes."""
    print(f"Downloading {url} -> {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
            with open(tmp_path, "wb") as out_file:
                out_file.write(response.read())
        os.replace(tmp_path, dest_path)
        print("Download complete.")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def fetch(spec: DatasetSpec, cache_root: Path, dry_run: bool, download: bool) -> int:
    """Fetch (or describe) one dataset. Returns a process exit code."""
    describe(spec)
    dest_dir = cache_root / spec.id

    # 1. UNVERIFIED URLs are never auto-downloaded.
    if not spec.verified:
        issues = " / ".join(f"#{n}" for n in spec.issues)
        print(
            f"\n[REFUSED] URL UNVERIFIED - see issue {issues}. "
            "Resolve the artifact from the dataset page's Download tab; "
            "do NOT hardcode a guessed hotlink."
        )
        return 0 if (dry_run or not download) else 2

    # 2. Loader-based datasets (squidpy/scanpy/HF/Census/page bundle) are one-liners.
    if spec.loader is not LoaderKind.HTTP:
        hint = _LOADER_HINTS.get(spec.loader, "see DATASETS.md")
        print(f"\n[loader] {spec.loader.value}: {hint}")
        print(f"         materialise into: {dest_dir}/")
        return 0

    # 3. Direct HTTP artifact (verified URL).
    artifact_name = spec.url.rsplit("/", 1)[-1]
    dest_path = dest_dir / artifact_name
    if dry_run or not download:
        print(f"\n[dry-run] would download {spec.url}")
        print(f"          -> {dest_path}")
        print("          pass --download to fetch for real (no placeholder bytes written).")
        return 0

    _download_url(spec.url, dest_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LuminaST unified dataset fetcher (offline-safe; no fabricated URLs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="List all registered datasets and exit.")
    parser.add_argument("--dataset", help=f"Dataset id to fetch. One of: {', '.join(all_ids())}")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Raw-count cache root (default: {DEFAULT_CACHE_ROOT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe the fetch without downloading (default behaviour).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download verified-URL artifacts (otherwise dry-run only).",
    )
    args = parser.parse_args(argv)

    if args.list or not args.dataset:
        list_datasets()
        return 0

    if args.dataset not in DATASET_REGISTRY:
        print(f"Unknown dataset id {args.dataset!r}. Known ids: {', '.join(all_ids())}")
        return 2

    return fetch(
        DATASET_REGISTRY[args.dataset],
        cache_root=args.cache_root,
        dry_run=args.dry_run,
        download=args.download,
    )


if __name__ == "__main__":
    raise SystemExit(main())
