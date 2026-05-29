# LuminaST — Dataset Literature Links

Source paper / portal for every dataset enumerated in
[`docs/DATASETS.md`](../docs/DATASETS.md) and the machine-readable
[`src/lumina_st/data/dataset_registry.py`](../src/lumina_st/data/dataset_registry.py).
`citation_key` matches the `DatasetSpec.citation_key` field in that registry.

> Canonical source of truth for accessions: `ST_research/datasets/DATASET_REGISTRY.md`
> (2026-05-28/29). Where a download hotlink is unconfirmed the link below is the
> **canonical landing page**, carrying the same `⚠️ UNVERIFIED` flag as the registry —
> never a guessed hotlink.

| `citation_key` | Dataset | Issue(s) | Source paper / portal |
|----------------|---------|----------|-----------------------|

## Related infrastructure

- **#71** — *publish/register real LuminaST checkpoints*: the checkpoint download CLI
  (`src/lumina_st/cli/download.py`) is infra-adjacent to dataset ingestion; its
  fabricated `spatial-omics/lumina-st` HF URLs are flagged in `docs/DATASETS.md`
  §"UNVERIFIED-URL caveats" (caveat #3) and tracked separately.
- **#182** — meta-issue requesting these 05-29 cards/fetchers/literature links be
  consolidated into draft PR #61 (this PR).
