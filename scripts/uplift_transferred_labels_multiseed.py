#!/usr/bin/env python3
"""Multi-seed + negative-control transferred-label uplift for a no-native-GT target.

Wraps the single-seed ``scripts/uplift_transferred_labels.py`` logic (#337) to
produce a robust **mean ± std** uplift on the in-tissue COAD Visium target
(``coad_visium_10x``) paired with the matched COAD scRNA reference
(``lumina_ref_coad_gse132465``). This is the in-tissue replacement for the
earlier cross-tissue cervix -> COAD control (#296 / #315).

Protocol
--------
1. Transfer ``cell_type`` from the matched COAD scRNA reference onto the Visium
   target ONCE (``scanpy.ingest``), giving a fixed label proxy. We deliberately
   freeze the transferred labels so the reported ± std isolates **clustering
   stochasticity** (the actual uplift measurement) rather than conflating it
   with label-transfer noise.
2. For each clustering seed, score Leiden(latent_observed) and
   Leiden(latent_enhanced) against the frozen labels; report Δ = enhanced −
   observed per metric, aggregated as mean ± std across seeds.
3. NEGATIVE CONTROL — permute the frozen labels (destroying the label⇄latent
   correspondence) and repeat. A trustworthy pipeline collapses both agreements
   to ~0 with Δ ≈ 0; a non-negligible permuted Δ would mean the measured uplift
   is an artifact.

CIRCULARITY CAVEAT (unchanged from #337)
----------------------------------------
The enhancement is conditioned on the SAME reference the labels are transferred
from, biasing the uplift upward. A NEGATIVE Δ is trustworthy (washout kills
agreement vs any labels); a POSITIVE Δ is CONFOUNDED and must NOT be promoted.
The definitive native-GT matched verdict lives in #336 (st_COAD_test).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np

from lumina_st.metrics.further_metrics import clustering_agreement, silhouette_of_latent

# reuse the audited single-seed label-transfer routine (#337)
from uplift_transferred_labels import transfer_labels  # type: ignore

_METRICS = ("ari", "ami", "nmi", "homo")


def _mean_std(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def _uplift_over_seeds(lat_obs, lat_enh, labels, seeds):
    """Δ = enhanced − observed per metric, collected across clustering seeds."""
    per_metric = {m: [] for m in _METRICS}
    raw = []
    for s in seeds:
        cl_obs = clustering_agreement(lat_obs, labels, seed=s)
        cl_enh = clustering_agreement(lat_enh, labels, seed=s)
        raw.append({"seed": s, "observed": cl_obs, "enhanced": cl_enh})
        for m in _METRICS:
            per_metric[m].append(cl_enh[m] - cl_obs[m])
    agg = {f"uplift_{m}_transferred_delta": _mean_std(per_metric[m]) for m in _METRICS}
    return agg, raw


def main(a: argparse.Namespace) -> None:
    seeds = [int(s) for s in a.seeds.split(",")]
    enh = ad.read_h5ad(a.enhanced)
    assert "latent_observed" in enh.obsm and "latent_enhanced" in enh.obsm, "need both latents"
    lat_obs = np.asarray(enh.obsm["latent_observed"], dtype=np.float64)
    lat_enh = np.asarray(enh.obsm["latent_enhanced"], dtype=np.float64)

    ref = ad.read_h5ad(a.reference)
    tgt_X = enh.X.tocsr() if hasattr(enh.X, "tocsr") else np.asarray(enh.X, dtype=np.float32)
    tgt = ad.AnnData(X=tgt_X, obs=enh.obs.copy(), var=enh.var.copy())
    # freeze labels once (label-transfer seed fixed = first clustering seed)
    labels = transfer_labels(ref, tgt, a.label_key, seeds[0])

    real_agg, real_raw = _uplift_over_seeds(lat_obs, lat_enh, labels, seeds)

    # silhouette is seed-invariant (no clustering RNG) -> single value
    sil_obs = silhouette_of_latent(lat_obs, labels)
    sil_enh = silhouette_of_latent(lat_enh, labels)

    # negative control: permute the frozen labels, recompute uplift over seeds
    rng = np.random.default_rng(seeds[0])
    perm_labels = labels[rng.permutation(len(labels))]
    neg_agg, _ = _uplift_over_seeds(lat_obs, lat_enh, perm_labels, seeds)

    out = {
        "dataset_card_id": a.card_id,
        "pairing": "in-tissue: COAD Visium target x matched COAD scRNA reference (replaces cervix->COAD cross-tissue control)",
        "gt_source": "label-transfer (scanpy.ingest) from matched COAD scRNA ref -> Visium target",
        "gt_is_native": False,
        "seeds": seeds,
        "label_transfer_seed_frozen": seeds[0],
        "n_transferred_label_classes": int(len(set(map(str, labels)))),
        "circularity_caveat": (
            "Labels transferred from the SAME reference the enhancement is conditioned on. "
            "NEGATIVE delta is trustworthy (washout kills agreement vs any labels); "
            "POSITIVE delta is CONFOUNDED by shared reference structure -- do NOT promote on it. "
            "Definitive native-GT matched verdict: issue #336 (st_COAD_test)."
        ),
        "uplift_mean_std": real_agg,
        "uplift_silhouette_transferred_delta": float(sil_enh - sil_obs),
        "silhouette_observed_vs_transferred": float(sil_obs),
        "silhouette_enhanced_vs_transferred": float(sil_enh),
        "negative_control_permuted_labels": {
            "description": "frozen transferred labels permuted; Delta should be ~0 and agreements ~0",
            "uplift_mean_std": neg_agg,
        },
        "per_seed_raw": real_raw,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    print("\n=== Transferred-label uplift (mean +/- std; CIRCULAR -- see caveat) ===")
    for m in _METRICS:
        d = real_agg[f"uplift_{m}_transferred_delta"]
        n = neg_agg[f"uplift_{m}_transferred_delta"]
        print(f"  {m.upper():5s} Delta  {d['mean']:+.4f} +/- {d['std']:.4f}"
              f"   | neg-control {n['mean']:+.4f} +/- {n['std']:.4f}")
    print(f"  SILH  Delta  {sil_enh - sil_obs:+.4f} (seed-invariant)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--enhanced", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--label_key", default="cell_type")
    p.add_argument("--card_id", default="coad_visium_10x+lumina_ref_coad_gse132465")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--out", default="results/lumina-st/coad_visium/uplift_transferred_multiseed.json")
    main(p.parse_args())
