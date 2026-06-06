#!/usr/bin/env python3
"""Enhancement uplift for a matched ST target that has NO native ground truth.

The 10x COAD Visium target (``coad_visium_10x``) ships with no pathologist
annotation — ``obs`` is only the Visium grid. To score clustering uplift we
*transfer* labels from the matched same-tissue COAD scRNA reference
(``lumina_ref_coad_gse132465``, ``obs['cell_type']``) via ``scanpy.tl.ingest``,
then compare Leiden(latent_observed) vs Leiden(latent_enhanced) against those
transferred labels.

CIRCULARITY CAVEAT (read before trusting a POSITIVE delta)
----------------------------------------------------------
The enhancement is *conditioned on the same reference* the labels are
transferred from. Both the enhanced latent and the transferred labels therefore
inherit the reference's structure, which biases the uplift **upward**. The
asymmetry that makes this still useful:

  * a NEGATIVE uplift is trustworthy — the washout destroys cluster agreement
    against *any* label source, reference-derived or not;
  * a POSITIVE uplift is CONFOUNDED — it may simply reflect the shared reference
    structure, not a genuine gain. Do not promote on a positive number here.

The definitive matched-pair verdict lives in #336 (st_COAD_test, real *native*
GT). This script only corroborates on a second, cross-platform (Visium) target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

from lumina_st.data.datasets import align_to_shared_panel
from lumina_st.metrics.further_metrics import clustering_agreement, silhouette_of_latent


def transfer_labels(ref: ad.AnnData, target: ad.AnnData, label_key: str, seed: int) -> np.ndarray:
    """scanpy ingest: project target onto the reference PCA and transfer labels."""
    ref, target, stats = align_to_shared_panel(ref, target)
    print(f"[INFO] label-transfer on {stats['n_shared']} shared genes")
    for a in (ref, target):
        a.X = a.X.astype(np.float32)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    n_comps = min(50, ref.n_obs - 1, ref.n_vars - 1)
    sc.pp.pca(ref, n_comps=n_comps)
    sc.pp.neighbors(ref, n_neighbors=15, random_state=seed)
    sc.tl.umap(ref, random_state=seed)
    sc.tl.ingest(target, ref, obs=label_key)
    labels = target.obs[label_key].to_numpy()
    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"[INFO] transferred {label_key}: " + ", ".join(f"{u}={c}" for u, c in zip(uniq, cnt)))
    return labels


def main(a: argparse.Namespace) -> None:
    enh = ad.read_h5ad(a.enhanced)
    assert "latent_observed" in enh.obsm and "latent_enhanced" in enh.obsm, "need both latents"
    lat_obs = np.asarray(enh.obsm["latent_observed"], dtype=np.float64)
    lat_enh = np.asarray(enh.obsm["latent_enhanced"], dtype=np.float64)

    ref = ad.read_h5ad(a.reference)
    # target gene space = the enhanced object's panel; build a counts AnnData for ingest
    tgt_X = enh.X.tocsr() if hasattr(enh.X, "tocsr") else np.asarray(enh.X, dtype=np.float32)
    tgt = ad.AnnData(X=tgt_X, obs=enh.obs.copy(), var=enh.var.copy())
    labels = transfer_labels(ref, tgt, a.label_key, a.seed)

    cl_obs = clustering_agreement(lat_obs, labels, seed=a.seed)
    cl_enh = clustering_agreement(lat_enh, labels, seed=a.seed)
    sil_obs = silhouette_of_latent(lat_obs, labels)
    sil_enh = silhouette_of_latent(lat_enh, labels)

    deltas = {f"uplift_{m}_transferred_delta": cl_enh[m] - cl_obs[m] for m in ("ari", "ami", "nmi", "homo")}
    deltas["uplift_silhouette_transferred_delta"] = sil_enh - sil_obs

    out = {
        "dataset_card_id": a.card_id,
        "gt_source": "label-transfer (scanpy.ingest) from matched COAD scRNA ref -> Visium target",
        "gt_is_native": False,
        "circularity_caveat": (
            "Labels transferred from the SAME reference the enhancement is conditioned on. "
            "NEGATIVE delta is trustworthy (washout kills agreement vs any labels); "
            "POSITIVE delta is CONFOUNDED by shared reference structure — do NOT promote on it. "
            "Definitive native-GT matched verdict: issue #336 (st_COAD_test)."
        ),
        "n_transferred_label_classes": int(len(set(map(str, labels)))),
        "cluster_observed_vs_transferred": cl_obs,
        "cluster_enhanced_vs_transferred": cl_enh,
        "silhouette_observed_vs_transferred": sil_obs,
        "silhouette_enhanced_vs_transferred": sil_enh,
        **deltas,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("\n=== Uplift vs transferred labels (CIRCULAR — see caveat) ===")
    for k in ("ari", "ami", "nmi", "homo"):
        print(f"  uplift_{k}_transferred_delta: {deltas[f'uplift_{k}_transferred_delta']:+.4f} "
              f"(obs {cl_obs[k]:.4f} -> enh {cl_enh[k]:.4f})")
    print(f"  uplift_silhouette_transferred_delta: {deltas['uplift_silhouette_transferred_delta']:+.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--enhanced", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--label_key", default="cell_type")
    p.add_argument("--card_id", default="coad_visium_10x+lumina_ref_coad_gse132465")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/lumina-st/coad_visium/uplift_transferred.json")
    main(p.parse_args())
