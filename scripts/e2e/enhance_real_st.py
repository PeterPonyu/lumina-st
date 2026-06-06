#!/usr/bin/env python3
"""
Practical End-to-End script for LuminaST enhancement.

If you do **not** have suitable real local ST data yet (common situation), the script
automatically falls back to high-quality synthetic data that mimics
a pan-cancer spatial-transcriptomics reference baseline.

Usage with real data (recommended when available):
    conda run -n dl python scripts/e2e/enhance_real_st.py \
        --reference /path/to/reference_atlas.h5ad \
        --target    /path/to/real_st_slice.h5ad \
        --cancer    COAD \
        --output    ./enhanced.h5ad

Usage with synthetic data (works today, produces meaningful metrics):
    conda run -n dl python scripts/e2e/enhance_real_st.py --output ./synthetic_enhanced.h5ad

The script always prints training curves + final enhancement metrics so you can
judge whether the model is learning something useful.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from lumina_st.config.lumina_config import LuminaSTConfig  # noqa: E402
from lumina_st.latents.tiny_vae import TinyVAE  # noqa: E402
from lumina_st.latents.scvi_vae import SCVILatentEncoder  # noqa: E402
from lumina_st.models.lumina_transformer import LuminaTransformer  # noqa: E402
from lumina_st.modules.lumina_flow_module import LuminaFlowModule  # noqa: E402
from lumina_st.core.lumina_imputer import LuminaImputer  # noqa: E402
from lumina_st.data.datasets import ReferenceAtlasDataset, align_to_shared_panel  # noqa: E402
from lumina_st.data.cancer_registry import CancerRegistry  # noqa: E402
from lumina_st import results_contract  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_flow.generate_synthetic_st import generate_synthetic_reference_and_st  # noqa: E402

try:
    from scvi.model import SCVI
except ImportError:
    SCVI = None


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        probe = torch.zeros(1, device="cuda")
        _ = torch.relu(probe)
        return torch.device("cuda")
    except Exception as exc:
        print(f"[WARNING] CUDA is available but failed a kernel probe: {exc}")
        print("          Falling back to CPU for this run.")
        return torch.device("cpu")


def _pearson_per_column(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Vectorized per-column Pearson r between two (n_cells, n_cols) matrices.

    Columns with zero variance in either array yield NaN (excluded downstream
    via nanmean/nanmedian).
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    pc = pred - pred.mean(axis=0, keepdims=True)
    oc = obs - obs.mean(axis=0, keepdims=True)
    num = (pc * oc).sum(axis=0)
    den = np.sqrt((pc**2).sum(axis=0) * (oc**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[~np.isfinite(r)] = np.nan
    return r


def compute_intrinsic_metrics(
    enhanced,
    observed: np.ndarray,
    held_mask: np.ndarray,
    seed: int,
    silhouette_max: int = 10000,
) -> dict:
    """Compute intrinsic, self-supervised enhancement metrics (NO ground truth).

    Metrics:
      * held_out_gene_pearson_mean / _median: recovered vs observed on the
        held-out (masked) gene columns (self-supervised recovery).
      * self_consistency_pearson: recovered vs observed on the NON-masked gene
        columns (enhancement should preserve observed signal).
      * latent_silhouette / _n_subsample: silhouette of obsm['latent_enhanced']
        vs Leiden clusters computed on the latent itself, on a seeded
        <= silhouette_max subsample (silhouette_score is O(n^2)). null if the
        latent obsm key is absent.
      * imputed_nonzero_fraction / imputed_dynamic_range: structure sanity stats.
    """
    metrics: dict = {}

    # Recovered full-gene matrix (only meaningful when imputed has gene dim).
    recovered = None
    if "imputed" in enhanced.layers:
        recovered = np.asarray(enhanced.layers["imputed"], dtype=np.float32)

    if recovered is not None and recovered.shape == observed.shape:
        held_r = _pearson_per_column(recovered[:, held_mask], observed[:, held_mask])
        metrics["held_out_gene_pearson_mean"] = float(np.nanmean(held_r))
        metrics["held_out_gene_pearson_median"] = float(np.nanmedian(held_r))

        keep_mask = ~held_mask
        if keep_mask.any():
            sc_r = _pearson_per_column(recovered[:, keep_mask], observed[:, keep_mask])
            metrics["self_consistency_pearson"] = float(np.nanmean(sc_r))
        else:
            metrics["self_consistency_pearson"] = None

        nonzero = float(np.count_nonzero(recovered)) / float(recovered.size)
        metrics["imputed_nonzero_fraction"] = nonzero
        metrics["imputed_dynamic_range"] = float(
            np.nanmax(recovered) - np.nanmin(recovered)
        )
    else:
        # Latent-only imputation: gene-level recovery undefined.
        metrics["held_out_gene_pearson_mean"] = None
        metrics["held_out_gene_pearson_median"] = None
        metrics["self_consistency_pearson"] = None
        metrics["imputed_nonzero_fraction"] = None
        metrics["imputed_dynamic_range"] = None

    # Latent silhouette (intrinsic separability), seeded <=10k subsample.
    sil_val = None
    sil_n = None
    if "latent_enhanced" in enhanced.obsm:
        try:
            import scanpy as _sc
            from sklearn.metrics import silhouette_score

            latent = np.asarray(enhanced.obsm["latent_enhanced"], dtype=np.float32)
            n = latent.shape[0]
            rng = np.random.default_rng(seed)
            if n > silhouette_max:
                sub_idx = np.sort(rng.choice(n, size=silhouette_max, replace=False))
            else:
                sub_idx = np.arange(n)
            sil_n = int(sub_idx.shape[0])
            sub = latent[sub_idx]

            import anndata as _ad

            tmp = _ad.AnnData(X=np.zeros((sub.shape[0], 1), dtype=np.float32))
            tmp.obsm["latent_enhanced"] = sub
            _sc.pp.neighbors(tmp, use_rep="latent_enhanced", n_neighbors=15)
            _sc.tl.leiden(tmp, key_added="leiden_latent", resolution=1.0)
            labels = tmp.obs["leiden_latent"].to_numpy()
            if len(set(labels)) > 1:
                sil_val = float(silhouette_score(sub, labels))
            else:
                sil_val = None  # single cluster -> silhouette undefined
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] latent_silhouette computation failed: {exc}")
            sil_val = None
    metrics["latent_silhouette"] = sil_val
    metrics["latent_silhouette_n_subsample"] = sil_n

    return metrics


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    # === Smart data source selection ===
    # Priority: 1. Explicit --reference / --target
    #           2. Real baseline data under LUMINA_BASELINE_ROOT (if present)
    #           3. High-quality synthetic data (always works)
    # The baseline root is configurable (#121) so the script is not tied to a
    # specific on-disk layout; default is data/baselines/reference under the repo.
    _default_root = Path(__file__).resolve().parents[2] / "data" / "baselines" / "reference"
    DATA_ROOT = Path(os.environ.get("LUMINA_BASELINE_ROOT", str(_default_root)))

    # Provenance tracking for the results contract.
    used_real_data = False
    dataset_paths: list[str] = []

    if args.reference and args.target:
        ref = sc.read_h5ad(args.reference)
        target = sc.read_h5ad(args.target)
        used_real_data = True
        dataset_paths = [str(Path(args.reference).resolve()), str(Path(args.target).resolve())]
        print("Using user-provided real data files.")
    elif DATA_ROOT.exists() and any((DATA_ROOT / sub).exists() for sub in ("processed", "processed_data")):
        # Accept both folder names — the gdown download yields processed_data/,
        # while the upstream README documents processed/.
        processed = DATA_ROOT / "processed" if (DATA_ROOT / "processed").exists() else DATA_ROOT / "processed_data"
        print(f"\n[INFO] Found real baseline data at: {DATA_ROOT}")
        print(f"       Using {processed.name}/ for E2E run.\n")
        ref_path = processed / "sc_train.h5ad"
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference atlas {ref_path} missing — re-run `python -m lumina_st.cli.download`."
            )
        ref = sc.read_h5ad(ref_path)
        # Pick the first available cancer test file as target
        possible_targets = list(processed.glob("st_*_test.h5ad"))
        if not possible_targets:
            raise FileNotFoundError(f"No st_*_test.h5ad found in {processed}")
        target_path = possible_targets[0]
        target = sc.read_h5ad(target_path)
        used_real_data = True
        dataset_paths = [str(ref_path.resolve()), str(target_path.resolve())]
        if args.cancer is None:
            args.cancer = target_path.stem.replace("st_", "").replace("_test", "")
    else:
        print("\n[INFO] No real data provided and no baseline data found locally.")
        print("       Falling back to high-fidelity synthetic data (mimics the reference baseline usage).\n")

        ref, target, cancer_names = generate_synthetic_reference_and_st(
            n_ref_cells=2500, n_st_cells=500, n_genes=180, n_cancer_types=4, seed=42
        )
        if args.cancer is None:
            args.cancer = cancer_names[0]

    print(f"Reference: {ref.n_obs} cells, {ref.n_vars} genes")
    print(f"Target ST : {target.n_obs} cells, {target.n_vars} genes")

    # Optional seeded target subsample guard (large 344k-cell target).
    target_subsample_n = None
    target_n_obs_full = int(target.n_obs)
    if args.max_cells is not None and target.n_obs > args.max_cells:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(target.n_obs, size=args.max_cells, replace=False))
        target = target[keep].copy()
        target_subsample_n = int(args.max_cells)
        print(
            f"[INFO] Subsampled target from {target_n_obs_full} to "
            f"{target.n_obs} cells (seed={args.seed}) via --max_cells guard."
        )

    compute_start = time.time()

    # The ReferenceAtlasDataset indexes ``cfg.vae_batch_key`` to produce the
    # conditioning label ``y`` that is ALSO passed as the scVI ``batch_index``.
    # scVI here is set up with ``batch_key="cancer_type"`` (n_batch = #cancer_type
    # values), so the dataset MUST index the same column or the scVI batch
    # embedding is indexed out of range. On the synthetic path cancer_type was
    # the conditioning column; on real data the default ``vae_batch_key="batch"``
    # holds sample IDs distinct from cancer_type and breaks this alignment.
    # Pin the dataset key to the scVI batch key so training labels match the
    # encoder's batch space.
    batch_key = "cancer_type" if "cancer_type" in ref.obs else "batch"

    # Build the registry from the SAME column the dataset/scVI condition on,
    # plus a UNKNOWN fallback.  The scVI model is set up with batch_key =
    # batch_key, so every reference label in that column must be in the
    # registry (= in cfg.cancer_types) or the dataset's y-lookup fails.
    #
    # LuminaImputer.enhance rebuilds its OWN internal registry from
    # cfg.cancer_types, so the enhance_label must also be in that list.
    # We therefore include BOTH the reference batch labels AND the requested
    # --cancer label in cancer_types so both the training DataLoader and the
    # enhance() call resolve their labels.
    ref_labels = sorted(ref.obs[batch_key].astype(str).unique().tolist())
    # scVI batch index for the enhancement pass must stay within [0, n_batch).
    # n_batch = number of distinct cancer_type values in the reference.
    # Use the first reference label (index 0) as the enhancement label so the
    # scVI batch embedding is never indexed out of range.
    enhance_label = ref_labels[0] if ref_labels else (args.cancer or "UNKNOWN")

    cancer_types_list = list(dict.fromkeys(
        ref_labels + ([args.cancer] if args.cancer else []) + ["UNKNOWN"]
    ))
    registry = CancerRegistry({c: i for i, c in enumerate(cancer_types_list)})

    cfg = LuminaSTConfig(
        latent_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        guidance_scale=args.guidance_scale,
        cancer_types=cancer_types_list,
        vae_batch_key=batch_key,  # dataset/scVI condition on the same column; #106
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
    )

    # Align reference + target onto their shared gene panel: the VAE is trained
    # on the reference's gene space (n_input = ref.n_vars), so the target must be
    # encoded in that exact space. ``align_to_shared_panel`` is robust to either
    # direction of containment — the legacy ref⊆target case (a ~9.9k atlas vs a
    # 10k target) AND the inverted target⊆ref case (a whole-transcriptome ~33k
    # scRNA atlas vs a ~5k Xenium panel), where the reference is subset *down* to
    # the panel rather than hard-failing. Genes end up in reference order.
    n_genes_dropped = 0
    n_ref_genes_dropped = 0
    if list(target.var_names) != list(ref.var_names):
        ref, target, align_stats = align_to_shared_panel(ref, target)
        n_genes_dropped = align_stats["n_target_dropped"]
        n_ref_genes_dropped = align_stats["n_reference_dropped"]
        print(
            f"[INFO] Aligned to shared gene panel: kept {align_stats['n_shared']} "
            f"genes (dropped {n_ref_genes_dropped} reference-only, "
            f"{n_genes_dropped} target-only)."
        )

    # 1. Train scVI VAE on reference (if not provided)
    if args.scvi_model and Path(args.scvi_model).exists():
        if SCVI is None:
            raise ImportError("scvi-tools is required to load an existing SCVI model")
        print("Loading existing SCVI model...")
        scvi_model = SCVI.load(args.scvi_model)
    else:
        if SCVI is not None:
            print("Training quick SCVI VAE on reference atlas...")
            SCVI.setup_anndata(ref, batch_key="cancer_type" if "cancer_type" in ref.obs else None)
            scvi_model = SCVI(ref, n_latent=cfg.latent_dim)
            scvi_model.train(max_epochs=min(30, args.max_epochs))
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            scvi_save_dir = Path(args.output).parent / "scvi_ref"
            # scvi save() refuses to overwrite by default; allow re-runs to succeed.
            scvi_model.save(str(scvi_save_dir), overwrite=True)
        else:
            print("[WARNING] scvi-tools not found in current env. Using TinyVAE for fast synthetic demo.")
            print("          For real results, run with: conda run -n dl python ...")
            scvi_model = None  # signal to use TinyVAE path

    # 2. Prepare datasets. ReferenceAtlasDataset.__getitem__ indexes a single
    #    row of ref.X and calls torch.from_numpy; a sparse CSR row yields an
    #    object-dtype array that torch can't convert. Densify ref.X to float32
    #    first (20000x9906 ~ 0.75 GB) so per-row indexing returns real floats.
    if hasattr(ref.X, "toarray"):
        ref.X = ref.X.toarray().astype(np.float32)
    else:
        ref.X = np.asarray(ref.X, dtype=np.float32)
    dataset = ReferenceAtlasDataset(ref, cfg, registry)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    # 3. Lumina model
    transformer = LuminaTransformer(
        latent_dim=cfg.latent_dim,
        patch_size=1,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=4.0,
        num_classes=len(registry),
        class_dropout_prob=0.1,
    )

    if scvi_model is None:
        vae_wrapper = TinyVAE(input_dim=ref.n_vars, latent_dim=cfg.latent_dim).to(device)
        vae_opt = torch.optim.AdamW(vae_wrapper.parameters(), lr=cfg.lr)
        x_ref = torch.as_tensor(np.asarray(ref.X), dtype=torch.float32, device=device)
        print("Training TinyVAE fallback encoder...")
        for epoch in range(min(5, cfg.max_epochs)):
            vae_loss = vae_wrapper(x_ref)["loss"]
            vae_opt.zero_grad()
            vae_loss.backward()
            vae_opt.step()
            print(f"  TinyVAE epoch {epoch+1} - loss {vae_loss.item():.4f}")
    else:
        vae_wrapper = SCVILatentEncoder(scvi_model)

    module = LuminaFlowModule(cfg, transformer, vae=vae_wrapper).to(device)

    # Quick training of the flow model
    opt = torch.optim.AdamW(module.parameters(), lr=cfg.lr)
    print("Training Lumina flow model on latent space...")
    for epoch in range(cfg.max_epochs):
        last_loss = None
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            z, _ = vae_wrapper.encode_to_latent(x, y)   # may need adaptation for real SCVI
            # For real SCVI we should use the model's get_latent_representation on AnnData batches
            # This is a simplified path — for production use the high-level API on AnnData
            loss_dict = module.transport.training_losses(transformer, z.detach(), {"y": y})
            loss = loss_dict["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss
        if epoch % 5 == 0 and last_loss is not None:
            print(f"  Epoch {epoch+1}/{cfg.max_epochs} - loss {last_loss.item():.4f}")

    # 4. Held-out-gene enhancement (self-supervised). Mask a seeded random
    #    subset of genes on the target, enhance, then correlate the recovered
    #    values against the original observed values on those held-out columns.
    print("Enhancing target ST slice (held-out-gene benchmark)...")
    imputer = LuminaImputer(cfg, module)

    obs_mat = target.X.toarray() if hasattr(target.X, "toarray") else np.asarray(target.X)
    obs_mat = np.asarray(obs_mat, dtype=np.float32)
    n_genes = target.n_vars
    var_names = list(target.var_names)

    rng = np.random.default_rng(args.seed)
    n_held = max(1, int(round(args.held_out_frac * n_genes)))
    held_idx = np.sort(rng.choice(n_genes, size=n_held, replace=False))
    held_out_genes = [var_names[i] for i in held_idx]
    held_mask = np.zeros(n_genes, dtype=bool)
    held_mask[held_idx] = True
    print(f"[INFO] Holding out {n_held}/{n_genes} genes (seed={args.seed}) for recovery.")

    enhanced = imputer.enhance(
        target, cancer_type=enhance_label, held_out_genes=held_out_genes, seed=args.seed
    )

    # 5. Intrinsic metrics (NO ground-truth / ARI). All self-supervised.
    metrics = compute_intrinsic_metrics(
        enhanced=enhanced,
        observed=obs_mat,
        held_mask=held_mask,
        seed=args.seed,
        silhouette_max=args.silhouette_max,
    )
    print("\n=== Intrinsic Enhancement Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    runtime_s = time.time() - compute_start

    # 6. Emit via the vendored uniform results contract.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced.write(out_path)
    print(f"\nEnhanced AnnData saved to {out_path}")

    notes_parts: list[str] = []
    if not used_real_data:
        notes_parts.append("SYNTHETIC fallback data (no real reference/target provided)")
    if target_subsample_n is not None:
        notes_parts.append(
            f"target subsampled to {target_subsample_n}/{target_n_obs_full} cells "
            f"(seed={args.seed}) via --max_cells"
        )
    if metrics.get("latent_silhouette") is None:
        notes_parts.append("latent_silhouette=null (obsm['latent_enhanced'] absent or single cluster)")
    notes_parts.append(f"held_out genes={n_held}/{n_genes} (frac={args.held_out_frac})")
    notes_parts.append(
        f"requested_cancer={args.cancer!r}; conditioned scVI batch on reference "
        f"{batch_key}={enhance_label!r} (target --cancer label outside scVI batch space)"
    )
    if n_genes_dropped or n_ref_genes_dropped:
        notes_parts.append(
            f"aligned to shared gene panel: dropped {n_ref_genes_dropped} "
            f"reference-only + {n_genes_dropped} target-only genes"
        )

    card_id = results_contract.dataset_card_id(dataset_paths) if dataset_paths else "synthetic"

    results_root = Path(__file__).resolve().parents[2] / "results"
    write_out = results_contract.write_results(
        project="lumina-st",
        dataset_card_id=card_id,
        metrics=metrics,
        outputs={"enhanced": "outputs/enhanced.h5ad", "imputed": "outputs/imputed.npy"},
        run_metadata={
            "dataset_paths": dataset_paths,
            "n_obs": int(enhanced.n_obs),
            "n_vars": int(enhanced.n_vars),
            "seed": int(args.seed),
            "runtime_s": float(runtime_s),
            "device": "cuda" if device.type == "cuda" else "cpu",
            "deterministic": False,
            "num_threads": int(torch.get_num_threads()),
            "reproducibility_level": "seeded",
            "normalization": {"applied": False, "method": "none"},
            "interpretability": {
                "model_is_learned": True,
                "encoder": "SCVI VAE latent encoder + LuminaTransformer flow-matching",
                "domain_assignment": "n/a (enhancement/imputation, no domain labels)",
                "caveats": [],
            },
            "notes": "; ".join(notes_parts),
        },
        results_dir=str(results_root),
    )

    # Write native output artifacts into outputs/.
    outputs_dir = Path(write_out["outputs_dir"])
    enhanced.write(outputs_dir / "enhanced.h5ad")
    imputed_layer = "imputed" if "imputed" in enhanced.layers else (
        "imputed_latent" if "imputed_latent" in enhanced.layers else None
    )
    if imputed_layer is not None:
        np.save(outputs_dir / "imputed.npy", np.asarray(enhanced.layers[imputed_layer]))

    print("\n=== Contract emission ===")
    print(f"  metrics.json      -> {write_out['metrics']}")
    print(f"  run_metadata.json -> {write_out['run_metadata']}")
    print(f"  outputs/          -> {outputs_dir}")
    print(f"  used_real_data    -> {used_real_data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=None, help="Path to reference scRNA atlas h5ad (optional - will use synthetic)")
    parser.add_argument("--target", default=None, help="Path to target ST h5ad to enhance (optional - will use synthetic)")
    parser.add_argument("--cancer", default=None)
    parser.add_argument("--output", default="./lumina_enhanced.h5ad")
    parser.add_argument("--scvi_model", default=None)
    parser.add_argument("--latent_dim", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0, help="Seed for held-out gene/cell subsampling and silhouette.")
    parser.add_argument("--max_cells", type=int, default=None, help="Optional seeded target-cell subsample cap (large targets).")
    parser.add_argument("--held_out_frac", type=float, default=0.1, help="Fraction of target genes held out for self-supervised recovery.")
    parser.add_argument("--silhouette_max", type=int, default=10000, help="Max cells (seeded subsample) for latent silhouette (O(n^2)).")
    parser.add_argument("--sampling_method", type=str, default="euler", help="ODE solver for enhancement (euler/heun/dopri5). Default euler avoids GPU OOM on large batches.")
    parser.add_argument("--num_sampling_steps", type=int, default=10, help="Number of ODE integration steps (fixed-step solvers).")
    args = parser.parse_args()
    main(args)
