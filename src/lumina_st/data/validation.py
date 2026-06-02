import logging
from typing import Iterable, List, Optional, Set, Union
import anndata as ad
import numpy as np

logger = logging.getLogger("lumina_st.data.validation")

# Cancer-type tokens that are deliberately non-specific and therefore always
# compatible with any target (pan-cancer / multi-tissue / unlabeled reference
# pools). See CancerRegistry.default_pan_cancer / the dataset registry.
_WILDCARD_CANCER_TYPES = frozenset({"UNKNOWN", "MIXED", "PAN", "PANCANCER", "PAN_CANCER"})

class AnnDataSchemaValidator:
    """Validator for verifying target ST and reference scRNA AnnData schemas."""

    @staticmethod
    def validate_spatial_data(
        adata: ad.AnnData,
        required_obs: Optional[List[str]] = None,
        required_obsm: Optional[List[str]] = None,
        check_sparsity: bool = True
    ) -> bool:
        """
        Validates target Spatial Transcriptomics (ST) AnnData object.
        
        Args:
            adata: AnnData object to validate.
            required_obs: List of keys required in adata.obs (e.g. ['cancer_type']).
            required_obsm: List of keys required in adata.obsm (e.g. ['spatial']).
            check_sparsity: If True, rejects ``adata.X`` matrices that
                are ``None``, contain non-finite values (NaN/Inf), or have
                zero per-gene variance across every gene (e.g. all-zero
                or all-constant matrices).
            
        Returns:
            True if valid, raises ValueError or returns False otherwise.
        """
        logger.info("Validating target Spatial Transcriptomics (ST) dataset schema...")
        
        # 1. Basic check
        if not isinstance(adata, ad.AnnData):
            raise TypeError("Input must be an AnnData object.")
            
        # 2. Coordinates check
        required_obsm = required_obsm or ["spatial"]
        for obsm_key in required_obsm:
            if obsm_key not in adata.obsm:
                raise ValueError(f"Missing required spatial coordinates key '.obsm[\"{obsm_key}\"]' in AnnData.")
            coords = adata.obsm[obsm_key]
            if not isinstance(coords, np.ndarray):
                raise TypeError(f".obsm['{obsm_key}'] must be a numpy ndarray.")
            if coords.ndim != 2 or coords.shape[1] < 2:
                raise ValueError(f".obsm['{obsm_key}'] must be 2D with at least 2 columns (x, y). Got shape {coords.shape}.")
                
        # 3. Observations check
        required_obs = required_obs or []
        for obs_key in required_obs:
            if obs_key not in adata.obs:
                raise ValueError(f"Missing required metadata column '.obs[\"{obs_key}\"]' in AnnData.")
                
        # 4. Expression check
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError(f"AnnData contains no data. Shape: {adata.shape}")
            
        if check_sparsity:
            # Check if empty
            if adata.X is None:
                raise ValueError("AnnData.X expression matrix is None.")

            # Densify if sparse for the cheap per-matrix checks; ST
            # matrices in `validate_spatial_data` are user-sized so this
            # is acceptable (and densification is bounded by the .X the
            # caller already loaded).
            X = adata.X
            X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

            # Non-finite values (NaN / +-Inf) — the docstring promises
            # "extreme values"; previously these passed silently and went
            # straight into the encoder.
            if not np.all(np.isfinite(X_dense)):
                raise ValueError(
                    "AnnData.X contains non-finite values (NaN or Inf); "
                    "this is rejected when check_sparsity=True."
                )

            # Zero-variance check — the docstring promises "zero-variance"
            # rejection. Flag when every gene is constant across cells
            # (e.g. all-zero or all-same matrices). We check per-gene
            # rather than global variance so a single non-constant gene
            # rescues an otherwise-degenerate matrix.
            if X_dense.shape[0] > 1:
                per_gene_var = np.var(X_dense, axis=0)
                if not np.any(per_gene_var > 0):
                    raise ValueError(
                        "AnnData.X has zero per-gene variance across all genes "
                        "(every column is constant); this is rejected when "
                        "check_sparsity=True."
                    )

        logger.info("Spatial Transcriptomics dataset schema validation passed.")
        return True

    @staticmethod
    def validate_reference_data(
        adata: ad.AnnData,
        required_obs: Optional[List[str]] = None,
    ) -> bool:
        """
        Validates reference scRNA-seq AnnData object.
        
        Args:
            adata: AnnData object to validate.
            required_obs: List of keys required in adata.obs (e.g. ['cell_type', 'cancer_type']).
            
        Returns:
            True if valid, raises ValueError or returns False otherwise.
        """
        logger.info("Validating reference scRNA-seq dataset schema...")
        
        # 1. Basic check
        if not isinstance(adata, ad.AnnData):
            raise TypeError("Input must be an AnnData object.")
            
        # 2. Observations check
        required_obs = required_obs or ["cell_type"]
        for obs_key in required_obs:
            if obs_key not in adata.obs:
                raise ValueError(f"Missing required metadata column '.obs[\"{obs_key}\"]' in reference AnnData.")
                
        # 3. Expression check
        if adata.X is None:
            raise ValueError("Reference AnnData.X expression matrix is None.")
            
        logger.info("Reference scRNA-seq dataset schema validation passed.")
        return True

    @staticmethod
    def align_genes(
        target_adata: ad.AnnData,
        ref_adata: ad.AnnData,
        min_overlap_ratio: float = 0.5
    ) -> List[str]:
        """
        Checks gene overlap between target ST and reference scRNA-seq.
        
        Args:
            target_adata: Target ST AnnData.
            ref_adata: Reference scRNA AnnData.
            min_overlap_ratio: Minimum ratio of target genes that must exist in reference.
            
        Returns:
            List of overlapping gene names, sorted alphabetically so the
            returned ordering is deterministic across processes (independent
            of ``PYTHONHASHSEED``).
        """
        target_genes = set(target_adata.var_names)
        ref_genes = set(ref_adata.var_names)

        # Deterministic order: sort the intersection so the returned list
        # (and any AnnData subset / column reorder built from it) does NOT
        # depend on Python's randomized string hashing.
        overlap = sorted(target_genes & ref_genes)
        overlap_ratio = len(overlap) / max(1, len(target_genes))
        
        logger.info(f"Gene overlap count: {len(overlap)} / {len(target_genes)} ({overlap_ratio * 100:.2f}%)")
        
        if overlap_ratio < min_overlap_ratio:
            raise ValueError(
                f"Low gene overlap between target and reference: {overlap_ratio * 100:.2f}% "
                f"(required at least {min_overlap_ratio * 100:.2f}%)."
            )

        return overlap

    @staticmethod
    def _resolve_cancer_types(
        source: Union["ad.AnnData", str, Iterable[str]],
        obs_key: str,
        role: str,
    ) -> Set[str]:
        """Normalize a cancer-type source into a set of upper-cased tokens.

        Accepts an AnnData (reads ``.obs[obs_key]``), a single string, or any
        iterable of strings. Raises ``KeyError`` if an AnnData lacks ``obs_key``.
        """
        if isinstance(source, ad.AnnData):
            if obs_key not in source.obs:
                raise KeyError(
                    f"{role} AnnData has no .obs['{obs_key}']; pass the cancer "
                    "type explicitly (str) so suitability can be checked."
                )
            values: Iterable[str] = source.obs[obs_key].astype(str).tolist()
        elif isinstance(source, str):
            values = [source]
        else:
            values = [str(v) for v in source]
        return {v.strip().upper() for v in values if str(v).strip()}

    @staticmethod
    def check_reference_target_suitability(
        reference: Union["ad.AnnData", str, Iterable[str]],
        target: Union["ad.AnnData", str, Iterable[str]],
        *,
        obs_key: str = "cancer_type",
        on_mismatch: str = "warn",
    ) -> bool:
        """Guard against pairing a reference atlas with a mismatched target.

        The headline LuminaST failure mode (issue #212) is enhancing a
        **solid-tumor** ST target (e.g. COAD) with a **mismatched** reference
        atlas (e.g. GSE132509 — Acute Lymphocytic Leukemia PBMC), whose
        ``cancer_type`` was synthesised from a GEO accession. That invalidates
        every gene-recovery / clustering *uplift* claim, yet nothing flagged it.

        Compatibility rule: the reference and target cancer-type token sets are
        suitable if they intersect, or if either side is a deliberate wildcard
        (``UNKNOWN``/``MIXED``/pan-cancer). Disjoint specific tokens are a
        mismatch.

        Args:
            reference: reference cancer type(s) — AnnData, a string, or an
                iterable of strings.
            target: target cancer type(s) — same accepted forms. ST targets
                often carry no ``cancer_type`` obs column, so pass the card's
                declared type as a string.
            obs_key: ``.obs`` column read when an AnnData is supplied.
            on_mismatch: ``"warn"`` (log a warning, return ``False``),
                ``"raise"`` (raise ``ValueError``), or ``"ignore"``
                (return ``False`` silently).

        Returns:
            ``True`` if the pairing is suitable, ``False`` on a mismatch
            (unless ``on_mismatch="raise"``).

        Raises:
            ValueError: on a mismatch when ``on_mismatch="raise"``, or if
                ``on_mismatch`` is not a recognised mode.
        """
        if on_mismatch not in {"warn", "raise", "ignore"}:
            raise ValueError(
                f"on_mismatch must be 'warn', 'raise' or 'ignore'; got {on_mismatch!r}."
            )

        ref_types = AnnDataSchemaValidator._resolve_cancer_types(reference, obs_key, "Reference")
        tgt_types = AnnDataSchemaValidator._resolve_cancer_types(target, obs_key, "Target")

        ref_specific = ref_types - _WILDCARD_CANCER_TYPES
        tgt_specific = tgt_types - _WILDCARD_CANCER_TYPES

        # Suitable if a wildcard is present on either side, or the specific
        # tokens overlap.
        wildcard = (ref_types & _WILDCARD_CANCER_TYPES) or (tgt_types & _WILDCARD_CANCER_TYPES)
        if wildcard or (ref_specific & tgt_specific):
            logger.info(
                "Reference/target cancer-type suitability OK (reference=%s, target=%s).",
                sorted(ref_types), sorted(tgt_types),
            )
            return True

        msg = (
            f"Reference/target cancer-type MISMATCH: reference={sorted(ref_types)} "
            f"vs target={sorted(tgt_types)}. Enhancing a target with a mismatched "
            "reference atlas invalidates gene-recovery and clustering uplift "
            "metrics (issue #212). Swap in a matched reference or document the "
            "caveat explicitly."
        )
        if on_mismatch == "raise":
            raise ValueError(msg)
        if on_mismatch == "warn":
            logger.warning(msg)
        return False
