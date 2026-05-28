import logging
from typing import List, Optional
import anndata as ad
import numpy as np

logger = logging.getLogger("lumina_st.data.validation")

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
            List of overlapping gene names.
        """
        target_genes = set(target_adata.var_names)
        ref_genes = set(ref_adata.var_names)
        
        overlap = list(target_genes.intersection(ref_genes))
        overlap_ratio = len(overlap) / max(1, len(target_genes))
        
        logger.info(f"Gene overlap count: {len(overlap)} / {len(target_genes)} ({overlap_ratio * 100:.2f}%)")
        
        if overlap_ratio < min_overlap_ratio:
            raise ValueError(
                f"Low gene overlap between target and reference: {overlap_ratio * 100:.2f}% "
                f"(required at least {min_overlap_ratio * 100:.2f}%)."
            )
            
        return overlap
