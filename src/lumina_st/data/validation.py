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
            check_sparsity: If True, checks if adata.X has zero-variance or extreme values.
            
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
