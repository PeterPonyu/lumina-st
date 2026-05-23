import pytest
import numpy as np
import anndata as ad
from lumina_st.data.validation import AnnDataSchemaValidator

def test_validate_spatial_data():
    # Valid dataset
    X = np.random.rand(10, 5)
    obs = {"cancer_type": ["CESC"] * 10}
    obsm = {"spatial": np.random.rand(10, 2)}
    adata = ad.AnnData(X=X, obs=obs, obsm=obsm)
    
    assert AnnDataSchemaValidator.validate_spatial_data(adata, required_obs=["cancer_type"]) == True

    # Missing spatial coordinate
    adata_no_spatial = ad.AnnData(X=X, obs=obs)
    with pytest.raises(ValueError, match="Missing required spatial coordinates"):
        AnnDataSchemaValidator.validate_spatial_data(adata_no_spatial)

    # Missing required observation column
    with pytest.raises(ValueError, match="Missing required metadata column"):
        AnnDataSchemaValidator.validate_spatial_data(adata, required_obs=["nonexistent_obs"])

    # Invalid coordinates type/dimension
    adata_bad_spatial = ad.AnnData(X=X, obs=obs, obsm={"spatial": np.random.rand(10)})
    with pytest.raises(ValueError, match="must be 2D with at least 2 columns"):
        AnnDataSchemaValidator.validate_spatial_data(adata_bad_spatial)

def test_validate_reference_data():
    # Valid
    X = np.random.rand(10, 5)
    obs = {"cell_type": ["T-cell"] * 10}
    adata = ad.AnnData(X=X, obs=obs)
    assert AnnDataSchemaValidator.validate_reference_data(adata) == True

    # Missing cell_type
    adata_no_cell = ad.AnnData(X=X)
    with pytest.raises(ValueError, match="Missing required metadata column"):
        AnnDataSchemaValidator.validate_reference_data(adata_no_cell)

def test_align_genes():
    target_genes = ["A", "B", "C"]
    ref_genes = ["B", "C", "D", "E"]
    
    target_adata = ad.AnnData(X=np.random.rand(2, 3), var={"gene_short_name": target_genes})
    target_adata.var_names = target_genes
    
    ref_adata = ad.AnnData(X=np.random.rand(2, 4), var={"gene_short_name": ref_genes})
    ref_adata.var_names = ref_genes
    
    overlap = AnnDataSchemaValidator.align_genes(target_adata, ref_adata, min_overlap_ratio=0.5)
    assert set(overlap) == {"B", "C"}
    
    # Check trigger low overlap ratio error
    with pytest.raises(ValueError, match="Low gene overlap"):
        AnnDataSchemaValidator.align_genes(target_adata, ref_adata, min_overlap_ratio=0.9)
