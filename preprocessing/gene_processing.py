"""
Omni-ST: Gene Expression Preprocessing
========================================
Scanpy-orchestrated preprocessing pipeline for ST gene expression data.

Steps:
  1. Quality control filtering (cells and genes)
  2. Normalization (library-size / scran)
  3. Log1p transformation
  4. Highly variable gene (HVG) selection
  5. PCA for initial embedding
  6. Optional batch correction (Harmony)
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import anndata as ad
import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore")


def qc_filter(
    adata: ad.AnnData,
    min_genes: int = 200,
    max_genes: int = 8000,
    min_cells: int = 3,
    max_pct_mito: float = 20.0,
    mito_prefix: str = "MT-",
    verbose: bool = True,
) -> ad.AnnData:
    """
    Quality control filtering.

    Parameters
    ----------
    adata : AnnData
    min_genes : int  minimum genes per cell/spot
    max_genes : int  maximum genes per cell/spot (removes doublets)
    min_cells : int  minimum cells/spots per gene
    max_pct_mito : float  maximum mitochondrial gene percentage
    mito_prefix : str  mitochondrial gene prefix (``"MT-"`` for human)

    Returns
    -------
    Filtered AnnData
    """
    # Mitochondrial genes
    adata.var["mito"] = adata.var_names.str.startswith(mito_prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], percent_top=None, log1p=False, inplace=True)

    n_before = adata.n_obs
    adata = adata[
        (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["n_genes_by_counts"] <= max_genes)
        & (adata.obs["pct_counts_mito"] <= max_pct_mito)
    ].copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if verbose:
        print(f"QC: retained {adata.n_obs}/{n_before} spots, {adata.n_vars} genes")
    return adata


def normalize_expression(
    adata: ad.AnnData,
    target_sum: float = 1e4,
    log1p: bool = True,
    copy: bool = False,
) -> ad.AnnData:
    """
    Library-size normalization and optional log1p transform.

    Parameters
    ----------
    adata : AnnData  (raw counts expected in .X)
    target_sum : float  normalize each spot to this total count
    log1p : bool  apply log(x+1) after normalization
    """
    if copy:
        adata = adata.copy()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(adata)
    return adata


def select_hvgs(
    adata: ad.AnnData,
    n_top_genes: int = 3000,
    flavor: str = "seurat_v3",
    batch_key: Optional[str] = None,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Highly variable gene (HVG) selection.

    Parameters
    ----------
    adata : AnnData  (must be log-normalized for Seurat, raw counts for v3)
    n_top_genes : int
    flavor : str  ``"seurat"``, ``"seurat_v3"``, or ``"cell_ranger"``
    batch_key : str | None  correct for batch when selecting HVGs
    """
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        batch_key=batch_key,
        inplace=inplace,
        subset=False,
    )
    print(f"HVG selection: {adata.var['highly_variable'].sum()} HVGs selected from {adata.n_vars}")
    return adata


def compute_pca(
    adata: ad.AnnData,
    n_comps: int = 50,
    hvg_only: bool = True,
) -> ad.AnnData:
    """
    PCA on HVG-subset of log-normalized data.
    Result stored in ``adata.obsm["X_pca"]``.
    """
    if hvg_only and "highly_variable" in adata.var.columns:
        sc.pp.pca(adata, n_comps=n_comps, use_highly_variable=True)
    else:
        sc.pp.scale(adata, max_value=10)
        sc.pp.pca(adata, n_comps=n_comps)
    print(f"PCA: computed {n_comps} components")
    return adata


def preprocess_pipeline(
    adata: ad.AnnData,
    min_genes: int = 200,
    max_genes: int = 8000,
    min_cells: int = 3,
    max_pct_mito: float = 20.0,
    target_sum: float = 1e4,
    n_hvgs: int = 3000,
    n_pca: int = 50,
    batch_key: Optional[str] = None,
    harmony_batch_key: Optional[str] = None,
    verbose: bool = True,
) -> ad.AnnData:
    """
    End-to-end preprocessing for a single AnnData object.

    Returns the preprocessed AnnData with:
      - adata.X: log-normalized expression
      - adata.layers["counts"]: raw counts
      - adata.var["highly_variable"]: HVG flags
      - adata.obsm["X_pca"]: PCA coordinates
      - adata.obsm["X_pca_harmony"]: Harmony-corrected PCA (if requested)
    """
    if verbose:
        print(f"Input: {adata.n_obs} spots × {adata.n_vars} genes")

    adata = qc_filter(adata, min_genes, max_genes, min_cells, max_pct_mito, verbose=verbose)
    adata = normalize_expression(adata, target_sum=target_sum)
    adata = select_hvgs(adata, n_top_genes=n_hvgs, batch_key=batch_key)
    adata = compute_pca(adata, n_comps=n_pca)

    if harmony_batch_key and harmony_batch_key in adata.obs.columns:
        try:
            import harmonypy as hm
            ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, harmony_batch_key)
            adata.obsm["X_pca_harmony"] = ho.Z_corr.T
            if verbose:
                print(f"Harmony correction applied on '{harmony_batch_key}'")
        except ImportError:
            warnings.warn("harmonypy not installed. Skipping batch correction.")

    return adata
