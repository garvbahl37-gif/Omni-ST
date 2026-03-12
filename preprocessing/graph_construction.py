"""
Omni-ST: Spatial Graph Construction
======================================
Builds spatial neighborhood graphs from tissue spot coordinates
for use as input to the Graph encoder.

Methods:
  - k-Nearest Neighbors (kNN) graph
  - Radius-based neighborhood graph
  - Delaunay triangulation graph
  - Squidpy-based spatial neighborhood graphs
"""

from __future__ import annotations

from typing import Optional, Tuple

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch


def build_knn_graph(
    coords: np.ndarray,
    k: int = 6,
    include_self: bool = False,
    normalize_coords: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a k-nearest neighbor graph from spatial coordinates.

    Parameters
    ----------
    coords : np.ndarray [N, 2]  (x, y) pixel or array coordinates
    k : int  number of neighbours
    include_self : bool  include self-loops
    normalize_coords : bool  normalize coordinates to [0, 1]

    Returns
    -------
    edge_index : np.ndarray [2, E]  (src, dst) edge list
    edge_weights : np.ndarray [E]   Euclidean distances
    """
    from sklearn.neighbors import NearestNeighbors

    if normalize_coords:
        mn, mx = coords.min(axis=0), coords.max(axis=0)
        rng = np.where(mx - mn > 0, mx - mn, 1.0)
        coords = (coords - mn) / rng

    k_actual = k + 1 if not include_self else k
    nbrs = NearestNeighbors(n_neighbors=k_actual, algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    if not include_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    N = coords.shape[0]
    src = np.repeat(np.arange(N), k)
    dst = indices.flatten()
    weights = distances.flatten()

    edge_index = np.stack([src, dst], axis=0)  # [2, E]
    return edge_index, weights


def build_radius_graph(
    coords: np.ndarray,
    radius: float = 2.0,
    max_neighbors: int = 20,
    normalize_coords: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a radius-based spatial graph.

    Parameters
    ----------
    coords : np.ndarray [N, 2]
    radius : float  distance threshold (in normalized units if normalize_coords)
    max_neighbors : int  cap on neighbours per node
    """
    from sklearn.neighbors import NearestNeighbors

    if normalize_coords:
        mn, mx = coords.min(axis=0), coords.max(axis=0)
        rng = np.where(mx - mn > 0, mx - mn, 1.0)
        coords = (coords - mn) / rng

    nbrs = NearestNeighbors(radius=radius).fit(coords)
    distances, indices = nbrs.radius_neighbors(coords, sort_results=True)

    src_list, dst_list, w_list = [], [], []
    for i in range(len(indices)):
        neighbours = [(d, j) for d, j in zip(distances[i], indices[i]) if j != i]
        neighbours = neighbours[:max_neighbors]
        for d, j in neighbours:
            src_list.append(i)
            dst_list.append(j)
            w_list.append(d)

    edge_index = np.stack([np.array(src_list), np.array(dst_list)], axis=0)
    return edge_index, np.array(w_list)


def anndata_to_graph_tensors(
    adata: ad.AnnData,
    graph_method: str = "knn",
    k: int = 6,
    use_pca: bool = True,
    pca_key: str = "X_pca",
    coord_key: str = "spatial",
) -> dict:
    """
    Convert an AnnData object into a PyTorch-Geometric-compatible graph dict.

    Parameters
    ----------
    adata : AnnData  (preprocessed, with PCA in obsm)
    graph_method : str  ``"knn"``, ``"radius"``
    k : int  neighbours for kNN
    use_pca : bool  use PCA embeddings as node features
    pca_key : str  key in adata.obsm for PCA features
    coord_key : str  key in adata.obsm for spatial coordinates

    Returns
    -------
    dict with keys:
      ``node_feat``, ``coords``, ``edge_index``, ``edge_attr``, ``batch``
    """
    coords = adata.obsm[coord_key].astype(np.float32)

    if use_pca and pca_key in adata.obsm:
        node_feat = adata.obsm[pca_key].astype(np.float32)
    else:
        # Fall back to raw expression
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        node_feat = X.astype(np.float32)

    if graph_method == "knn":
        edge_index, edge_weights = build_knn_graph(coords, k=k)
    elif graph_method == "radius":
        edge_index, edge_weights = build_radius_graph(coords)
    else:
        raise ValueError(f"Unknown graph method: {graph_method}")

    return {
        "node_feat": torch.tensor(node_feat),
        "coords": torch.tensor(coords),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_attr": torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1),
        "batch": torch.zeros(len(adata), dtype=torch.long),
    }


def build_squidpy_graph(
    adata: ad.AnnData,
    n_rings: int = 1,
    coord_type: str = "generic",
    library_id: Optional[str] = None,
) -> ad.AnnData:
    """
    Build spatial neighborhood graph using Squidpy.
    Requires Squidpy to be installed.

    Stores the graph in ``adata.obsp["spatial_connectivities"]``.
    """
    try:
        import squidpy as sq
        sq.gr.spatial_neighbors(
            adata,
            n_rings=n_rings,
            coord_type=coord_type,
            library_id=library_id,
        )
    except ImportError:
        raise ImportError("Install squidpy: pip install squidpy")
    return adata
