"""
Omni-ST: Visualization Suite
==============================
Multi-modal spatial transcriptomics visualization utilities.

Includes:
  - Spatial gene expression heatmaps
  - Spatial domain segmentation maps
  - Attention heatmaps over histology images
  - UMAP / t-SNE embedding projections
  - Multimodal alignment plots
  - Interactive Streamlit dashboard launcher
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Spatial Gene Expression Heatmap
# ---------------------------------------------------------------------------

def plot_spatial_gene_expression(
    coords: np.ndarray,
    expression: np.ndarray,
    gene_name: str = "Gene",
    spot_size: float = 10.0,
    cmap: str = "magma",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Render a spatial heatmap of a single gene's expression.

    Parameters
    ----------
    coords : np.ndarray [N, 2]  spatial (x, y) coordinates
    expression : np.ndarray [N]  expression values
    gene_name : str
    spot_size : float
    cmap : str  matplotlib colormap
    ax : plt.Axes | None
    title : str | None

    Returns
    -------
    matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0f0f1a")
    else:
        fig = ax.get_figure()

    ax.set_facecolor("#0f0f1a")
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=expression,
        s=spot_size,
        cmap=cmap,
        alpha=0.9,
        linewidths=0,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Log-Normalized Expression", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_title(title or f"{gene_name} Expression", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Spatial X", color="#aaa", fontsize=9)
    ax.set_ylabel("Spatial Y", color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spatial Domain Segmentation Map
# ---------------------------------------------------------------------------

def plot_domain_map(
    coords: np.ndarray,
    domain_labels: np.ndarray,
    domain_names: Optional[List[str]] = None,
    spot_size: float = 12.0,
    cmap: str = "tab20",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Render a color-coded spatial domain segmentation map.

    Parameters
    ----------
    coords : np.ndarray [N, 2]
    domain_labels : np.ndarray [N]  integer domain assignments
    domain_names : list[str] | None  human-readable domain names
    """
    unique_domains = np.unique(domain_labels)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(unique_domains)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0f0f1a")
    else:
        fig = ax.get_figure()

    ax.set_facecolor("#0f0f1a")

    for i, d in enumerate(unique_domains):
        mask = domain_labels == d
        label = domain_names[i] if domain_names and i < len(domain_names) else f"Domain {d}"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=spot_size, c=[colors[i]], label=label, alpha=0.85, linewidths=0,
        )

    ax.legend(
        loc="upper right", fontsize=7,
        facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", markerscale=1.5,
    )
    ax.set_title("Spatial Domain Segmentation", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Spatial X", color="#aaa", fontsize=9)
    ax.set_ylabel("Spatial Y", color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Attention Map Overlay on Histology
# ---------------------------------------------------------------------------

def plot_attention_overlay(
    image: np.ndarray,
    attention_weights: np.ndarray,
    patch_grid: Tuple[int, int],
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.5,
    cmap: str = "hot",
) -> plt.Figure:
    """
    Overlay patch-level attention weights on a histology image.

    Parameters
    ----------
    image : np.ndarray [H, W, 3]  uint8 RGB histology image
    attention_weights : np.ndarray [N_patches]  attention values
    patch_grid : (rows, cols)  patch grid dimensions
    alpha : float  attention overlay transparency
    """
    H, W = image.shape[:2]
    rows, cols = patch_grid
    ph, pw = H // rows, W // cols

    attn_map = attention_weights.reshape(rows, cols)
    attn_scaled = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-9)
    from PIL import Image as PILImage
    attn_img = np.array(
        PILImage.fromarray((plt.get_cmap(cmap)(attn_scaled)[:, :, :3] * 255).astype(np.uint8))
        .resize((W, H))
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="black")
    else:
        fig = ax.get_figure()

    ax.imshow(image)
    ax.imshow(attn_img, alpha=alpha)
    ax.set_title("Attention Map Overlay", color="white", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UMAP / t-SNE Embedding Visualization
# ---------------------------------------------------------------------------

def plot_embedding(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "umap",
    label_names: Optional[List[str]] = None,
    title: str = "Embedding Space",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Project N-D embeddings to 2-D and scatter-plot colored by labels.

    Parameters
    ----------
    embeddings : np.ndarray [N, D]
    labels : np.ndarray [N]  integer cluster/class labels
    method : str  ``"umap"`` or ``"tsne"``
    label_names : list[str] | None
    """
    if method == "umap":
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, min_dist=0.25, n_neighbors=30)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)

    emb_2d = reducer.fit_transform(embeddings)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0f0f1a")
    else:
        fig = ax.get_figure()

    ax.set_facecolor("#0f0f1a")
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, len(np.unique(labels))))

    for i, lbl in enumerate(np.unique(labels)):
        mask = labels == lbl
        name = label_names[lbl] if label_names and lbl < len(label_names) else str(lbl)
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=8, c=[palette[i]], label=name, alpha=0.8)

    ax.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", markerscale=2)
    ax.set_title(f"{title} — {method.upper()}", color="white", fontsize=13)
    ax.set_xlabel(f"{method.upper()} 1", color="#aaa", fontsize=9)
    ax.set_ylabel(f"{method.upper()} 2", color="#aaa", fontsize=9)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-gene Heatmap Panel
# ---------------------------------------------------------------------------

def plot_gene_panel(
    coords: np.ndarray,
    expression_matrix: np.ndarray,
    gene_names: List[str],
    ncols: int = 3,
    cmap: str = "magma",
) -> plt.Figure:
    """Plot multiple genes in a panel grid."""
    n_genes = len(gene_names)
    nrows = (n_genes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), facecolor="#0a0a14")

    axes_flat = np.array(axes).flatten()
    for i, gene in enumerate(gene_names):
        plot_spatial_gene_expression(
            coords, expression_matrix[:, i], gene_name=gene,
            ax=axes_flat[i], cmap=cmap,
        )
    for ax in axes_flat[n_genes:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig
