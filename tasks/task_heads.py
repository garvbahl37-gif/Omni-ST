"""
Omni-ST: Task Modules
======================
Task-specific heads that sit on top of the shared multimodal backbone.
Each task head takes the backbone's output and produces task predictions.
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageToGeneHead(nn.Module):
    """Predicts gene expression profile from image embedding."""

    def __init__(self, embed_dim: int = 512, num_genes: int = 3000, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_genes),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class GeneToCellTypeHead(nn.Module):
    """Classifies cell type from gene embedding."""

    def __init__(self, embed_dim: int = 512, num_classes: int = 20) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class GraphToDomainHead(nn.Module):
    """Segments tissue into spatial domains from graph embedding."""

    def __init__(self, embed_dim: int = 512, num_domains: int = 7) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class TextToSpatialRetrievalHead(nn.Module):
    """
    Projects text embedding for cosine retrieval over spatial spot embeddings.
    No learnable parameters — alignment is done in shared latent space.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        text_emb: torch.Tensor,
        spot_embs: torch.Tensor,
        top_k: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        text_emb : [B, D]
        spot_embs : [N_spots, D]  pre-encoded spatial spots

        Returns
        -------
        dict with ``"scores"`` [B, N] and ``"indices"`` [B, top_k]
        """
        t = F.normalize(text_emb, dim=-1)
        s = F.normalize(spot_embs, dim=-1)
        scores = t @ s.T  # [B, N]
        indices = scores.topk(top_k, dim=-1).indices
        return {"scores": scores, "indices": indices}


__all__ = [
    "ImageToGeneHead",
    "GeneToCellTypeHead",
    "GraphToDomainHead",
    "TextToSpatialRetrievalHead",
]
