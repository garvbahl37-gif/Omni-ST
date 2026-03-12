"""
Omni-ST: Image Encoder
======================
Vision Transformer (ViT) and Swin Transformer based encoder for
histology patches (H&E / Immunofluorescence).

Supports:
  - Pretrained ViT-B/16, ViT-L/16, ViT-H/14 from timm/HuggingFace
  - Hierarchical Swin Transformer for multi-scale feature extraction
  - Patch-level and region-level embeddings
  - Configurable output projection dimension
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import ViTModel, ViTConfig


# ---------------------------------------------------------------------------
# Patch Embedding Utilities
# ---------------------------------------------------------------------------

class PatchPositionEmbedding(nn.Module):
    """Learnable 2-D sinusoidal position embedding for image patches."""

    def __init__(self, num_patches: int, embed_dim: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Core ViT Image Encoder
# ---------------------------------------------------------------------------

class ViTImageEncoder(nn.Module):
    """
    Vision Transformer encoder for histology image patches.

    Parameters
    ----------
    model_name : str
        timm model name, e.g. ``"vit_base_patch16_224"`` or
        ``"swin_base_patch4_window7_224"``.
    pretrained : bool
        Load ImageNet / pathology pretrained weights.
    output_dim : int
        Dimension of the projected output embedding.
    freeze_backbone : bool
        If ``True``, freeze all ViT parameters and only train the projection.
    dropout : float
        Dropout rate on the projected output.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        output_dim: int = 512,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        hidden_dim = self.backbone.num_features

        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.output_dim = output_dim

    def forward(
        self, pixel_values: torch.Tensor, return_patch_tokens: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        pixel_values : Tensor [B, C, H, W]
        return_patch_tokens : bool
            If ``True``, also return the per-patch token sequence.

        Returns
        -------
        cls_embed : Tensor [B, output_dim]
        patch_tokens : Tensor [B, N, output_dim]  (only if ``return_patch_tokens``)
        """
        features = self.backbone.forward_features(pixel_values)  # [B, N+1, D] or [B, D]

        # Handle both ViT (sequence) and CNN-style (flat) outputs
        if features.dim() == 3:
            cls_token = features[:, 0]  # [B, D]
            patch_tokens = features[:, 1:]  # [B, N, D]
        else:
            cls_token = features
            patch_tokens = features.unsqueeze(1)

        cls_embed = self.projection(cls_token)  # [B, output_dim]

        if return_patch_tokens:
            patch_embeds = self.projection(patch_tokens)  # [B, N, output_dim]
            return cls_embed, patch_embeds

        return cls_embed


# ---------------------------------------------------------------------------
# Multi-Scale Swin Transformer Encoder
# ---------------------------------------------------------------------------

class SwinImageEncoder(nn.Module):
    """
    Hierarchical Swin Transformer encoder that provides multi-scale
    feature maps suitable for dense spatial tasks.

    Parameters
    ----------
    model_name : str
        timm swin model identifier.
    pretrained : bool
    output_dim : int
    scales : list[int]
        Which Swin stages to use (0-indexed). Features are pooled
        and concatenated.
    """

    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        output_dim: int = 512,
        scales: Optional[list] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        scales = scales or [1, 2, 3]

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, out_indices=scales
        )
        feature_dims = [info["num_chs"] for info in self.backbone.feature_info]
        total_dim = sum(feature_dims)

        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] → [B, output_dim]"""
        feature_maps = self.backbone(pixel_values)
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feature_maps]
        concat = torch.cat(pooled, dim=-1)  # [B, total_dim]
        return self.projection(concat)  # [B, output_dim]


# ---------------------------------------------------------------------------
# Unified Histology Encoder (factory)
# ---------------------------------------------------------------------------

class HistologyEncoder(nn.Module):
    """
    Factory wrapper that selects ViT or Swin encoder based on ``arch``.

    Parameters
    ----------
    arch : str
        ``"vit"`` or ``"swin"``.
    **kwargs
        Passed directly to the underlying encoder.
    """

    _ENCODERS = {
        "vit": ViTImageEncoder,
        "swin": SwinImageEncoder,
    }

    def __init__(self, arch: str = "vit", **kwargs) -> None:
        super().__init__()
        if arch not in self._ENCODERS:
            raise ValueError(f"Unknown arch '{arch}'. Choose from {list(self._ENCODERS)}")
        self.encoder = self._ENCODERS[arch](**kwargs)
        self.output_dim = self.encoder.output_dim

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encoder(pixel_values, **kwargs)


# ---------------------------------------------------------------------------
# Patch Aggregation with Cross-Attention (for multi-patch ROI encoding)
# ---------------------------------------------------------------------------

class MultiPatchAggregator(nn.Module):
    """
    Aggregates variable-length patch sequences into a single region embedding
    using multi-head cross-attention with a learnable query.

    Parameters
    ----------
    embed_dim : int
    num_heads : int
    num_queries : int
        Number of learnable output queries (set to 1 for a single ROI vector).
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, num_queries: int = 1) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patch_tokens : Tensor [B, N, D]

        Returns
        -------
        Tensor [B, num_queries, D]
        """
        B = patch_tokens.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.attn(q, patch_tokens, patch_tokens)
        return self.norm(out)
