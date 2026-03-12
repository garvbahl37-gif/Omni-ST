"""
Omni-ST: Multimodal Fusion Backbone
=====================================
Unified cross-modal transformer that fuses embeddings from all modalities
(image, gene, graph, text) into a shared latent representation.

Architecture:
  1. Modality-specific linear projectors → unified token dim
  2. Modality type + order embeddings
  3. Stacked Cross-Modal Transformer blocks with bidirectional cross-attention
  4. Unified CLS token for task-conditioned global representation
  5. Task-specific output heads

Design inspired by:
  - Perceiver IO (Jaegle et al., 2021)
  - FLAVA (Singh et al., 2022)
  - BioViL-T (Bannur et al., 2023)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Modality Token Types
# ---------------------------------------------------------------------------

MODALITY_IDS = {
    "image": 0,
    "gene": 1,
    "graph": 2,
    "text": 3,
    "instruction": 4,
}

NUM_MODALITIES = len(MODALITY_IDS)


# ---------------------------------------------------------------------------
# Cross-Attention Block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention layer: query from modality A attends to key/value from modality B.
    Followed by self-attention within query modality and FFN.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_self = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_mult, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query   : [B, Tq, D]
        context : [B, Tc, D]  (keys + values)
        """
        # Cross-attention
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        ctx_key_padding = ~context_mask if context_mask is not None else None
        cross_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=ctx_key_padding)
        query = query + cross_out

        # Self-attention
        q = self.norm_self(query)
        q_key_padding = ~query_mask if query_mask is not None else None
        self_out, _ = self.self_attn(q, q, q, key_padding_mask=q_key_padding)
        query = query + self_out

        # FFN
        query = query + self.ff(self.norm_ff(query))
        return query


# ---------------------------------------------------------------------------
# Multimodal Transformer Block (all modalities jointly)
# ---------------------------------------------------------------------------

class MultimodalTransformerBlock(nn.Module):
    """
    Joint self-attention over all modality tokens concatenated together,
    followed by a per-token FFN.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_mult, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B, T_total, D]"""
        residual = x
        x = self.norm1(x)
        kpm = ~key_padding_mask if key_padding_mask is not None else None
        x, _ = self.attn(x, x, x, key_padding_mask=kpm)
        x = residual + x
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Multimodal Fusion Backbone
# ---------------------------------------------------------------------------

class MultimodalFusionBackbone(nn.Module):
    """
    Unified multimodal transformer backbone.

    Takes encoded embeddings from one or more modality encoders and fuses
    them through a stack of joint multimodal transformer layers with
    cross-attention interactions.

    Parameters
    ----------
    embed_dim : int
        Unified embedding dimension for all modality tokens.
    num_layers : int
        Number of joint multimodal transformer layers.
    num_heads : int
    modality_dims : dict[str, int]
        Map of modality name → encoder output dimension.
        Projectors will map each to ``embed_dim``.
    num_register_tokens : int
        Learnable register tokens prepended to the sequence (Vision Foundation Model trick).
    dropout : float
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        modality_dims: Optional[Dict[str, int]] = None,
        num_register_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        if modality_dims is None:
            modality_dims = {
                "image": 512,
                "gene": 512,
                "graph": 512,
                "text": 512,
                "instruction": 512,
            }

        # Modality projectors: map each encoder output dim → embed_dim
        self.modality_projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                )
                for name, dim in modality_dims.items()
            }
        )

        # Modality type embedding
        self.modality_type_emb = nn.Embedding(NUM_MODALITIES, embed_dim)

        # Global CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Register tokens
        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, embed_dim))
        self.num_register_tokens = num_register_tokens

        # Positional embedding (temporal position within each modality sequence)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, embed_dim))

        # Joint multimodal transformer
        self.transformer = nn.ModuleList(
            [MultimodalTransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _project_modality(
        self,
        embeddings: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """Project modality embedding to unified dim and add type embedding.

        Parameters
        ----------
        embeddings : [B, T, D_mod] or [B, D_mod]
        modality : str

        Returns
        -------
        Tensor [B, T, embed_dim]
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)  # [B, 1, D]
        B, T, _ = embeddings.shape

        projected = self.modality_projectors[modality](embeddings)  # [B, T, embed_dim]

        mod_id = MODALITY_IDS[modality]
        type_emb = self.modality_type_emb(
            torch.tensor(mod_id, device=embeddings.device)
        )  # [embed_dim]
        projected = projected + type_emb.unsqueeze(0).unsqueeze(0)

        # Add positional embeddings
        projected = projected + self.pos_embedding[:, :T]

        return projected

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        modality_inputs : dict
            Keys are modality names, values are encoded tensors.
            Each value: [B, T_mod, D_mod] or [B, D_mod]
        modality_masks : dict | None
            Optional boolean masks per modality (True = valid).

        Returns
        -------
        dict with:
          - ``"cls"`` : Tensor [B, embed_dim]  global representation
          - ``"tokens"`` : Tensor [B, T_total, embed_dim]  all tokens
          - ``"modality_cls"`` : dict[str, Tensor [B, embed_dim]]
        """
        B = next(iter(modality_inputs.values())).shape[0]
        device = next(iter(modality_inputs.values())).device

        # Project and collect all tokens
        all_tokens: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []

        # Prepend CLS + register tokens
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        reg = self.register_tokens.expand(B, -1, -1)  # [B, R, D]
        all_tokens.append(cls)
        all_tokens.append(reg)
        # Masks: all valid
        all_masks.append(torch.ones(B, 1, dtype=torch.bool, device=device))
        all_masks.append(torch.ones(B, self.num_register_tokens, dtype=torch.bool, device=device))

        modality_token_ranges: Dict[str, Tuple[int, int]] = {}
        offset = 1 + self.num_register_tokens

        for modality, emb in modality_inputs.items():
            projected = self._project_modality(emb, modality)  # [B, T, D]
            T = projected.size(1)
            all_tokens.append(projected)

            if modality_masks and modality in modality_masks:
                m = modality_masks[modality]
                if m.dim() == 1:
                    m = m.unsqueeze(1).expand(B, T)
            else:
                m = torch.ones(B, T, dtype=torch.bool, device=device)
            all_masks.append(m)

            modality_token_ranges[modality] = (offset, offset + T)
            offset += T

        # Concatenate all tokens
        x = torch.cat(all_tokens, dim=1)  # [B, T_total, D]
        mask = torch.cat(all_masks, dim=1)  # [B, T_total]

        # Run transformer
        for block in self.transformer:
            x = block(x, key_padding_mask=mask)

        x = self.norm(x)

        cls_out = x[:, 0]  # [B, D]

        # Extract per-modality CLS (mean of modality tokens)
        modality_cls: Dict[str, torch.Tensor] = {}
        for modality, (start, end) in modality_token_ranges.items():
            modality_cls[modality] = x[:, start:end].mean(dim=1)

        return {
            "cls": cls_out,
            "tokens": x,
            "modality_cls": modality_cls,
        }
