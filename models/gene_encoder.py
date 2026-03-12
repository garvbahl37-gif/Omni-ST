"""
Omni-ST: Gene Expression Encoder
=================================
Transformer-based encoder for sparse high-dimensional gene expression vectors.
Models gene-gene co-expression dependencies via self-attention over gene tokens,
analogous to scBERT / Geneformer architectures.

Key design choices:
  - Each expressed gene is a token: (gene_id_embed + expression_embed)
  - Perceiver-style cross-attention compression to fixed-length output
  - Supports both full and top-k gene tokenisation strategies
  - Optional performer/linear-attention for O(N) complexity on large gene panels
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Gene Tokenisation
# ---------------------------------------------------------------------------

class GeneExpressionTokeniser(nn.Module):
    """
    Converts a dense expression vector into a sequence of gene tokens.

    Each gene ``i`` with expression ``v_i`` is mapped to:
        token_i = gene_emb[i] + expr_proj(v_i)

    Only expressed genes (value > ``expr_threshold``) are kept, with optional
    top-k selection to cap sequence length.

    Parameters
    ----------
    num_genes : int
        Total number of genes in the panel.
    embed_dim : int
        Token embedding dimension.
    expr_threshold : float
        Minimum expression value to include a gene token.
    max_genes : int | None
        If set, keep only the top-k highest-expressed genes.
    """

    def __init__(
        self,
        num_genes: int,
        embed_dim: int = 256,
        expr_threshold: float = 0.0,
        max_genes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.max_genes = max_genes
        self.threshold = expr_threshold

        # Learnable gene identity embeddings
        self.gene_embeddings = nn.Embedding(num_genes, embed_dim)

        # Expression value → embedding
        self.expr_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # CLS token for aggregate representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        nn.init.normal_(self.gene_embeddings.weight, std=0.02)

    def forward(
        self, expr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        expr : Tensor [B, G]  (batch × genes), log-normalised expression.

        Returns
        -------
        tokens : Tensor [B, T, D]   (T = num kept genes + 1 CLS)
        mask   : BoolTensor [B, T]  True = valid token.
        """
        B, G = expr.shape
        device = expr.device

        # Build gene-id indices
        gene_ids = torch.arange(G, device=device).unsqueeze(0).expand(B, -1)  # [B, G]

        # Gene identity embeddings [B, G, D]
        id_emb = self.gene_embeddings(gene_ids)

        # Expression embeddings [B, G, D]
        val_emb = self.expr_proj(expr.unsqueeze(-1))  # [B, G, 1] → [B, G, D]

        tokens_all = id_emb + val_emb  # [B, G, D]

        # Mask out unexpressed genes
        active_mask = expr > self.threshold  # [B, G]

        # Optional top-k selection
        if self.max_genes is not None and G > self.max_genes:
            _, topk_idx = torch.topk(expr, self.max_genes, dim=1)  # [B, k]
            sel_mask = torch.zeros(B, G, dtype=torch.bool, device=device)
            sel_mask.scatter_(1, topk_idx, True)
            active_mask = active_mask & sel_mask

        # Gather active tokens (pad with zeros, build attention mask)
        max_active = int(active_mask.sum(dim=1).max().item()) + 1  # +1 for CLS
        out_tokens = torch.zeros(B, max_active, tokens_all.size(-1), device=device)
        out_mask = torch.zeros(B, max_active, dtype=torch.bool, device=device)

        for b in range(B):
            active_idx = active_mask[b].nonzero(as_tuple=False).squeeze(-1)
            n = active_idx.size(0)
            out_tokens[b, 1 : n + 1] = tokens_all[b, active_idx]
            out_mask[b, : n + 1] = True

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        out_tokens[:, :1] = cls
        out_mask[:, 0] = True

        return out_tokens, out_mask  # [B, T, D], [B, T]


# ---------------------------------------------------------------------------
# Gene Transformer Block
# ---------------------------------------------------------------------------

class GeneTransformerBlock(nn.Module):
    """Standard Pre-LN Transformer block."""

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
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)
        x = residual + x
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Gene Expression Transformer Encoder
# ---------------------------------------------------------------------------

class GeneExpressionEncoder(nn.Module):
    """
    Full gene expression encoder:
      Tokenise → Transformer layers → project to ``output_dim``.

    Parameters
    ----------
    num_genes : int
    embed_dim : int
        Internal token dimension.
    output_dim : int
        Projected output dimension (shared latent space dim).
    num_layers : int
    num_heads : int
    max_genes : int | None
    dropout : float
    """

    def __init__(
        self,
        num_genes: int = 33538,
        embed_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_genes: Optional[int] = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.tokeniser = GeneExpressionTokeniser(
            num_genes=num_genes,
            embed_dim=embed_dim,
            max_genes=max_genes,
        )

        self.transformer = nn.ModuleList(
            [GeneTransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        self.output_dim = output_dim

    def forward(
        self, expr: torch.Tensor, return_all_tokens: bool = False
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        expr : Tensor [B, G]
        return_all_tokens : bool
            If True, return full token sequence [B, T, output_dim].

        Returns
        -------
        Tensor [B, output_dim] or [B, T, output_dim]
        """
        tokens, mask = self.tokeniser(expr)  # [B, T, D], [B, T]

        for block in self.transformer:
            tokens = block(tokens, key_padding_mask=mask)

        tokens = self.norm(tokens)  # [B, T, D]

        if return_all_tokens:
            return self.projection(tokens)

        cls = tokens[:, 0]  # [B, D]
        return self.projection(cls)  # [B, output_dim]


# ---------------------------------------------------------------------------
# Gene Expression Decoder (for gene prediction head)
# ---------------------------------------------------------------------------

class GeneExpressionDecoder(nn.Module):
    """
    Decodes a latent vector back into full gene expression predictions.

    Parameters
    ----------
    latent_dim : int
    num_genes : int
    hidden_dim : int
    """

    def __init__(self, latent_dim: int = 512, num_genes: int = 33538, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_genes),
            nn.Softplus(),  # non-negative expression
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """[B, latent_dim] → [B, num_genes]"""
        return self.decoder(latent)
