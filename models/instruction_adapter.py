"""
Omni-ST: Instruction Conditioning Adapter
==========================================
Conditions the multimodal backbone on natural language instructions
using prefix tokens and lightweight adapter layers.

Strategies implemented:
  1. Prefix Token Conditioning — prepend learnable prefix tokens modulated
     by the instruction embedding (GPT-Prefix style)
  2. Cross-Attention Adapter — inject instruction signal via cross-attention
     into each transformer block (LLaMA-Adapter style)
  3. LoRA-style Adapter — low-rank adaptation of backbone attention layers

The InstructionAdapter is task-agnostic: the same adapter processes any
natural language instruction and steers the shared backbone.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Prefix Token Generator
# ---------------------------------------------------------------------------

class PrefixTokenGenerator(nn.Module):
    """
    Generates instruction-conditioned soft prefix tokens.

    Parameters
    ----------
    instruction_dim : int
        Dimension of the instruction embedding (from text encoder).
    embed_dim : int
        Backbone embedding dimension.
    num_prefix_tokens : int
        Number of prefix tokens to prepend.
    num_layers : int
        Number of backbone layers to supply prefix tokens for.
    """

    def __init__(
        self,
        instruction_dim: int = 512,
        embed_dim: int = 512,
        num_prefix_tokens: int = 8,
        num_layers: int = 8,
    ) -> None:
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # Maps instruction embedding → flat prefix tokens for all layers
        self.prefix_proj = nn.Sequential(
            nn.Linear(instruction_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_layers * num_prefix_tokens * embed_dim),
        )

    def forward(self, instruction_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        instruction_emb : Tensor [B, instruction_dim]

        Returns
        -------
        prefix_tokens : Tensor [num_layers, B, num_prefix_tokens, embed_dim]
        """
        B = instruction_emb.size(0)
        flat = self.prefix_proj(instruction_emb)  # [B, L * P * D]
        prefix = flat.view(B, self.num_layers, self.num_prefix_tokens, self.embed_dim)
        prefix = prefix.permute(1, 0, 2, 3)  # [L, B, P, D]
        return prefix


# ---------------------------------------------------------------------------
# Cross-Attention Adapter Layer (LLaMA-Adapter style)
# ---------------------------------------------------------------------------

class CrossAttentionAdapterLayer(nn.Module):
    """
    Lightweight cross-attention adapter injected between transformer layers.
    The main residual stream is augmented with a cross-attention signal
    from the instruction embedding.

    Parameters
    ----------
    embed_dim : int
    instruction_dim : int
    num_heads : int
    gate_init : float
        Initial value of the tanh gating scalar. Starting near 0 prevents
        destabilisation of pretrained weights at the start of fine-tuning.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        instruction_dim: int = 512,
        num_heads: int = 8,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.instruction_proj = nn.Linear(instruction_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(
        self,
        x: torch.Tensor,
        instruction_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        x : [B, T, D]
        instruction_emb : [B, instruction_dim] or [B, Ti, instruction_dim]
        """
        if instruction_emb.dim() == 2:
            instruction_emb = instruction_emb.unsqueeze(1)  # [B, 1, instruction_dim]
        kv = self.instruction_proj(instruction_emb)  # [B, Ti, D]

        attn_out, _ = self.cross_attn(self.norm(x), kv, kv)
        return x + torch.tanh(self.gate) * attn_out


# ---------------------------------------------------------------------------
# LoRA Linear Layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation of a linear layer (Hu et al., 2021).

    Replaces W·x with (W + α/r · B·A)·x where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}.

    Parameters
    ----------
    original_layer : nn.Linear
    rank : int
    alpha : float  scaling factor
    """

    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        d_in = original_layer.in_features
        d_out = original_layer.out_features
        self.original = original_layer
        self.rank = rank
        self.scale = alpha / rank

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.scale * self.lora_B(self.lora_A(x))


import math  # noqa: E402


# ---------------------------------------------------------------------------
# Main Instruction Adapter
# ---------------------------------------------------------------------------

class InstructionAdapter(nn.Module):
    """
    Complete instruction conditioning adapter.

    Supports three strategies, controllable via ``strategy``:
      - ``"prefix"`` : Prefix token generation (fast, minimal params)
      - ``"cross_attn"`` : Cross-attention adapter per backbone layer
      - ``"lora"`` : LoRA on backbone's self-attention projections

    Parameters
    ----------
    instruction_dim : int
    embed_dim : int
    num_backbone_layers : int
    num_heads : int
    strategy : str
    num_prefix_tokens : int
    lora_rank : int
    """

    def __init__(
        self,
        instruction_dim: int = 512,
        embed_dim: int = 512,
        num_backbone_layers: int = 8,
        num_heads: int = 8,
        strategy: str = "cross_attn",
        num_prefix_tokens: int = 8,
        lora_rank: int = 8,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.embed_dim = embed_dim

        if strategy == "prefix":
            self.prefix_generator = PrefixTokenGenerator(
                instruction_dim=instruction_dim,
                embed_dim=embed_dim,
                num_prefix_tokens=num_prefix_tokens,
                num_layers=num_backbone_layers,
            )

        elif strategy == "cross_attn":
            self.adapter_layers = nn.ModuleList(
                [
                    CrossAttentionAdapterLayer(
                        embed_dim=embed_dim,
                        instruction_dim=instruction_dim,
                        num_heads=num_heads,
                    )
                    for _ in range(num_backbone_layers)
                ]
            )

        # LoRA layers must be applied directly to backbone in trainer

    def get_prefix_tokens(self, instruction_emb: torch.Tensor) -> Optional[torch.Tensor]:
        """For prefix strategy: [L, B, P, D]"""
        if self.strategy == "prefix":
            return self.prefix_generator(instruction_emb)
        return None

    def apply_to_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        instruction_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Apply instruction adapter to a specific backbone layer's output."""
        if self.strategy == "cross_attn":
            return self.adapter_layers[layer_idx](hidden_states, instruction_emb)
        return hidden_states

    def apply_prefix(
        self,
        tokens: torch.Tensor,
        instruction_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Prepend generated prefix tokens to the token sequence."""
        if self.strategy == "prefix":
            B = tokens.size(0)
            prefix = self.prefix_generator(instruction_emb)[0]  # [B, P, D]
            return torch.cat([prefix, tokens], dim=1)
        return tokens
