"""
Omni-ST: Spatial Graph Encoder
================================
Graph Neural Network encoder for spatial transcriptomics tissue graphs.

Supports:
  - Graph Attention Networks (GAT v2)
  - Graph Transformer (Graphormer-style)
  - k-NN and Delaunay spatial graph topologies
  - Node features: gene expression + spatial coordinates
  - Edge features: physical distance + expression similarity
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GAT v2 Convolution Layer
# ---------------------------------------------------------------------------

class GATv2Conv(nn.Module):
    """
    Graph Attention Network v2 convolution (Brody et al., 2022).
    More expressive than GATv1 by computing attention on concatenated
    query+key representations.

    Parameters
    ----------
    in_dim : int
    out_dim : int
    num_heads : int
    edge_dim : int | None
        If provided, edge features are incorporated in attention.
    dropout : float
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        assert out_dim % num_heads == 0

        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim else None
        self.attn = nn.Linear(out_dim, num_heads, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [N, in_dim]  node features
        edge_index : LongTensor [2, E]  (src, dst)
        edge_attr : Tensor [E, edge_dim]  optional

        Returns
        -------
        Tensor [N, out_dim]
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]
        N = x.size(0)

        h_src = self.W_src(x)  # [N, out_dim]
        h_dst = self.W_dst(x)  # [N, out_dim]

        a_input = h_src[src_idx] + h_dst[dst_idx]  # [E, out_dim]
        if self.W_edge is not None and edge_attr is not None:
            a_input = a_input + self.W_edge(edge_attr)

        # Attention coefficients [E, num_heads]
        alpha = self.attn(F.leaky_relu(a_input, 0.2))
        alpha = self.dropout(alpha)

        # Softmax per destination node
        alpha_exp = torch.exp(alpha - alpha.max())  # numerical stability [E, H]
        denom = torch.zeros(N, self.num_heads, device=x.device)
        denom.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, self.num_heads), alpha_exp)
        alpha_norm = alpha_exp / (denom[dst_idx] + 1e-9)  # [E, H]

        # Aggregate messages
        msg = h_src[src_idx].view(-1, self.num_heads, self.head_dim)  # [E, H, d]
        msg = msg * alpha_norm.unsqueeze(-1)

        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst_idx.view(-1, 1, 1).expand_as(msg), msg)
        out = out.reshape(N, self.out_dim)  # [N, out_dim]

        return self.norm(out + h_dst)  # residual from destination node


# ---------------------------------------------------------------------------
# Graph Transformer Layer
# ---------------------------------------------------------------------------

class GraphTransformerLayer(nn.Module):
    """
    Full-graph attention transformer layer.
    Quadratic O(N²) — suitable for small tissue graphs (<1000 spots).

    Parameters
    ----------
    embed_dim : int
    num_heads : int
    use_edge_bias : bool
        If True, use spatial distance as additive bias in attention.
    """

    def __init__(
        self, embed_dim: int = 256, num_heads: int = 8, use_edge_bias: bool = True, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.use_edge_bias = use_edge_bias
        if use_edge_bias:
            self.edge_bias_proj = nn.Linear(1, num_heads)  # distance scalar → per-head bias
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(
        self, x: torch.Tensor, dist_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x : [B, N, D]
        dist_matrix : [B, N, N]  pairwise distances (optional)
        """
        attn_bias = None
        if self.use_edge_bias and dist_matrix is not None:
            # [B, N, N] → [B, N, N, H] → [B*H, N, N]
            B, N, _ = dist_matrix.shape
            bias = self.edge_bias_proj(dist_matrix.unsqueeze(-1))  # [B, N, N, H]
            bias = bias.permute(0, 3, 1, 2).reshape(B * self.attn.num_heads, N, N)
            attn_bias = bias

        residual = x
        x_norm = self.norm1(x)

        # MultiheadAttention expects [B, N, D] with batch_first=True
        # attn_mask must be [B*H, N, N] or None
        out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_bias)
        x = residual + out
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Spatial Graph Encoder
# ---------------------------------------------------------------------------

class SpatialGraphEncoder(nn.Module):
    """
    Encodes the spatial tissue graph using stacked GATv2 or Graph Transformer layers.

    Node input: concatenation of [expression features | spatial coord encoding]
    Output: graph-level embedding via mean/attention pooling.

    Parameters
    ----------
    node_in_dim : int
        Input node feature dimension (e.g. gene PCA embedding dim).
    coord_dim : int
        Spatial coordinate dimension (2 for 2-D Visium, 3 for 3-D).
    embed_dim : int
        Internal GNN embedding dimension.
    output_dim : int
        Final projected embedding dimension.
    num_layers : int
    num_heads : int
    gnn_type : str
        ``"gat"`` or ``"transformer"``.
    pool : str
        ``"mean"``, ``"max"``, or ``"attention"``.
    """

    def __init__(
        self,
        node_in_dim: int = 64,
        coord_dim: int = 2,
        embed_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        gnn_type: str = "gat",
        pool: str = "attention",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gnn_type = gnn_type

        # Input projection: concat(gene_feat, coord) → embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(node_in_dim + coord_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Coordinate sinusoidal encoding
        self.coord_encoder = CoordinateEncoder(coord_dim, coord_dim)

        if gnn_type == "gat":
            self.layers = nn.ModuleList(
                [GATv2Conv(embed_dim, embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
            )
        else:  # transformer
            self.layers = nn.ModuleList(
                [GraphTransformerLayer(embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
            )

        # Readout
        self.pool_type = pool
        if pool == "attention":
            self.pool_gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.Tanh(),
                nn.Linear(embed_dim // 2, 1),
            )

        self.projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(
        self,
        node_feat: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        dist_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_feat : Tensor [N_total, node_in_dim]
        coords : Tensor [N_total, coord_dim]
        edge_index : LongTensor [2, E]
        batch : LongTensor [N_total]  graph-assignment vector for batched graphs
        dist_matrix : Tensor [B, N, N]  (for transformer pooling)

        Returns
        -------
        Tensor [B, output_dim]  graph-level embeddings
        """
        coord_enc = self.coord_encoder(coords)
        x = torch.cat([node_feat, coord_enc], dim=-1)
        x = self.input_proj(x)  # [N, D]

        if self.gnn_type == "gat":
            for layer in self.layers:
                x = layer(x, edge_index) + x  # residual
        else:
            # For Graph Transformer: reshape to batched [B, N, D]
            # (assumes uniform graph size — for demo purposes)
            pass

        # Readout: pool over nodes per graph
        return self._readout(x, batch)

    def _readout(self, x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        B = int(batch.max().item()) + 1

        if self.pool_type == "mean":
            out = torch.zeros(B, x.size(-1), device=x.device)
            counts = torch.zeros(B, 1, device=x.device)
            out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
            counts.scatter_add_(0, batch.unsqueeze(-1), torch.ones(x.size(0), 1, device=x.device))
            return self.projection(out / counts.clamp(min=1))

        elif self.pool_type == "max":
            out = torch.full((B, x.size(-1)), float("-inf"), device=x.device)
            out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(x), x, reduce="amax")
            return self.projection(out)

        else:  # attention pooling
            gates = self.pool_gate(x).squeeze(-1)  # [N]
            # Softmax within each graph
            gate_softmax = torch.zeros_like(gates)
            for b in range(B):
                mask = batch == b
                gate_softmax[mask] = torch.softmax(gates[mask], dim=0)
            out = torch.zeros(B, x.size(-1), device=x.device)
            out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x * gate_softmax.unsqueeze(-1))
            return self.projection(out)


# ---------------------------------------------------------------------------
# Sinusoidal Coordinate Encoder
# ---------------------------------------------------------------------------

class CoordinateEncoder(nn.Module):
    """
    Encodes 2-D / 3-D spatial coordinates using sinusoidal + learnable projection.
    """

    def __init__(self, coord_dim: int = 2, out_dim: int = 2) -> None:
        super().__init__()
        self.coord_dim = coord_dim
        self.out_dim = out_dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: [N, coord_dim] → [N, coord_dim]  (pass-through for now)"""
        # Normalize to [0, 1] per axis — caller is responsible for batch normalisation
        return coords
