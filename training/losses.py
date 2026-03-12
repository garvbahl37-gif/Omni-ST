"""
Omni-ST: Contrastive Losses for Stage 2 Cross-Modal Alignment
===================================================================
Implements CLIP-style symmetric InfoNCE loss and other contrastive
objectives for aligning image, gene, graph, and text embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPContrastiveLoss(nn.Module):
    """
    Symmetric InfoNCE contrastive loss (CLIP-style).
    Aligns two modality embedding sequences within a batch.

    Parameters
    ----------
    temperature : float | nn.Parameter
        Softmax temperature. Can be learnable.
    learnable_temp : bool
        If True, treat temperature as a learnable parameter clamped to [0.07, 100].
    """

    def __init__(self, temperature: float = 0.07, learnable_temp: bool = True) -> None:
        super().__init__()
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.01, 100.0)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        emb_a : Tensor [B, D]  normalized embeddings from modality A
        emb_b : Tensor [B, D]  normalized embeddings from modality B

        Returns
        -------
        Scalar loss
        """
        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)

        logits = (emb_a @ emb_b.T) / self.temperature  # [B, B]
        targets = torch.arange(logits.size(0), device=logits.device)

        loss_a = F.cross_entropy(logits, targets)
        loss_b = F.cross_entropy(logits.T, targets)
        return (loss_a + loss_b) / 2.0


class MultiModalAlignmentLoss(nn.Module):
    """
    Aligns all pairs of modality embeddings jointly.

    Parameters
    ----------
    modalities : list[str]  e.g. ["image", "gene", "graph", "text"]
    temperature : float
    pair_weights : dict | None  per-pair weight overrides
    """

    def __init__(
        self,
        modalities: list = None,
        temperature: float = 0.07,
        pair_weights: dict = None,
    ) -> None:
        super().__init__()
        modalities = modalities or ["image", "gene", "graph", "text"]
        self.modalities = modalities
        self.clip_loss = CLIPContrastiveLoss(temperature=temperature)
        self.pair_weights = pair_weights or {}

    def forward(self, embeddings: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : dict[str, Tensor [B, D]]

        Returns
        -------
        Scalar alignment loss
        """
        total, count = 0.0, 0
        mods = [m for m in self.modalities if m in embeddings]
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                a, b = mods[i], mods[j]
                w = self.pair_weights.get(f"{a}_{b}", 1.0)
                total += w * self.clip_loss(embeddings[a], embeddings[b])
                count += w
        return total / max(count, 1)


class ReconstructionLoss(nn.Module):
    """Weighted MSE + Cosine loss for gene expression reconstruction."""

    def __init__(self, mse_weight: float = 1.0, cosine_weight: float = 0.1) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_w = mse_weight
        self.cos_w = cosine_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        cos_loss = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        return self.mse_w * mse_loss + self.cos_w * cos_loss
