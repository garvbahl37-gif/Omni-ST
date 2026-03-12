"""
Omni-ST: Text Encoder
======================
Biomedical language model encoder based on BioBERT / SciBERT.
Encodes natural language instructions and biological descriptions into
the shared multimodal latent space.

Features:
  - Supports HuggingFace Bio_ClinicalBERT, allenai/scibert, dmis-lab/biobert-v1.1
  - CLS token extraction with optional mean/max pooling
  - Biomedical domain-adaptive tokenisation
  - Optional LoRA / prefix-tuning for parameter-efficient adaptation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertModel, BertConfig


# ---------------------------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------------------------

class BiomedicalTextEncoder(nn.Module):
    """
    Encodes text using a biomedical pre-trained language model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
        Default: ``"dmis-lab/biobert-base-cased-v1.2"``
    output_dim : int
        Projected output dimension for the shared latent space.
    pool_strategy : str
        ``"cls"`` | ``"mean"`` | ``"max"``
    freeze_backbone : bool
        Freeze all BERT parameters (train only projection).
    max_length : int
        Maximum token sequence length.
    dropout : float
    """

    SUPPORTED_MODELS = {
        "biobert": "dmis-lab/biobert-base-cased-v1.2",
        "scibert": "allenai/scibert_scivocab_uncased",
        "bioclinical": "emilyalsentzer/Bio_ClinicalBERT",
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    }

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.2",
        output_dim: int = 512,
        pool_strategy: str = "cls",
        freeze_backbone: bool = False,
        max_length: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Allow shorthand names
        if model_name in self.SUPPORTED_MODELS:
            model_name = self.SUPPORTED_MODELS[model_name]

        self.model_name = model_name
        self.max_length = max_length
        self.pool_strategy = pool_strategy

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_dim = self.bert.config.hidden_size

        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    @torch.no_grad()
    def get_tokenizer(self) -> AutoTokenizer:
        """Return the associated tokenizer (cached)."""
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def tokenize(
        self,
        texts: Union[str, List[str]],
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """Convenience method: tokenise text strings into model inputs."""
        tokenizer = self.get_tokenizer()
        if isinstance(texts, str):
            texts = [texts]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token representations → [B, hidden_dim]."""
        if self.pool_strategy == "cls":
            return last_hidden[:, 0]
        elif self.pool_strategy == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:  # max
            mask = attention_mask.unsqueeze(-1).bool()
            last_hidden = last_hidden.masked_fill(~mask, float("-inf"))
            return last_hidden.max(dim=1).values

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids, attention_mask, token_type_ids : pre-tokenised inputs
        texts : raw strings (auto-tokenised on the fly)
        return_all_tokens : bool
            If True, return full token sequence [B, T, output_dim].

        Returns
        -------
        Tensor [B, output_dim] or [B, T, output_dim]
        """
        if texts is not None:
            device = next(self.parameters()).device
            enc = self.tokenize(texts, device=device)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            token_type_ids = enc.get("token_type_ids")

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        last_hidden = self.dropout(last_hidden)

        if return_all_tokens:
            return self.norm(self.projection(last_hidden))

        pooled = self._pool(last_hidden, attention_mask)
        return self.norm(self.projection(pooled))


# ---------------------------------------------------------------------------
# Instruction Tokeniser Helper
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATES = {
    "image_to_gene": (
        "Given a histology patch from spatial transcriptomics, "
        "predict the gene expression profile of the tissue region."
    ),
    "gene_to_celltype": (
        "Classify the cell type based on the following gene expression vector."
    ),
    "graph_to_domain": (
        "Segment the tissue into spatial domains based on the spatial gene expression graph."
    ),
    "region_to_text": (
        "Describe the biological characteristics of the highlighted tissue region."
    ),
    "text_to_spatial": (
        "Retrieve the spatial locations in the tissue that match the following biological description."
    ),
}


def build_instruction(task: str, context: str = "") -> str:
    """
    Build a natural language instruction string for a given task.

    Parameters
    ----------
    task : str  task key from INSTRUCTION_TEMPLATES
    context : str  optional additional context appended to the template
    """
    template = INSTRUCTION_TEMPLATES.get(task, "Perform the spatial transcriptomics task.")
    if context:
        return f"{template} Context: {context}"
    return template
