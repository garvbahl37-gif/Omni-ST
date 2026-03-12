"""
Omni-ST: 10x Genomics Visium Dataset
======================================
PyTorch Dataset wrapping AnnData objects from Visium spatial transcriptomics.

Supports:
  - spot-level gene expression + histology patch paired loading
  - spatial coordinate extraction
  - kNN spatial graph construction on-the-fly
  - instruction-conditioned task loading
  - multi-task yield format: (image_patch, gene_expr, coords, instruction, label)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset
from PIL import Image


class VisiumDataset(Dataset):
    """
    Dataset for 10x Genomics Visium spatial transcriptomics.

    Parameters
    ----------
    adata : AnnData | str
        Loaded AnnData object or path to .h5ad file.
    task : str
        Task key defining what (input, output) pairs to yield.
        One of: ``image_to_gene``, ``gene_to_celltype``,
        ``graph_to_domain``, ``region_to_text``.
    image_dir : str | None
        Directory containing histology image tiles (PNG/TIFF).
        If None, images are loaded from ``adata.uns["spatial"]``.
    gene_list : list[str] | None
        Subset of genes to include. If None, use all HVGs.
    patch_size : int
        Spatial patch size in pixels.
    transform : callable | None
        Optional torchvision transform for image patches.
    max_spots : int | None
        Subsample to at most this many spots (for fast debugging).
    instruction : str | None
        If provided, override the instruction string for all samples.
    """

    TASK_INSTRUCTIONS = {
        "image_to_gene": (
            "Given the histology patch from a Visium spot, predict the gene expression profile."
        ),
        "gene_to_celltype": (
            "Based on the gene expression vector, classify the cell type of this spatial spot."
        ),
        "graph_to_domain": (
            "Identify the spatial transcriptomics domain this spot belongs to."
        ),
        "region_to_text": (
            "Describe the biological characteristics of this tissue region."
        ),
    }

    def __init__(
        self,
        adata: Union[ad.AnnData, str, Path],
        task: str = "image_to_gene",
        image_dir: Optional[str] = None,
        gene_list: Optional[List[str]] = None,
        patch_size: int = 224,
        transform: Optional[Callable] = None,
        max_spots: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> None:
        super().__init__()

        if isinstance(adata, (str, Path)):
            adata = ad.read_h5ad(adata)
        self.adata = adata
        self.task = task
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.transform = transform

        # Subselect genes
        if gene_list is not None:
            available = [g for g in gene_list if g in self.adata.var_names]
            self.adata = self.adata[:, available]
        self.gene_list = list(self.adata.var_names)

        # Subselect spots
        if max_spots is not None:
            idx = np.random.choice(len(self.adata), min(max_spots, len(self.adata)), replace=False)
            self.adata = self.adata[idx]

        self.instruction_text = instruction or self.TASK_INSTRUCTIONS.get(task, "")
        self._validate()

    def _validate(self) -> None:
        assert "spatial" in self.adata.obsm, (
            "AnnData must contain 'spatial' in obsm for spatial coordinates."
        )

    def __len__(self) -> int:
        return len(self.adata)

    def _load_patch(self, idx: int) -> Optional[torch.Tensor]:
        """Load the histology image patch for spot ``idx``."""
        # Try to load from library crop
        try:
            library_id = list(self.adata.uns["spatial"].keys())[0]
            full_image = self.adata.uns["spatial"][library_id]["images"]["hires"]  # numpy [H, W, 3]
            coords = self.adata.obsm["spatial"][idx]  # (x, y) pixel coords
            scale = self.adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
            cx, cy = int(coords[0] * scale), int(coords[1] * scale)
            half = self.patch_size // 2
            patch = full_image[
                max(cy - half, 0): cy + half,
                max(cx - half, 0): cx + half,
            ]
            patch = Image.fromarray((patch * 255).astype(np.uint8) if patch.max() <= 1 else patch.astype(np.uint8))
            patch = patch.resize((self.patch_size, self.patch_size))
        except Exception:
            patch = Image.fromarray(np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8))

        if self.transform:
            return self.transform(patch)
        return torch.tensor(np.array(patch), dtype=torch.float32).permute(2, 0, 1) / 255.0

    def _get_expression(self, idx: int) -> torch.Tensor:
        """Return log-normalised expression vector [G]."""
        expr = self.adata.X[idx]
        if hasattr(expr, "toarray"):
            expr = expr.toarray().squeeze()
        return torch.tensor(expr, dtype=torch.float32)

    def _get_label(self, idx: int) -> Optional[torch.Tensor]:
        """Return the ground-truth label depending on task."""
        if self.task == "gene_to_celltype":
            if "cell_type" in self.adata.obs.columns:
                ct = self.adata.obs["cell_type"].iloc[idx]
                cats = self.adata.obs["cell_type"].cat.categories if hasattr(
                    self.adata.obs["cell_type"], "cat"
                ) else sorted(self.adata.obs["cell_type"].unique())
                return torch.tensor(list(cats).index(ct), dtype=torch.long)
        elif self.task == "graph_to_domain":
            if "domain" in self.adata.obs.columns:
                d = self.adata.obs["domain"].iloc[idx]
                domains = sorted(self.adata.obs["domain"].unique())
                return torch.tensor(domains.index(d), dtype=torch.long)
        return None

    def __getitem__(self, idx: int) -> Dict:
        item: Dict = {"instruction": self.instruction_text, "spot_idx": idx}

        # Always include expression and spatial coords
        item["gene_expr"] = self._get_expression(idx)
        coords = self.adata.obsm["spatial"][idx]
        item["spatial_coords"] = torch.tensor(coords, dtype=torch.float32)

        # Task-specific additions
        if self.task in ("image_to_gene", "region_to_text"):
            item["image"] = self._load_patch(idx)

        label = self._get_label(idx)
        if label is not None:
            item["label"] = label

        return item


def create_visium_dataloaders(
    adata: Union[ad.AnnData, str],
    task: str = "image_to_gene",
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    **dataset_kwargs,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Split AnnData into train/val/test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    from torch.utils.data import random_split, DataLoader

    dataset = VisiumDataset(adata, task=task, **dataset_kwargs)
    n = len(dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    def collate(batch):
        out = {}
        for key in batch[0]:
            vals = [b[key] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                out[key] = torch.stack(vals)
            else:
                out[key] = vals
        return out

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate, pin_memory=True)
    return (
        torch.utils.data.DataLoader(train_ds, shuffle=True, **dl_kwargs),
        torch.utils.data.DataLoader(val_ds, shuffle=False, **dl_kwargs),
        torch.utils.data.DataLoader(test_ds, shuffle=False, **dl_kwargs),
    )
